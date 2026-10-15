<!--
Copyright © Advanced Micro Devices, Inc., or its affiliates.

SPDX-License-Identifier: MIT
-->

# Run standalone fine-tuning on Kubernetes

This guide deploys a supervised fine-tuning (SFT) job with Kubernetes manifests

It has the following setup:

- A model PVC and a dataset PVC are mounted read-only at `/model` and `/data`.
- An ephemeral PVC is mounted at `/output` for checkpoints and rendered runtime
  files.
- An in-memory `emptyDir` is mounted at `/dev/shm`.
- An optional sidecar copies the final adapter to a persistent export PVC.

This workflow assumes that the model and dataset have already been downloaded
to PVCs.

## Prerequisites

You need access to a Kubernetes cluster with AMD GPU resources and a storage
class that supports the requested volumes. The model and dataset must already
be available in persistent volume claims.

Verify the input claims before creating the job:

```bash
kubectl get pvc
```

Record the input claim names and paths. The example below assumes:

- Model claim: `aimft-dev-model-google-gemma-3-27b-it`
- Dataset claim: `aimft-dev-dataset-sft-demo-data`
- Dataset path: `/data/sft-demo-data.jsonl`

## Deploy the training job

Save the following manifest as `standalone-sft.yaml`. Replace the image,
claim names, model ID, dataset path, storage class, and resource values for your
environment before applying it.

```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: standalone-sft
spec:
  ttlSecondsAfterFinished: 3600
  backoffLimit: 0
  template:
    spec:
      restartPolicy: Never
      containers:
        - name: sft
          image: docker.io/amdenterpriseai/aimft-aimfttk:YOUR_TAG
          imagePullPolicy: Always
          resources:
            requests:
              amd.com/gpu: "1"
              cpu: "8"
              memory: "64Gi"
            limits:
              amd.com/gpu: "1"
              cpu: "8"
              memory: "64Gi"
          env:
            - name: AIMFT_MODEL_PATH
              value: /model/google/gemma-3-27b-it
            - name: AIMFT_OUTPUT_PATH
              value: /output
            - name: AIMFT_TRAIN_DATA_PATH
              value: /data/sft-demo-data.jsonl
            - name: AIM_MODEL_ID
              value: google/gemma-3-27b-it
            - name: AIMFT_USE_ADAPTER
              value: "true"
          volumeMounts:
            - name: model
              mountPath: /model
              readOnly: true
            - name: data
              mountPath: /data
              readOnly: true
            - name: output
              mountPath: /output
            - name: dshm
              mountPath: /dev/shm
      volumes:
        - name: model
          persistentVolumeClaim:
            claimName: aimft-dev-model-google-gemma-3-27b-it
            readOnly: true
        - name: data
          persistentVolumeClaim:
            claimName: aimft-dev-dataset-sft-demo-data
            readOnly: true
        - name: output
          ephemeral:
            volumeClaimTemplate:
              spec:
                accessModes: [ReadWriteOnce]
                storageClassName: mlstorage
                resources:
                  requests:
                    storage: 100Gi
        - name: dshm
          emptyDir:
            medium: Memory
            sizeLimit: 12Gi
```

Apply the manifest and follow the job:

```bash
kubectl apply -f standalone-sft.yaml
kubectl get pods -l job-name=standalone-sft -w
kubectl logs job/standalone-sft -f
```

Size the output volume for the selected model and training method. Full-parameter checkpoints can require substantially more space. Omit `storageClassName` to use the cluster's default storage class.

The ephemeral output claim is deleted with the Job's pod. Export the adapter if
you need to retain it after the job finishes.

## Export the final adapter

Add the following resources before the `Job` in `standalone-sft.yaml` to persist
`/output/checkpoint-final-adapter` in a shared PVC. The export sidecar watches
`/output/runtime/train-status.json` instead of inspecting the training
container through the Kubernetes API. The example uses `ReadWriteMany` so the
export and inference pods can mount the claim at the same time; choose an access
mode supported by your storage class and workload topology.

```yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: standalone-sft-adapters
spec:
  accessModes: [ReadWriteMany]
  storageClassName: mlstorage
  resources:
    requests:
      storage: 128Gi
---
```

Add the following sidecar under `containers`, and add the `model-export` volume
shown below. Replace `my-adapter` with a unique destination name. Atomic directory creation prevents concurrent jobs from overwriting an existing export.

```yaml
- name: model-export
  image: busybox:1.36
  command:
    - /bin/sh
    - -c
    - |
      set -e
      DEST=/model-export/my-adapter
      STATUS_FILE=/output/runtime/train-status.json
      STATUS_TIMEOUT=180
      STATUS_STARTUP_TIMEOUT=300
      WAIT_STARTED=$(date +%s)
      die() {
        echo "ERROR: $1" >&2
        exit 1
      }
      if ! mkdir "$DEST" 2>/dev/null; then
        die "destination $DEST already exists, refusing to overwrite."
      fi
      trap 'rmdir "$DEST" 2>/dev/null' EXIT
      echo "Waiting for training status in $STATUS_FILE..."
      while true; do
        NOW=$(date +%s)
        if [ -f "$STATUS_FILE" ]; then
          STATUS=$(awk -F'"' '/"status"/ {print $4; exit}' "$STATUS_FILE")
          case "$STATUS" in
            COMPLETED) break ;;
            FAILED) die "training failed, skipping export." ;;
            RUNNING)
              LAST_UPDATE=$(stat -c %Y "$STATUS_FILE")
              if [ $((NOW - LAST_UPDATE)) -gt "$STATUS_TIMEOUT" ]; then
                die "training status has not been updated for more than ${STATUS_TIMEOUT} seconds."
              fi
              ;;
          esac
        elif [ $((NOW - WAIT_STARTED)) -gt "$STATUS_STARTUP_TIMEOUT" ]; then
          die "training status file did not appear within ${STATUS_STARTUP_TIMEOUT} seconds."
        fi
        sleep 10
      done
      echo "Training completed, copying adapter..."
      cp -r /output/checkpoint-final-adapter/. "$DEST"
      trap - EXIT
      exit 0
  volumeMounts:
    - name: output
      mountPath: /output
      readOnly: true
    - name: model-export
      mountPath: /model-export
```

```yaml
      volumes:
        - name: model-export
          persistentVolumeClaim:
            claimName: standalone-sft-adapters
```

After the Job reports `Complete`, the adapter is available at
`my-adapter/` in the `standalone-sft-adapters` claim. The directory must use the
Hugging Face PEFT layout expected by the inference image:

```text
my-adapter/
├── adapter_config.json
└── adapter_model.safetensors
```

## Deploy the adapter for inference

Deploy the trained LoRA adapter as an AIM. Replace the image, model ID, and PVC
names for your environment. Use an AIM image built for the same base model as
the adapter.

### Verify the adapter PVC

The PVC must contain the adapter directory created by the export step. Verify
that it exists before creating the deployment:

```bash
export ADAPTER_PVC=standalone-sft-adapters

kubectl get pvc "$ADAPTER_PVC"
```

### Create the Deployment

Save this manifest as `gemma-aim-deployment.yaml`:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: gemma-aim
spec:
  replicas: 1
  selector:
    matchLabels:
      app: gemma-aim
  template:
    metadata:
      labels:
        app: gemma-aim
    spec:
      containers:
        - name: aim
          image: amdenterpriseai/aim-instinct-google-gemma-3-27b-it:0.13.0
          imagePullPolicy: Always
          ports:
            - containerPort: 8000
          resources:
            requests:
              amd.com/gpu: "1"
              memory: "64Gi"
              cpu: "8"
            limits:
              amd.com/gpu: "1"
              memory: "64Gi"
              cpu: "8"
          env:
            - name: AIM_MODEL_ID
              value: google/gemma-3-27b-it
            - name: AIM_ADAPTER_SOURCE
              value: /adapters
            - name: AIM_ADAPTER_MODE
              value: dynamic
            - name: AIM_ADAPTER_MAX_COUNT
              value: "8"
            - name: AIM_ADAPTER_MAX_CPU_COUNT
              value: "16"
            - name: AIM_ADAPTER_MAX_RANK
              value: "64"
          volumeMounts:
            - name: adapter
              mountPath: /adapters
              readOnly: true
      volumes:
        - name: adapter
          persistentVolumeClaim:
            claimName: standalone-sft-adapters
            readOnly: true
```

Apply the Deployment and wait for it to become ready:

```bash
kubectl apply -f gemma-aim-deployment.yaml
kubectl rollout status deployment/gemma-aim
kubectl logs deployment/gemma-aim -f
```

Set `AIM_ADAPTER_MAX_RANK` to at least the LoRA rank used during training. The
Gemma recipe and the corresponding `deploy-aim` override use rank `64`.

### Create the Service

Save this manifest as `gemma-aim-service.yaml`:

```yaml
apiVersion: v1
kind: Service
metadata:
  name: gemma-aim
spec:
  selector:
    app: gemma-aim
  ports:
    - port: 8000
      targetPort: 8000
```

Apply the Service and inspect the endpoint:

```bash
kubectl apply -f gemma-aim-service.yaml
kubectl get deployment gemma-aim
kubectl get pods -l app=gemma-aim
kubectl get service gemma-aim
```

For an alternative automated deployment approach, use AIM Engine. For more information, see the [AIM Engine repository](https://github.com/amd-enterprise-ai/aim-engine).
