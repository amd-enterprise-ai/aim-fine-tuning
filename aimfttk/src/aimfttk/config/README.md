<!--
Copyright © Advanced Micro Devices, Inc., or its affiliates.

SPDX-License-Identifier: MIT
-->

# Config subpackage

This subpackage implements [Pydantic](https://docs.pydantic.dev/latest/)-based config definition and parsing.

### Some features

- Nested structure, compositional definition
- Default values
- Write config files in YAML or JSON or any similar nested format
- Configs are validated during instantiation (including type-coercing)
- Dump the config back out
- Attribute-style access (e.g. `exp_conf.data.packing`)

### Terminology

The word config ends up being very overloaded, unfortunately. It can mean a class or an instantiated object in Python,
or we may also use it to refer to the file (YAML, JSON, etc.) which holds the configuration data. HuggingFace has its
own idea of a Config, which are consistently Python native dataclasses. Our Configs look very similar (Pydantic also
uses class variables).

## How to define a config

Use a class with class variables, inheriting from `aimfttk.config.base.BaseConfig`

```python
from aimfttk.config.base import BaseConfig
class ExperimentConfig(BaseConfig):
    exp_name: str
    foo_value: float = 5.0
```

Corresponding YAML:

```yaml
exp_name: "agi-v0.1"
foo_value: 9000.1
```

Compose configs from existing sub-configs:

```python
from aimfttk.config.base import BaseConfig
from aimfttk.config import TrainingArguments, GenericPeftConfig
class ExperimentConfig(BaseConfig):
    exp_name: str
    foo_value: float = 5.0
    training_args: TrainingArguments
    peft_conf: GenericPeftConfig
```

Corresponding YAML:

```yaml
exp_name: "agi-v0.1"
foo_value: 9000.1
training_args:
  num_train_epochs: 2
  seed: 123141
peft_conf:
  peft_type: "LORA"
  task_type: "CAUSAL_LM"
  peft_kwargs:
    r: 32
```

## Other features

Load from YAML and dump the config back out:

```python
import yaml
from aimfttk.config.base import BaseConfig
class ExperimentConfig(BaseConfig):
    exp_name: str
    foo_value: float = 5.0

yamlstr = '''
exp_name: "agi-v0.1"
foo_value: 9000.1
'''
conf = ExperimentConfig.from_yaml(yamlstr)
print(yaml.dump(conf.model_dump()))
```
