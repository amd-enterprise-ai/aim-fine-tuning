# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict
from pydantic.functional_validators import BeforeValidator
from typing_extensions import Annotated


def _disallow(v: Any) -> Any:
    """Helper for disallowing user inputs in certain fields"""
    raise ValueError("This field may not be specified by the user.")


# This is a type for pydantic that, in combination with validate_default=False, disallows the user to specify it.
# This is used for some class variables that HuggingFace TrainingArguments computes during __post_init__, as those
# should remain class variables, but should not be specified by the user directly.
DisallowedInput = Annotated[Any, BeforeValidator(_disallow)]


class BaseConfig(BaseModel):
    """Base class for Configs to inherit

    This is very close to a regular pydantic BaseModel, but just sets some different defaults.

    All configs should inherit from this, since the defaults define important behaviour such as extra=Extra.forbid
    instead of extra=Extra.ignore (so that if you typo a configuration option, it won't be silently ignored, but instead
    will raise a validation error.
    """

    # Pydantic default is Extra.ignore (silently ignore additional variables in configs)
    # protected_namespaces is set to be an empty tuple, because huggingface uses model_ attributes; if we set a model_args variable elsewhere, pydantic will emit a warning about this.
    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    @classmethod
    def from_yaml(cls, yamlstream):
        """Instantiate a Config from a YAML stream (e.g. open file)

        Example:
            >>> class ExperimentConfig(BaseConfig):
            ...     exp_name: str
            ...     num_epochs: int
            >>> yamlstr = '''
            ... exp_name: "example-experiment"
            ... num_epochs: 4
            ... '''
            >>> conf = ExperimentConfig.from_yaml(yamlstr)
            >>> conf.exp_name
            'example-experiment'
        """
        data = yaml.safe_load(yamlstream)
        return cls(**data)
