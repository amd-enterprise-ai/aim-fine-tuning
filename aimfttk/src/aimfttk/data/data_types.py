# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""Definitions of the aimfttk data types"""

from typing import Dict, List

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: str  # non emptpy
    content: str  # non empty
    tags: List[str] = Field(default_factory=list, exclude=True)
    labels: dict[str, str] = Field(default_factory=dict, exclude=True)


class ChatConversation(BaseModel):
    dataset: str | None = None
    id: str | None = None
    messages: List[ChatMessage]
    tags: List[str] = []
    labels: dict[str, str] = {}


class DirectPreference(BaseModel):
    """Records for direct preferences by providing rejected versus preferred
    options.
    """

    dataset: str | None = None
    id: str | None = None
    prompt_messages: List[ChatMessage]
    chosen_messages: List[ChatMessage]
    rejected_messages: List[ChatMessage]
    tags: List[str] = []
    labels: Dict[str, str] = {}


class EvalConversation(BaseModel):
    """A conversation used for evaluation, which contains the conversation history and the expected response"""

    dataset: str | None = None
    id: str | None = None
    prompt_messages: List[ChatMessage]
    response: str | None = None
    tags: List[str] = []
    labels: Dict[str, str] = {}


data_type_by_method = {"sft": ChatConversation, "dpo": DirectPreference}
data_type_by_name = {cls.__name__: cls for cls in data_type_by_method.values()}
