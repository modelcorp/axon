# Copyright 2025 Model AI Corp.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import random
from enum import Enum

import numpy as np
import torch
import torch.distributed
from transformers import PreTrainedTokenizer, ProcessorMixin

from axon.utils.torch import get_device_name, get_torch_device


class StateSaveMode(str, Enum):
    """State save modes for FSDP training.

    - SHARDED: Save sharded model, optimizer, and extra state per rank (for resuming training)
    - HF: Save only HuggingFace format model (full weights gathered, for inference)
    - BOTH: Save both sharded state and HuggingFace model
    """

    SHARDED = "sharded"
    HF = "hf"
    BOTH = "both"


class BaseStateManager:
    """
    A state manager that saves and loads state in a SPMD way.

    Configuration:
        state_config:
          save_mode: "sharded" | "hf" | "both"

    Save modes:
        - sharded: Save model/optimizer/extra per rank (for resuming training)
        - hf: Save only HuggingFace model (for inference/deployment)
        - both: Save both sharded and HuggingFace
    """

    def __init__(
        self,
        model,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: torch.optim.lr_scheduler.LRScheduler = None,
        processing_class: PreTrainedTokenizer | ProcessorMixin = None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.processing_class = processing_class

        self.rank = torch.distributed.get_rank()
        self.world_size = torch.distributed.get_world_size()

    def load_state(self, local_path: str, save_mode: str = None):
        raise NotImplementedError("Implement this method in the subclass")

    def save_state(
        self,
        local_path: str,
        global_step: int = 0,
        max_ckpt_to_keep: int = None,
        save_mode: str = None,
        async_save: bool = None,
    ):
        raise NotImplementedError("Implement this method in the subclass")

    @staticmethod
    def get_rng_state():
        rng_state = {
            "cpu": torch.get_rng_state(),
            "numpy": np.random.get_state(),
            "random": random.getstate(),
        }
        if get_device_name() != "cpu":
            rng_state[get_device_name()] = get_torch_device().get_rng_state()
        return rng_state

    @staticmethod
    def load_rng_state(rng_state):
        torch.set_rng_state(rng_state["cpu"])
        np.random.set_state(rng_state["numpy"])
        random.setstate(rng_state["random"])
        if get_device_name() != "cpu":
            get_torch_device().set_rng_state(rng_state[get_device_name()])
