# Copyright 2025 Model AI Corp.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from enum import Enum


class Role(Enum):
    """Roles in the Axon training pipeline.

    Each role corresponds to a worker group in the resource pool. The Hydra
    config grafts the trainer template four times — once each under ``actor``,
    ``ref``, ``critic``, and ``reward_model`` — so per-role overrides under those
    keys map to the matching ``Role`` member.

    Members:
        Trainer: The trainer process itself (rare to refer to directly).
        Sampler: The vLLM / SGLang inference worker group.
        Actor: The policy being optimised.
        Critic: The value model used by ``advantage: gae``. Disabled by default.
        RefPolicy: The frozen reference policy used by ``loss_args.kl_coef`` /
            ``kl_reward``. Loaded only when needed.
        RewardModel: An optional learned reward model. Disabled by default;
            enable via ``reward_model.enable: true``.

    Use :meth:`from_string` to convert a config string back to the enum.
    """

    Trainer = "trainer"
    Sampler = "sampler"
    Actor = "actor"
    Critic = "critic"
    RefPolicy = "ref"
    RewardModel = "rm"

    def __str__(self):
        return self.value

    @classmethod
    def from_string(cls, name: str):
        string_mapping = {
            "actor": cls.Actor,
            "sampler": cls.Sampler,
            "critic": cls.Critic,
            "ref": cls.RefPolicy,
            "rm": cls.RewardModel,
            "trainer": cls.Trainer,
        }
        role = string_mapping.get(name.lower())
        if role is None:
            raise ValueError(f"No Role found for string: {name}")
        return role
