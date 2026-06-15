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
"""TrainingClient — mirrors Tinker's TrainingClient surface.

Exposes only the methods that Tinker documents:
    forward, forward_backward, optim_step,
    save_state, load_state, get_tokenizer.
"""

from __future__ import annotations

from contextlib import contextmanager

from axon.protocol import DataProto


class TrainingClient:
    """Client for model training operations on a Ray worker group.

    Mirrors the Tinker TrainingClient API surface:

    * ``forward()``          — forward pass, returns log-probs (no gradients).
    * ``forward_backward()`` — forward + backward, accumulates gradients.
    * ``optim_step()``       — applies accumulated gradients.
    * ``save_state()``       — persist weights + optimizer.
    * ``load_state()``       — restore weights from checkpoint.
    * ``get_tokenizer()``    — return the tokenizer.

    Internally handles model load/offload to GPU so callers don't have to.

    Example::

        client = TrainingClient(worker_group=actor_wg, tokenizer=tokenizer)

        # log-probs only (auto loads model, offloads after)
        old_log_probs = client.forward(batch)

        # gradient step (caller manages load/offload via load_model_context)
        with client.load_model_context():
            output = client.forward_backward(batch, "ppo", {"entropy_coef": 0.01})
            client.optim_step(step_lr=True)

        # checkpoint
        client.save_state("ckpt/step_100", global_step=100)
    """

    def __init__(self, worker_group, config=None, tokenizer=None, processor=None):
        """
        Args:
            worker_group: Ray worker group (actor, ref-policy, …).
            config: Full training config or sub-config for this role.
            tokenizer: HuggingFace tokenizer.
            processor: Optional multimodal processor.
        """
        self.worker_group = worker_group
        self.config = config
        self.tokenizer = tokenizer
        self.processor = processor

    # ------------------------------------------------------------------
    # Model lifecycle (internal helper)
    # ------------------------------------------------------------------

    @contextmanager
    def load_model_context(self, include_model=True, include_optimizer=True):
        """Load model/optimizer to GPU, yield, then offload.

        Args:
            include_model: Move model parameters to GPU.
            include_optimizer: Move optimizer state to GPU.
        """
        self.worker_group.load_model(
            include_model=include_model,
            include_optimizer=include_optimizer,
        )
        try:
            yield
        finally:
            self.worker_group.offload_model(
                include_model=include_model,
                include_optimizer=include_optimizer,
            )

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------

    def forward(self, batch: DataProto) -> DataProto:
        """Forward-only pass — compute log-probs without gradients.

        Automatically loads model to GPU (no optimizer) and offloads after.

        Args:
            batch: DataProto with ``input_ids``, ``attention_mask``, etc.

        Returns:
            DataProto with ``log_probs`` (and ``entropys``) in ``batch``.
        """
        with self.load_model_context(include_optimizer=False):
            return self.worker_group.forward(batch)

    # ------------------------------------------------------------------
    # forward_backward
    # ------------------------------------------------------------------

    def forward_backward(
        self,
        batch: DataProto,
        loss_fn: str,
        loss_fn_args: dict | None = None,
    ) -> DataProto:
        """Forward + backward pass — accumulates gradients.

        Does **not** step the optimizer; call :meth:`optim_step` after.

        Args:
            batch: DataProto mini-batch.
            loss_fn: Loss function name (``"ppo"``, ``"cross_entropy"``, …).
            loss_fn_args: Extra keyword arguments for the loss function.

        Returns:
            DataProto with ``meta_info["metrics"]`` containing per-worker
            training metrics.
        """
        return self.worker_group.forward_backward(
            batch,
            loss_fn,
            loss_fn_args or {},
        )

    # ------------------------------------------------------------------
    # optim_step
    # ------------------------------------------------------------------

    def optim_step(self, step_lr: bool = False) -> DataProto:
        """Apply accumulated gradients to update model weights.

        Args:
            step_lr: Also step the learning-rate scheduler (typically
                ``True`` only on the last mini-batch of the last epoch).

        Returns:
            DataProto with ``meta_info["metrics"]`` (grad_norm, memory, lr).
        """
        return self.worker_group.optim_step(step_lr=step_lr)

    # ------------------------------------------------------------------
    # save_state / load_state
    # ------------------------------------------------------------------

    def save_state(
        self,
        checkpoint_path: str,
        global_step: int = 0,
        max_ckpt_to_keep: int | None = None,
        save_mode: str | None = None,
        async_save: bool | None = None,
    ):
        """Persist model weights and optimizer state.

        Args:
            checkpoint_path: Directory to write the checkpoint into.
            global_step: Training step (for metadata / checkpoint pruning).
            max_ckpt_to_keep: Keep at most this many checkpoints.
            save_mode: ``"sharded"``, ``"hf"``, or ``"both"``.
            async_save: Write asynchronously (Megatron only).
        """
        self.worker_group.save_state(
            checkpoint_path,
            global_step=global_step,
            max_ckpt_to_keep=max_ckpt_to_keep,
            save_mode=save_mode,
            async_save=async_save,
        )

    def load_state(self, checkpoint_path: str | None, save_mode: str | None = None):
        """Load model weights from a saved checkpoint.

        Args:
            checkpoint_path: Checkpoint directory. ``None`` offloads model.
            save_mode: Format override.
        """
        self.worker_group.load_state(checkpoint_path, save_mode=save_mode)

    # ------------------------------------------------------------------
    # get_tokenizer
    # ------------------------------------------------------------------

    def get_tokenizer(self):
        """Return the tokenizer associated with this model."""
        return self.tokenizer
