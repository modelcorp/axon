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
"""
Tests for the sampler infrastructure in axon/sampler/.

Covers:
- Engine ABC and engine registry
- Server, SERVER_REGISTRY (ClassRegistry)
- SamplerMode enum
- TokenOutput model
"""

from unittest.mock import MagicMock, patch

import pytest

from axon.sampler import _ENGINE_REGISTRY, SERVER_REGISTRY, get_engine_class, get_server_class
from axon.sampler.base.engine import Engine
from axon.sampler.base.server import (
    SamplerMode,
    Server,
    TokenOutput,
)

# =============================================================================
# TokenOutput tests
# =============================================================================


class TestTokenOutput:
    """Tests for the TokenOutput model."""

    def test_basic_creation(self):
        output = TokenOutput(token_ids=[1, 2, 3])
        assert output.token_ids == [1, 2, 3]
        assert output.log_probs is None
        assert output.routed_experts is None
        assert output.stop_reason is None

    def test_full_creation(self):
        output = TokenOutput(
            token_ids=[10, 20, 30],
            log_probs=[-0.1, -0.2, -0.3],
            routed_experts=[[0, 1], [2, 3], [4, 5]],
            stop_reason="completed",
        )
        assert output.token_ids == [10, 20, 30]
        assert output.log_probs == [-0.1, -0.2, -0.3]
        assert output.routed_experts == [[0, 1], [2, 3], [4, 5]]
        assert output.stop_reason == "completed"

    def test_empty_token_ids(self):
        output = TokenOutput(token_ids=[])
        assert output.token_ids == []

    def test_stop_reasons(self):
        for reason in ["completed", "aborted", None]:
            output = TokenOutput(token_ids=[1], stop_reason=reason)
            assert output.stop_reason == reason


# =============================================================================
# SamplerMode tests
# =============================================================================


class TestSamplerMode:
    """Tests for the SamplerMode enum."""

    def test_hybrid_mode(self):
        assert SamplerMode.HYBRID.value == "hybrid"

    def test_colocated_mode(self):
        assert SamplerMode.COLOCATED.value == "colocated"

    def test_standalone_mode(self):
        assert SamplerMode.STANDALONE.value == "standalone"

    def test_all_modes_exist(self):
        modes = {m.value for m in SamplerMode}
        assert modes == {"hybrid", "colocated", "standalone"}


# =============================================================================
# Engine registry tests (axon.sampler.engine)
# =============================================================================


class TestEngineRegistry:
    """Tests for the engine class registry."""

    def test_registry_contains_vllm(self):
        assert "vllm" in _ENGINE_REGISTRY

    def test_registry_contains_sglang(self):
        assert "sglang" in _ENGINE_REGISTRY

    def test_registry_vllm_path(self):
        assert _ENGINE_REGISTRY["vllm"] == "axon.sampler.vllm.engine.vLLMEngine"

    def test_registry_sglang_path(self):
        assert _ENGINE_REGISTRY["sglang"] == "axon.sampler.sglang.engine.SGLangEngine"

    def test_get_engine_class_unknown_raises(self):
        with pytest.raises(AssertionError):
            get_engine_class("nonexistent_engine")

    @patch("axon.sampler.importlib.import_module")
    def test_get_engine_class_imports_correctly(self, mock_import):
        """Test that get_engine_class resolves the correct module and class."""
        mock_module = MagicMock()
        mock_class = type("vLLMEngine", (), {})
        mock_module.vLLMEngine = mock_class
        mock_import.return_value = mock_module

        result = get_engine_class("vllm")

        mock_import.assert_called_once_with("axon.sampler.vllm.engine")
        assert result is mock_class


# =============================================================================
# SERVER_REGISTRY tests
# =============================================================================


class TestServerRegistry:
    """Tests for the SERVER_REGISTRY (ClassRegistry)."""

    def test_get_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown server"):
            SERVER_REGISTRY.get("nonexistent")

    def test_register_custom(self):
        """Test registering a custom server class."""
        mock_class = type("CustomReplica", (), {})
        SERVER_REGISTRY["test_custom"] = mock_class
        result = SERVER_REGISTRY.get("test_custom")
        assert result is mock_class
        # Clean up
        del SERVER_REGISTRY._registry["test_custom"]

    def test_get_server_class_unknown_raises(self):
        """Test that get_server_class raises for unknown names."""
        with pytest.raises(ValueError):
            get_server_class("nonexistent")


# =============================================================================
# Server tests
# =============================================================================


class TestServer:
    """Tests for the Server base class."""

    def _make_config(self, tp=1, dp=1, pp=1):
        config = MagicMock()
        config.tensor_model_parallel_size = tp
        config.data_parallel_size = dp
        config.pipeline_model_parallel_size = pp
        return config

    def test_world_size_calculation(self):
        """Test that world_size = tp * dp * pp."""
        config = self._make_config(tp=2, dp=4, pp=1)

        class ConcreteServer(Server):
            def get_ray_class_with_init_args(self):
                return MagicMock()

            async def launch_servers(self):
                pass

        replica = ConcreteServer(replica_rank=0, config=config, decoding_config=MagicMock())
        assert replica.world_size == 8  # 2 * 4 * 1

    def test_gpus_per_node_capped_at_world_size(self):
        """Test that gpus_per_node is capped at world_size."""
        config = self._make_config(tp=1, dp=2, pp=1)

        class ConcreteServer(Server):
            def get_ray_class_with_init_args(self):
                return MagicMock()

            async def launch_servers(self):
                pass

        replica = ConcreteServer(replica_rank=0, config=config, decoding_config=MagicMock(), gpus_per_node=8)
        assert replica.gpus_per_node == 2  # min(8, world_size=2)

    def test_nnodes_calculation(self):
        """Test that nnodes = world_size / gpus_per_node."""
        config = self._make_config(tp=2, dp=4, pp=2)

        class ConcreteServer(Server):
            def get_ray_class_with_init_args(self):
                return MagicMock()

            async def launch_servers(self):
                pass

        replica = ConcreteServer(replica_rank=0, config=config, decoding_config=MagicMock(), gpus_per_node=8)
        # world_size = 2*4*2 = 16, gpus_per_node = 8, nnodes = 2
        assert replica.nnodes == 2

    def test_invalid_gpus_per_node_raises(self):
        """Test that non-divisible gpus_per_node raises assertion."""
        config = self._make_config(tp=3, dp=1, pp=1)  # world_size = 3

        class ConcreteServer(Server):
            def get_ray_class_with_init_args(self):
                return MagicMock()

            async def launch_servers(self):
                pass

        with pytest.raises(AssertionError, match="must be divisible"):
            # gpus_per_node = min(2, 3) = 2, but 3 % 2 != 0
            ConcreteServer(replica_rank=0, config=config, decoding_config=MagicMock(), gpus_per_node=2)

    def test_server_address_property(self):
        """Test server_address property returns internal value."""
        config = self._make_config()

        class ConcreteServer(Server):
            def get_ray_class_with_init_args(self):
                return MagicMock()

            async def launch_servers(self):
                pass

        replica = ConcreteServer(replica_rank=0, config=config, decoding_config=MagicMock())
        assert replica.server_address is None

        replica._server_address = "http://localhost:8080"
        assert replica.server_address == "http://localhost:8080"

    def test_sampler_mode_initially_none(self):
        """Test that sampler_mode is None before init_hybrid/standalone."""
        config = self._make_config()

        class ConcreteServer(Server):
            def get_ray_class_with_init_args(self):
                return MagicMock()

            async def launch_servers(self):
                pass

        replica = ConcreteServer(replica_rank=0, config=config, decoding_config=MagicMock())
        assert replica.sampler_mode is None

    def test_is_reward_model_flag(self):
        """Test the is_reward_model flag."""
        config = self._make_config()

        class ConcreteServer(Server):
            def get_ray_class_with_init_args(self):
                return MagicMock()

            async def launch_servers(self):
                pass

        replica = ConcreteServer(replica_rank=0, config=config, decoding_config=MagicMock(), is_reward_model=True)
        assert replica.is_reward_model is True


# =============================================================================
# Engine tests
# =============================================================================


class TestEngine:
    """Tests for the Engine abstract base class."""

    def test_cannot_instantiate_directly(self):
        """Test that Engine cannot be instantiated."""
        with pytest.raises(TypeError):
            Engine(config=MagicMock(), device_mesh=MagicMock())

    def test_concrete_subclass(self):
        """Test that a concrete subclass can be created."""

        class ConcreteEngine(Engine):
            async def resume(self, tags):
                pass

            async def update_weights(self, weights, **kwargs):
                pass

            async def release(self):
                pass

        engine = ConcreteEngine(config=MagicMock(), device_mesh=MagicMock())
        assert engine.config is not None
        assert engine.device_mesh is not None
