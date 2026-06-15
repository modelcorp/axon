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
Tests for the SamplingClient (axon.tinker).

Covers load balancing, address management, data locality, and request routing.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from axon.tinker import SamplingClient


def _make_mock_config(moe_replay=False, pausing_strategy="drain"):
    """Build a mock config for SamplingClient initialization."""
    config = MagicMock()
    config.sampler.get.return_value = 1  # tensor_parallel_size
    config.sampler.max_model_len = 128
    config.sampler.enable_prefix_caching = False
    config.decoding.n = 1
    config.decoding.temperature = 1.0
    config.decoding.top_p = 1.0
    config.decoding.top_k = -1
    config.decoding.repetition_penalty = 1.0
    config.decoding.get.side_effect = lambda key, default=None: {"logprobs": 1}.get(key, default)
    config.model_path = "org/test-model"
    config.moe_replay = moe_replay
    config.sampler_pausing_strategy = pausing_strategy
    return config


def _make_mock_tokenizer():
    tokenizer = MagicMock()
    tokenizer.pad_token_id = 0
    tokenizer.eos_token_id = 1
    return tokenizer


class TestRouterInit:
    """Tests for SamplingClient initialization."""

    def test_basic_init(self):
        config = _make_mock_config()
        tokenizer = _make_mock_tokenizer()
        router = SamplingClient(config=config, tokenizer=tokenizer, processor=None, addresses=["localhost:8000"])

        assert router.addresses == ["localhost:8000"]
        assert router.pad_token_id == 0
        assert router.eos_token_id == 1
        assert router.model_name == "org/test-model"
        assert router.counter == 0

    def test_multiple_addresses(self):
        config = _make_mock_config()
        tokenizer = _make_mock_tokenizer()
        addrs = ["server1:8000", "server2:8000", "server3:8000"]
        router = SamplingClient(config=config, tokenizer=tokenizer, processor=None, addresses=addrs)

        assert len(router.addresses) == 3
        assert all(addr in router._usage for addr in addrs)
        assert all(router._usage[addr] == 0 for addr in addrs)

    def test_usage_initialized_to_zero(self):
        config = _make_mock_config()
        tokenizer = _make_mock_tokenizer()
        addrs = ["a:1", "b:2"]
        router = SamplingClient(config=config, tokenizer=tokenizer, processor=None, addresses=addrs)

        for addr in addrs:
            assert router._usage[addr] == 0

    def test_invalid_pausing_strategy_raises(self):
        config = _make_mock_config(pausing_strategy="invalid")
        tokenizer = _make_mock_tokenizer()
        with pytest.raises(AssertionError):
            SamplingClient(config=config, tokenizer=tokenizer, processor=None, addresses=["localhost:8000"])

    def test_valid_pausing_strategies(self):
        tokenizer = _make_mock_tokenizer()
        for strategy in ["drain", "hold", "continue", "reset"]:
            config = _make_mock_config(pausing_strategy=strategy)
            router = SamplingClient(config=config, tokenizer=tokenizer, processor=None, addresses=["localhost:8000"])
            assert router.pausing_strategy == strategy

    def test_semaphores_created_per_address(self):
        config = _make_mock_config()
        tokenizer = _make_mock_tokenizer()
        addrs = ["a:1", "b:2", "c:3"]
        router = SamplingClient(config=config, tokenizer=tokenizer, processor=None, addresses=addrs)

        for addr in addrs:
            assert addr in router._address_semaphores
            assert isinstance(router._address_semaphores[addr], asyncio.Semaphore)


class TestRouterLoadBalancing:
    """Tests for SamplingClient load balancing behavior."""

    def test_first_request_gets_address(self):
        config = _make_mock_config()
        tokenizer = _make_mock_tokenizer()
        router = SamplingClient(config=config, tokenizer=tokenizer, processor=None, addresses=["a:1", "b:2"])

        addr = asyncio.run(router.get_address("app_1"))
        assert addr in ["a:1", "b:2"]

    def test_requests_distributed_across_servers(self):
        config = _make_mock_config()
        tokenizer = _make_mock_tokenizer()
        router = SamplingClient(config=config, tokenizer=tokenizer, processor=None, addresses=["a:1", "b:2"])

        addr1 = asyncio.run(router.get_address("app_1"))
        addr2 = asyncio.run(router.get_address("app_2"))

        # With 2 servers and 2 apps, they should be on different servers
        assert addr1 != addr2

    def test_data_locality(self):
        """Test that same app_id stays on same server when load is balanced.

        The router rebalances when min_usage == 0, so we need all servers
        to have some load for data locality to kick in.
        """
        config = _make_mock_config()
        tokenizer = _make_mock_tokenizer()
        router = SamplingClient(config=config, tokenizer=tokenizer, processor=None, addresses=["a:1", "b:2"])

        async def run():
            # First, put load on both servers so neither has 0 usage
            addr_a = await router.get_address("app_1")
            await router.get_address("app_2")
            # Both servers now have usage=1

            # Now app_1 should stick to its original server (data locality)
            addr_repeat = await router.get_address("app_1")
            return addr_a, addr_repeat

        addr1, addr2 = asyncio.run(run())

        # Same app_id should remain on same server when skew < 4
        assert addr1 == addr2

    def test_release_address_decrements_usage(self):
        config = _make_mock_config()
        tokenizer = _make_mock_tokenizer()
        router = SamplingClient(config=config, tokenizer=tokenizer, processor=None, addresses=["a:1"])

        asyncio.run(router.get_address("app_1"))
        assert router._usage["a:1"] == 1

        asyncio.run(router.release_address("a:1", "app_1"))
        assert router._usage["a:1"] == 0

    def test_release_address_floors_at_zero(self):
        config = _make_mock_config()
        tokenizer = _make_mock_tokenizer()
        router = SamplingClient(config=config, tokenizer=tokenizer, processor=None, addresses=["a:1"])

        # Release without ever getting → usage stays at 0
        asyncio.run(router.release_address("a:1", "app_1"))
        assert router._usage["a:1"] == 0

    def test_load_balance_skew_rebalance(self):
        """Test that load balancing rebalances when there's significant skew."""
        config = _make_mock_config()
        tokenizer = _make_mock_tokenizer()
        router = SamplingClient(config=config, tokenizer=tokenizer, processor=None, addresses=["a:1", "b:2"])

        # Route app_1 to one server and inflate its usage
        addr = asyncio.run(router.get_address("app_1"))

        # Manually create skew: set the first server's usage to very high
        router._usage[addr] = 10
        router._usage["a:1" if addr == "b:2" else "b:2"] = 0

        # Now requesting for app_1 again should trigger rebalance due to skew >= 4
        new_addr = asyncio.run(router.get_address("app_1"))
        other_addr = "a:1" if addr == "b:2" else "b:2"
        assert new_addr == other_addr  # Should have moved to the less-used server

    def test_many_apps_balanced(self):
        """Test distribution of many apps across servers."""
        config = _make_mock_config()
        tokenizer = _make_mock_tokenizer()
        addrs = ["a:1", "b:2", "c:3", "d:4"]
        router = SamplingClient(config=config, tokenizer=tokenizer, processor=None, addresses=addrs)

        # Route 8 different apps
        for i in range(8):
            asyncio.run(router.get_address(f"app_{i}"))

        # Each server should have exactly 2 (8 apps / 4 servers)
        usages = list(router._usage.values())
        assert sum(usages) == 8
        assert max(usages) - min(usages) <= 1  # Should be fairly balanced


class TestSampling:
    def test_sample_preserves_finish_and_stop_reasons(self):
        config = _make_mock_config()
        tokenizer = _make_mock_tokenizer()
        router = SamplingClient(config=config, tokenizer=tokenizer, processor=None, addresses=["a:1"])
        router.submit_completions = AsyncMock(
            return_value={
                "choices": [
                    {
                        "token_ids": [10, 11],
                        "logprobs": {
                            "token_logprobs": [-0.1, -0.2],
                            "tokens": ["Action", "<turn|>"],
                        },
                        "finish_reason": "stop",
                        "stop_reason": "<turn|>",
                    }
                ]
            }
        )

        result = asyncio.run(router.sample([[1, 2]], application_id="app", multi_modal_data_list=[None]))

        assert result["token_ids"] == [[10, 11]]
        assert result["logprobs"] == [[-0.1, -0.2]]
        assert result["token_strs"] == [["Action", "<turn|>"]]
        assert result["finish_reasons"] == ["stop"]
        assert result["stop_reasons"] == ["<turn|>"]
