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
# ==============================================================================
# ======================== UNIT TESTS FOR THE MODULE ===========================
# ==============================================================================
import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import torch

from axon.core import Action, BaseAgent, BaseEnv
from axon.engine.chat_template.parser import ChatTemplateParser
from axon.engine.state.program_state import ModelOutput, ModelStopReason, ProgramState, Step, StopPartialProgram


# Mock classes for testing dependencies
class MockTokenizer:
    def __init__(self):
        self.name_or_path = "mock-tokenizer-for-testing"
        self.padding_side = "right"
        self.pad_token_id = 0
        self.pad_token = "<pad>"
        self.eos_token = "<eos>"
        self.eos_token_id = 2

    def encode(self, text, add_special_tokens=False):
        return [1] * len(text.split())

    def decode(self, tokens, skip_special_tokens=False):
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.tolist()
        return f"decoded_response_{len(tokens)}"

    def __call__(self, text_list, padding=True, return_tensors="pt", add_special_tokens=False):
        if not text_list:
            return {"input_ids": torch.tensor([]), "attention_mask": torch.tensor([])}
        encoded_list = [self.encode(s, add_special_tokens) for s in text_list]
        max_len = max(len(e) for e in encoded_list) if encoded_list else 0

        input_ids = [e + [self.pad_token_id] * (max_len - len(e)) for e in encoded_list]
        attention_mask = [[1] * len(e) + [0] * (max_len - len(e)) for e in encoded_list]

        if return_tensors == "pt":
            return {
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            }
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def apply_chat_template(self, conversation, tokenize=False, add_generation_prompt=False, **kwargs):
        if tokenize:
            raise NotImplementedError("Mock tokenizer does not support tokenizing in apply_chat_template")

        full_prompt = ""
        for message in conversation:
            full_prompt += f"{message['role']}: {message['content']}"

        if add_generation_prompt:
            full_prompt += "assistant:"

        return full_prompt.strip()


class MockAgent(BaseAgent):
    def __init__(self, **kwargs):
        super().__init__()
        self.reset()

    @property
    def system_prompt(self):
        return "system prompt"

    def reset(self):
        pass

    def process_observation(self, observation, reward, done, info, **kwargs):
        return str(observation)

    def process_action(self, action: str) -> Action:
        return Action(action=action)


class MockEnv(BaseEnv):
    def __init__(self, **kwargs):
        self.step_count = 0
        self.max_steps_in_env = kwargs.get("max_steps_in_env", 2)
        self.idx = kwargs.get("idx", 0)
        self.env_args = kwargs

    def reset(self):
        self.step_count = 0
        return "initial_observation", {"info": "reset_info"}

    def step(self, action):
        self.step_count += 1
        done = self.step_count >= self.max_steps_in_env
        reward = 1.0 if done else 0.1
        obs = f"observation_{self.step_count}"
        return obs, reward, done, {"info": f"step_{self.step_count}"}

    def compute_final_reward(self):
        return 10.0

    def close(self):
        pass

    @classmethod
    def is_multithread_safe(cls):
        return True

    @classmethod
    def from_dict(cls, config):
        return cls(**config)


def _make_mock_config():
    """Build a mock config that satisfies Engine's attribute access."""
    config = MagicMock()
    config.engine_endpoint.enable = False
    config.engine_endpoint.host = "localhost"
    config.engine_endpoint.port = 8000
    config.engine_endpoint.force_port = False
    config.partial_rollout.enable = False
    config.partial_rollout.n_iters = 1
    config.moe_replay = False
    return config


@patch("axon.engine.engine.TokenizerPool")
class TestEngine(unittest.TestCase):
    def setUp(self):
        self.mock_tokenizer = MockTokenizer()
        self.mock_chat_parser = MagicMock(spec=ChatTemplateParser)
        self.mock_chat_parser.parse.side_effect = lambda conversation, **kwargs: (
            self.mock_tokenizer.apply_chat_template(conversation, **kwargs)
        )

    def _make_engine(self, mock_tokenizer_pool_cls, **overrides):
        """Helper to create an engine with all mocks in place."""
        # Setup TokenizerPool mock
        mock_pool = MagicMock()
        mock_pool.start = AsyncMock(return_value=None)
        mock_pool.num_workers = 32
        mock_pool.batch_decode = AsyncMock(return_value=["text"])
        mock_pool.decode = AsyncMock(return_value="decoded")
        mock_tokenizer_pool_cls.return_value = mock_pool

        from axon.engine.engine import Engine

        defaults = {
            "tokenizer": self.mock_tokenizer,
            "chat_parser": self.mock_chat_parser,
            "sampling_client": MagicMock(),
            "config": _make_mock_config(),
            "max_steps": 3,
            "max_prompt_length": 512,
            "max_seq_length": 512,
        }
        defaults.update(overrides)
        engine = Engine(**defaults)
        return engine

    def _attach_state(self, engine, session_id="sess-1", *, enable_partial_rollout=False):
        state = ProgramState(
            uid="prog-1",
            session_id=session_id,
            enable_partial_rollout=enable_partial_rollout,
        )
        engine.session_state_map[session_id] = state
        return state

    def test_initialization(self, mock_tokenizer_pool_cls):
        """Test basic engine initialization with mocked dependencies."""
        engine = self._make_engine(mock_tokenizer_pool_cls)

        self.assertEqual(engine.max_steps, 3)
        self.assertIsNotNone(engine.chat_parser)
        self.assertIsInstance(engine.chat_parser, MagicMock)

        # TokenizerPool should have been constructed and started
        mock_tokenizer_pool_cls.assert_called_once()
        mock_tokenizer_pool_cls.return_value.start.assert_called_once()

    def test_config_attributes(self, mock_tokenizer_pool_cls):
        """Test that config attributes are properly set."""
        engine = self._make_engine(mock_tokenizer_pool_cls)

        self.assertFalse(engine.enable_api_server)
        self.assertFalse(engine.enable_partial_rollout)
        self.assertFalse(engine.moe_replay)
        self.assertEqual(engine.max_prompt_length, 512)
        self.assertEqual(engine.max_seq_length, 512)

    def test_sampling_client_initialization(self, mock_tokenizer_pool_cls):
        """Test engine initialization with a SamplingClient."""
        mock_sampling_client = MagicMock()
        engine = self._make_engine(
            mock_tokenizer_pool_cls,
            sampling_client=mock_sampling_client,
        )

        self.assertIs(engine.sampling_client, mock_sampling_client)

    def test_eval_train_mode(self, mock_tokenizer_pool_cls):
        """Test switching between eval and train modes."""
        engine = self._make_engine(mock_tokenizer_pool_cls)

        self.assertFalse(engine.validation_mode)
        engine.eval()
        self.assertTrue(engine.validation_mode)
        engine.train()
        self.assertFalse(engine.validation_mode)

    def test_set_global_steps(self, mock_tokenizer_pool_cls):
        """Test setting global steps."""
        engine = self._make_engine(mock_tokenizer_pool_cls)

        self.assertEqual(engine.global_steps, -1)
        engine.set_global_steps(100)
        self.assertEqual(engine.global_steps, 100)

    def test_model_stop_reason_uses_sampler_finish_reason(self, mock_tokenizer_pool_cls):
        engine = self._make_engine(mock_tokenizer_pool_cls)

        self.assertEqual(
            engine._model_stop_reason(
                token_ids=[123],
                finish_reason="length",
                sampler_stop_reason=None,
                sampling_params={},
            ),
            ModelStopReason.LENGTH,
        )
        self.assertEqual(
            engine._model_stop_reason(
                token_ids=[123],
                finish_reason="stop",
                sampler_stop_reason="<turn|>",
                sampling_params={},
            ),
            ModelStopReason.STOP,
        )

    def test_model_stop_reason_falls_back_to_stop_token_ids(self, mock_tokenizer_pool_cls):
        engine = self._make_engine(mock_tokenizer_pool_cls)

        self.assertEqual(
            engine._model_stop_reason(
                token_ids=[42],
                finish_reason=None,
                sampler_stop_reason=None,
                sampling_params={"stop_token_ids": [42]},
            ),
            ModelStopReason.STOP,
        )
        self.assertEqual(
            engine._model_stop_reason(
                token_ids=[7],
                finish_reason=None,
                sampler_stop_reason=None,
                sampling_params={"stop_token_ids": [42]},
            ),
            ModelStopReason.LENGTH,
        )

    def test_generate_llm_response_keeps_length_chunk_in_partial_state(self, mock_tokenizer_pool_cls):
        engine = self._make_engine(mock_tokenizer_pool_cls, max_seq_length=8)
        self._attach_state(engine, enable_partial_rollout=True)
        step = Step(uid="step-1", session_id="sess-1")
        engine._get_model_response = AsyncMock(
            return_value=ModelOutput.from_token_strs(
                token_ids=[10, 11],
                token_strs=["Act", "ion"],
                logprobs=[-0.1, -0.2],
                stop_reason=ModelStopReason.LENGTH,
                moe_routermap=[],
            )
        )

        with self.assertRaises(StopPartialProgram):
            asyncio.run(engine.generate_llm_response(step))

        self.assertEqual(step.partial_text, "Action")
        self.assertEqual(step.partial_tokens, [10, 11])
        self.assertEqual(step.partial_logprobs, [-0.1, -0.2])
        self.assertEqual(step.partial_token_strs, ["Act", "ion"])
        self.assertEqual(step.text, "")
        self.assertEqual(step.tokens, [])
        self.assertEqual(step.chat_completions, [])
        self.mock_chat_parser.assistant_message_content.assert_not_called()

    def test_generate_llm_response_commits_raw_partial_stream_and_clean_chat_content(self, mock_tokenizer_pool_cls):
        engine = self._make_engine(mock_tokenizer_pool_cls, max_seq_length=8)
        self._attach_state(engine, enable_partial_rollout=True)
        self.mock_chat_parser.assistant_message_content.side_effect = lambda text: text.removesuffix("<turn|>")
        step = Step(uid="step-1", session_id="sess-1")
        step.partial_text = "Act"
        step.partial_tokens = [10]
        step.partial_logprobs = [-0.1]
        step.partial_token_strs = ["Act"]
        engine._get_model_response = AsyncMock(
            return_value=ModelOutput.from_token_strs(
                token_ids=[11, 12],
                token_strs=["ion", "<turn|>"],
                logprobs=[-0.2, -0.3],
                stop_reason=ModelStopReason.STOP,
                moe_routermap=[],
            )
        )

        response = asyncio.run(engine.generate_llm_response(step))

        self.assertEqual(response, "Action")
        self.assertEqual(step.text, "Action<turn|>")
        self.assertEqual(step.tokens, [10, 11, 12])
        self.assertEqual(step.masks, [1, 1, 1])
        self.assertEqual(step.logprobs, [-0.1, -0.2, -0.3])
        self.assertEqual(step.chat_completions, [{"role": "assistant", "content": "Action"}])
        self.assertEqual(step.response_token_strs, ["Act", "ion", "<turn|>"])
        self.assertEqual(step.partial_text, "")
        self.assertEqual(step.partial_tokens, [])

    def test_generate_llm_response_cleans_after_multiple_partial_rollouts(self, mock_tokenizer_pool_cls):
        config = _make_mock_config()
        config.partial_rollout.n_iters = 4
        engine = self._make_engine(mock_tokenizer_pool_cls, config=config, max_seq_length=16)
        self._attach_state(engine, enable_partial_rollout=True)
        self.mock_chat_parser.assistant_message_content.side_effect = lambda text: text.removesuffix("<turn|>")
        step = Step(uid="step-1", session_id="sess-1")
        engine._get_model_response = AsyncMock(
            side_effect=[
                ModelOutput.from_token_strs(
                    token_ids=[10, 11],
                    token_strs=["Ac", "ti"],
                    logprobs=[-0.1, -0.2],
                    stop_reason=ModelStopReason.LENGTH,
                    moe_routermap=[],
                ),
                ModelOutput.from_token_strs(
                    token_ids=[12, 13],
                    token_strs=["on", ": "],
                    logprobs=[-0.3, -0.4],
                    stop_reason=ModelStopReason.LENGTH,
                    moe_routermap=[],
                ),
                ModelOutput.from_token_strs(
                    token_ids=[14, 15],
                    token_strs=["Up", "<turn|>"],
                    logprobs=[-0.5, -0.6],
                    stop_reason=ModelStopReason.STOP,
                    moe_routermap=[],
                ),
            ]
        )

        with self.assertRaises(StopPartialProgram):
            asyncio.run(engine.generate_llm_response(step))
        step.partial_rollout_max_tokens += engine.max_seq_length_per_iter
        with self.assertRaises(StopPartialProgram):
            asyncio.run(engine.generate_llm_response(step))
        step.partial_rollout_max_tokens += engine.max_seq_length_per_iter

        response = asyncio.run(engine.generate_llm_response(step))

        self.assertEqual(response, "Action: Up")
        self.assertEqual(step.text, "Action: Up<turn|>")
        self.assertEqual(step.tokens, [10, 11, 12, 13, 14, 15])
        self.assertEqual(step.chat_completions, [{"role": "assistant", "content": "Action: Up"}])
        self.mock_chat_parser.assistant_message_content.assert_called_once_with("Action: Up<turn|>")
        self.assertEqual(step.partial_text, "")
        self.assertEqual(step.partial_tokens, [])

    def test_session_management(self, mock_tokenizer_pool_cls):
        """Test init_session creates a session and end_session marks it done."""
        engine = self._make_engine(mock_tokenizer_pool_cls)

        async def run_test():
            session_id = await engine.init_session()
            self.assertIsInstance(session_id, str)
            self.assertIn(session_id, engine.session_state_map)

            # End session — marks program as done (stays in map until collected)
            await engine.end_session(session_id, reward=1.0)
            state = engine.session_state_map[session_id]
            self.assertTrue(state.done)
            self.assertEqual(state.reward, 1.0)

        asyncio.run_coroutine_threadsafe(run_test(), engine._loop).result(timeout=10)

    def test_init_session_with_group_id(self, mock_tokenizer_pool_cls):
        """Test init_session with group_id."""
        engine = self._make_engine(mock_tokenizer_pool_cls)

        async def run_test():
            session_id = await engine.init_session(group_id="batch_0")
            self.assertIn(session_id, engine.session_state_map)
            state = engine.session_state_map[session_id]
            self.assertEqual(state.group_id, "batch_0")

        asyncio.run_coroutine_threadsafe(run_test(), engine._loop).result(timeout=10)

    def test_program_managers_initialized(self, mock_tokenizer_pool_cls):
        """Test that program managers are properly initialized."""
        engine = self._make_engine(mock_tokenizer_pool_cls)

        self.assertIsNotNone(engine.program_manager)
        self.assertIsNotNone(engine.val_program_manager)
        self.assertEqual(len(engine.session_state_map), 0)

    def test_get_finished_programs_empty(self, mock_tokenizer_pool_cls):
        """Test get_finished_programs when no programs are finished."""
        engine = self._make_engine(mock_tokenizer_pool_cls)

        async def run_test():
            finished = await engine.get_finished_programs()
            self.assertEqual(len(finished), 0)

        asyncio.run_coroutine_threadsafe(run_test(), engine._loop).result(timeout=10)


if __name__ == "__main__":
    unittest.main()
