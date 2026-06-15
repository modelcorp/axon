import sys
from pathlib import Path

import pytest

# Add recipes folder to sys.path since it's outside the axon package
_recipes_path = Path(__file__).parent.parent.parent.parent / "recipes"
if str(_recipes_path) not in sys.path:
    sys.path.insert(0, str(_recipes_path))

from geo3k.agent import Geo3kAgent  # noqa: E402

from axon.core import Action  # noqa: E402


class TestGeo3kAgentProcessObservation:
    """Tests for process_observation."""

    def test_first_observation_returns_image_message(self):
        """First call should format problem+instruction with an image and flip first_time."""
        agent = Geo3kAgent()
        dummy_image = {"bytes": b"fake", "path": None}
        observation = {
            "problem": "Find angle x.",
            "images": [dummy_image],
        }

        result = agent.process_observation(observation, 0.0, False, {})

        # Should return a list with one message dict
        assert isinstance(result, list) and len(result) == 1
        msg = result[0]
        assert msg["role"] == "user"

        # Content should be a list with text and image entries
        content = msg["content"]
        assert isinstance(content, list) and len(content) == 2

        text_part = content[0]
        assert text_part["type"] == "text"
        assert "Find angle x." in text_part["text"]
        assert agent.instruction in text_part["text"]

        image_part = content[1]
        assert image_part["type"] == "image"
        assert image_part["image"] is dummy_image

        # Also verify first_time was flipped
        assert agent.first_time is False

    def test_subsequent_observation_returns_text_message(self):
        """After the first call, observations return a plain text correction prompt."""
        agent = Geo3kAgent()
        first_obs = {
            "problem": "Solve.",
            "images": [{"bytes": b"img", "path": None}],
        }
        agent.process_observation(first_obs, 0.0, False, {})

        result = agent.process_observation("anything", 0.5, False, {"attempt": 2})

        assert isinstance(result, list) and len(result) == 1
        msg = result[0]
        assert msg["role"] == "user"
        assert isinstance(msg["content"], str)
        assert "previous answer" in msg["content"]

    def test_third_subsequent_observation_same_text(self):
        """Third call (and beyond) returns the same fixed text, no state accumulation."""
        agent = Geo3kAgent()
        first_obs = {
            "problem": "Solve.",
            "images": [{"bytes": b"img", "path": None}],
        }
        agent.process_observation(first_obs, 0.0, False, {})

        result2 = agent.process_observation("feedback1", 0.3, False, {})
        result3 = agent.process_observation("feedback2", 0.6, False, {})

        # Both subsequent observations should produce identical content
        assert result2[0]["content"] == result3[0]["content"]
        assert "previous answer" in result3[0]["content"]

    def test_first_observation_asserts_on_missing_problem(self):
        """Should raise AssertionError if 'problem' key is missing."""
        agent = Geo3kAgent()

        with pytest.raises(AssertionError):
            agent.process_observation({"images": [{"bytes": b"x", "path": None}]}, 0.0, False, {})

    def test_first_observation_asserts_on_missing_images(self):
        """Should raise AssertionError if 'images' is missing or empty."""
        agent = Geo3kAgent()

        with pytest.raises(AssertionError):
            agent.process_observation({"problem": "Solve.", "images": []}, 0.0, False, {})

    def test_first_observation_asserts_on_multiple_images(self):
        """Should raise AssertionError if more than one image is provided."""
        agent = Geo3kAgent()

        with pytest.raises(AssertionError):
            agent.process_observation(
                {"problem": "Solve.", "images": [{"bytes": b"a"}, {"bytes": b"b"}]},
                0.0,
                False,
                {},
            )

    # -- edge cases ---------------------------------------------------------

    def test_first_observation_string_raises_assertion(self):
        """First observation as a plain string (not dict) should raise AssertionError."""
        agent = Geo3kAgent()
        with pytest.raises(AssertionError):
            agent.process_observation("just a string", 0.0, False, {})

    def test_image_data_passthrough(self):
        """The agent doesn't validate image data, just passes it through."""
        agent = Geo3kAgent()
        # Use unconventional image data -- agent should not care
        weird_image = {"bytes": b"\x00\xff" * 100, "path": "/fake/path.png", "extra_key": 42}
        observation = {
            "problem": "What is this?",
            "images": [weird_image],
        }
        result = agent.process_observation(observation, 0.0, False, {})
        image_part = result[0]["content"][1]
        assert image_part["image"] is weird_image


class TestGeo3kAgentProcessAction:
    """Tests for process_action."""

    def test_returns_action_object(self):
        agent = Geo3kAgent()
        action = agent.process_action("some response")
        assert isinstance(action, Action)
        assert action.action == "some response"


class TestGeo3kAgentReset:
    """Tests for the reset method."""

    def test_reset_allows_reuse(self):
        """After reset, the agent should treat the next observation as the first."""
        agent = Geo3kAgent()
        obs = {"problem": "P1", "images": [{"bytes": b"i", "path": None}]}
        agent.process_observation(obs, 0.0, False, {})
        assert agent.first_time is False

        agent.reset()
        assert agent.first_time is True

        obs2 = {"problem": "P2", "images": [{"bytes": b"j", "path": None}]}
        result = agent.process_observation(obs2, 0.0, False, {})

        # Should be treated as first observation again (image message)
        assert isinstance(result[0]["content"], list)
        assert result[0]["content"][0]["text"].startswith("P2")


class TestGeo3kAgentMultipleEpisodes:
    """Test full episode lifecycle repeated."""

    def test_multiple_episode_lifecycle(self):
        """init -> observe -> act -> reset -> observe -> act, verify clean state each time."""
        agent = Geo3kAgent()
        obs1 = {"problem": "Episode 1", "images": [{"bytes": b"img1", "path": None}]}

        # Episode 1
        result1 = agent.process_observation(obs1, 0.0, False, {})
        assert agent.first_time is False
        assert isinstance(result1[0]["content"], list)
        assert "Episode 1" in result1[0]["content"][0]["text"]

        action1 = agent.process_action(r"<think>reasoning</think> \boxed{10}")
        assert isinstance(action1, Action)
        assert action1.action == r"<think>reasoning</think> \boxed{10}"

        # Reset
        agent.reset()
        assert agent.first_time is True

        # Episode 2 - should behave identically to a fresh agent
        obs2 = {"problem": "Episode 2", "images": [{"bytes": b"img2", "path": None}]}
        result2 = agent.process_observation(obs2, 0.0, False, {})
        assert agent.first_time is False
        assert isinstance(result2[0]["content"], list)
        assert "Episode 2" in result2[0]["content"][0]["text"]

        action2 = agent.process_action(r"<think>more reasoning</think> \boxed{20}")
        assert isinstance(action2, Action)

        # Subsequent observation in episode 2 should be text-based
        result3 = agent.process_observation("feedback", 0.5, False, {})
        assert isinstance(result3[0]["content"], str)
        assert "previous answer" in result3[0]["content"]
