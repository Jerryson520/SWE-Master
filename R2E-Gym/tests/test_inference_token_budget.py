import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from r2egym.agenthub.agent.agent import (
    Agent,
    ContextLimitError,
    SummaryContextLimitError,
)
from r2egym.agenthub.trajectory import Trajectory


def make_agent(**other_args):
    agent = Agent.__new__(Agent)
    agent.llm_name = "hosted_vllm/swe-master-sft"
    agent.llm_base_url = "http://127.0.0.1:18000/v1"
    agent.other_args = other_args
    agent.logger = Mock()
    agent.max_retries = 1
    agent.summary_completion_tokens = 0
    return agent


class InferenceTokenBudgetTests(unittest.TestCase):
    def test_normal_request_uses_configured_output_limit(self):
        agent = make_agent(
            context_window=32768,
            context_safety_margin=1024,
            max_output_tokens=2048,
        )
        self.assertEqual(agent._request_max_output_tokens(16000, 9000), 2048)

    def test_request_shrinks_near_context_limit(self):
        agent = make_agent(context_window=32768, context_safety_margin=1024)
        self.assertEqual(agent._request_max_output_tokens(30500, 9000), 1244)

    def test_request_shrinks_to_remaining_trajectory_budget(self):
        agent = make_agent(context_window=32768, context_safety_margin=1024)
        self.assertEqual(agent._request_max_output_tokens(10000, 321), 321)

    def test_no_context_space_raises_before_call(self):
        agent = make_agent(context_window=32768, context_safety_margin=1024)
        with self.assertRaises(ContextLimitError):
            agent._request_max_output_tokens(31744, 1000)

    def test_summary_has_independent_dynamic_budget(self):
        agent = make_agent(
            context_window=32768,
            max_output_tokens=2048,
            summary_context_window=8192,
            summary_max_output_tokens=512,
            context_safety_margin=128,
        )
        self.assertEqual(
            agent._request_max_output_tokens(7000, summary=True), 512
        )
        with self.assertRaises(SummaryContextLimitError):
            agent._request_max_output_tokens(8100, summary=True)

    def test_model_summary_tracks_tokens_separately(self):
        agent = make_agent(
            context_window=32768,
            summary_context_window=32768,
            summary_max_output_tokens=256,
        )
        agent.llm_name = "hosted_vllm/swe-master-sft"
        response = SimpleNamespace(
            usage=SimpleNamespace(completion_tokens=123)
        )
        with patch.dict("os.environ", {"OPENAI_API_KEY": "EMPTY"}), patch.object(
            agent, "_count_tokens", return_value=100
        ), patch(
            "r2egym.agenthub.agent.agent.litellm.completion", return_value=response
        ) as completion:
            agent.model_summary([{"role": "user", "content": "summary"}])
        self.assertEqual(completion.call_args.kwargs["max_tokens"], 256)
        self.assertEqual(agent.summary_completion_tokens, 123)

    def test_old_trajectory_json_remains_loadable(self):
        payload = {
            "trajectory_steps": [],
            "problem_statement": "issue",
            "docker_image": "image",
            "env_args": {},
            "agent_args": {},
            "max_steps": 1,
            "max_steps_absolute": 1,
            "max_token_limit": 16384,
            "max_llm_time": 1,
            "max_exec_time": 1,
            "max_total_time": 1,
            "exit_reason": "token_limit",
            "output_patch": "",
        }
        trajectory = Trajectory.model_validate(payload)
        self.assertEqual(trajectory.max_token_limit, 16384)
        self.assertEqual(trajectory.trajectory_completion_tokens, 0)


if __name__ == "__main__":
    unittest.main()
