# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0

from __future__ import annotations

import asyncio
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from openviking.session.train import (
    Case,
    ExecutionContext,
    Experience,
    ExperienceSet,
    Rubric,
    RubricCriterion,
    SingleTurnLLMRolloutExecutor,
    default_single_turn_prompt,
)
from openviking_cli.exceptions import NotFoundError


class FakeVLM:
    def __init__(self, response="assistant answer"):
        self.response = response
        self.calls = []

    async def get_completion_async(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


def _case() -> Case:
    return Case(
        name="case-1",
        task_signature="booking_duplicate",
        input={"user_request": "cancel duplicate booking"},
        rubric=Rubric(
            name="booking_rubric",
            description="Cancel only the verified duplicate booking.",
            criteria=[
                RubricCriterion(
                    name="verify_duplicate",
                    description="Verify duplicate status first.",
                    required=True,
                    weight=1.0,
                )
            ],
        ),
    )


def _policy_set() -> ExperienceSet:
    return ExperienceSet(
        root_uri="viking://user/u/memories/experiences",
        policies=[
            Experience(
                name="booking_policy",
                uri="viking://user/u/memories/experiences/booking_policy.md",
                version=2,
                status="production",
                content="Always verify duplicates before cancellation.",
            )
        ],
    )


@pytest.mark.asyncio
async def test_single_turn_llm_rollout_executor_produces_rollout_messages():
    vlm = FakeVLM()
    executor = SingleTurnLLMRolloutExecutor(vlm=vlm, thinking=False)
    context = ExecutionContext(policy_snapshot_id="snapshot-1")

    rollouts = await executor.execute([_case()], _policy_set(), context)

    assert len(rollouts) == 1
    rollout = rollouts[0]
    assert rollout.case.name == "case-1"
    assert rollout.policy_snapshot_id == "snapshot-1"
    assert [message.role for message in rollout.messages] == ["user", "assistant"]
    assert "Always verify duplicates" in rollout.messages[0].content
    assert "cancel duplicate booking" in rollout.messages[0].content
    assert rollout.messages[1].content == "assistant answer"
    assert vlm.calls[0]["thinking"] is False
    assert vlm.calls[0]["prompt"] == rollout.messages[0].content


@pytest.mark.asyncio
async def test_single_turn_llm_rollout_executor_accepts_custom_prompt_builder():
    vlm = FakeVLM(response=type("Resp", (), {"content": "structured answer"})())

    def build_prompt(case, policy_set, context):
        return f"custom:{case.name}:{len(policy_set.policies)}:{context.policy_snapshot_id}"

    executor = SingleTurnLLMRolloutExecutor(vlm=vlm, prompt_builder=build_prompt)

    rollouts = await executor.execute(
        [_case()],
        _policy_set(),
        ExecutionContext(policy_snapshot_id="snapshot-2"),
    )

    assert rollouts[0].messages[0].content == "custom:case-1:1:snapshot-2"
    assert rollouts[0].messages[1].content == "structured answer"


def test_default_single_turn_prompt_contains_case_policy_and_rubric():
    prompt = default_single_turn_prompt(
        _case(),
        _policy_set(),
        ExecutionContext(policy_snapshot_id="snapshot-3"),
    )

    assert "Policy snapshot: snapshot-3" in prompt
    assert "booking_policy v2 [production]" in prompt
    assert "cancel duplicate booking" in prompt
    assert "verify_duplicate" in prompt


def test_dataset_service_policy_set_from_dict_preserves_policies():
    from openviking.session.train.components.dataset_service import policy_set_from_dict

    policy_set = policy_set_from_dict(
        {
            "root_uri": "viking://user/u/memories/experiences",
            "policies": [
                {
                    "name": "booking_policy",
                    "uri": "viking://user/u/memories/experiences/booking_policy.md",
                    "version": 2,
                    "status": "production",
                    "content": "Always verify duplicates before cancellation.",
                    "metadata": {"domain": "booking"},
                }
            ],
            "metadata": {"snapshot": "remote"},
        }
    )

    assert policy_set.root_uri == "viking://user/u/memories/experiences"
    assert policy_set.metadata == {"snapshot": "remote"}
    assert len(policy_set.policies) == 1
    policy = policy_set.policies[0]
    assert policy.name == "booking_policy"
    assert policy.uri == "viking://user/u/memories/experiences/booking_policy.md"
    assert policy.version == 2
    assert policy.status == "production"
    assert policy.content == "Always verify duplicates before cancellation."
    assert policy.metadata == {"domain": "booking"}


def test_tau2_rollout_messages_use_completed_structured_tool_parts():
    from benchmark.tau2.train.rollout_executor import _build_rollout_messages
    from openviking.message import TextPart, ToolPart

    rollout_messages = _build_rollout_messages(
        system_prompt="policy",
        user_prompt="user request",
        tools_used=[
            {
                "tool_name": "get_user_details",
                "args": '{"user_id": "emma_kim_9957"}',
                "result": '{"membership": "gold"}',
            }
        ],
        final_content="done",
        evaluation_result=None,
        reward=1.0,
        runtime_messages=[
            {"role": "system", "content": "policy"},
            {"role": "user", "content": "user request"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {
                            "name": "get_user_details",
                            "arguments": '{"user_id": "emma_kim_9957"}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call-1",
                "name": "get_user_details",
                "content": '{"membership": "gold"}',
            },
            {"role": "assistant", "content": "done"},
        ],
    )

    assert isinstance(rollout_messages[0].parts[0], TextPart)
    assert rollout_messages[0].parts[0].text.startswith("system:\npolicy")

    tool_message = rollout_messages[2]
    assert tool_message.role == "user"
    assert isinstance(tool_message.parts[0], ToolPart)
    assert tool_message.parts[0].tool_status == "completed"
    assert tool_message.parts[0].tool_input == {"user_id": "emma_kim_9957"}
    assert tool_message.parts[0].tool_output == '{"membership": "gold"}'
    assert not any(
        isinstance(part, TextPart) and "tool-call:" in part.text
        for message in rollout_messages
        for part in message.parts
    )
    assert not any(
        isinstance(part, ToolPart) and part.tool_status == "running"
        for message in rollout_messages
        for part in message.parts
    )


def test_tau2_communicate_with_user_renders_as_dialogue():
    from benchmark.tau2.train.rollout_executor import _build_rollout_messages
    from openviking.message import TextPart, ToolPart

    rollout_messages = _build_rollout_messages(
        system_prompt="policy",
        user_prompt="user request",
        tools_used=[
            {
                "tool_name": "communicate_with_user",
                "args": {"content": "Could you provide your user ID?"},
                "result": "Sure, it is emma_kim_9957.",
            }
        ],
        final_content=None,
        evaluation_result=None,
        reward=1.0,
        runtime_messages=[
            {"role": "system", "content": "policy"},
            {"role": "user", "content": "user request"},
            {"role": "assistant", "content": "Could you provide your user ID?"},
            {"role": "user", "content": "Sure, it is emma_kim_9957."},
        ],
    )

    assert rollout_messages[2].role == "assistant"
    assert isinstance(rollout_messages[2].parts[0], TextPart)
    assert rollout_messages[2].parts[0].text == "Could you provide your user ID?"
    assert rollout_messages[3].role == "user"
    assert isinstance(rollout_messages[3].parts[0], TextPart)
    assert rollout_messages[3].parts[0].text == "Sure, it is emma_kim_9957."
    assert not any(
        isinstance(part, ToolPart) and part.tool_name == "communicate_with_user"
        for message in rollout_messages
        for part in message.parts
    )


def test_tau2_rollout_messages_omit_empty_final_after_done():
    from benchmark.tau2.train.rollout_executor import _build_rollout_messages

    rollout_messages = _build_rollout_messages(
        system_prompt="policy",
        user_prompt="user request",
        tools_used=[{"tool_name": "done", "args": "{}", "result": "Task Terminated"}],
        final_content=None,
        evaluation_result=None,
        reward=1.0,
        runtime_messages=[
            {"role": "system", "content": "policy"},
            {"role": "user", "content": "user request"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {"name": "done", "arguments": "{}"},
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call-1",
                "name": "done",
                "content": "Task Terminated",
            },
        ],
    )

    ids = {message.id for message in rollout_messages}
    assert "tau2-final" not in ids
    assert "tau2-reward" not in ids


def test_tau2_reward_info_is_normalized_into_generic_evaluation_criteria():
    import json

    from benchmark.tau2.train.rollout_executor import _build_rollout_messages, _tau2_evaluation

    reward_info = {
        "reward": 0.0,
        "reward_basis": ["DB", "COMMUNICATE"],
        "reward_breakdown": {"DB": 1.0, "COMMUNICATE": 0.0},
        "db_check": {"db_match": True, "db_reward": 1.0},
        "action_checks": [
            {
                "action": {
                    "name": "update_reservation_flights",
                    "arguments": {"reservation_id": "XEHM4B", "cabin": "business"},
                },
                "action_match": True,
                "action_reward": 1.0,
                "tool_type": "write",
            },
            {
                "action": {
                    "name": "send_certificate",
                    "arguments": {"reservation_id": "XEHM4B"},
                },
                "action_match": False,
                "action_reward": 0.0,
                "tool_type": "write",
            },
        ],
        "communicate_checks": [
            {
                "info": "1628",
                "met": False,
                "justification": "Required total was not communicated.",
            }
        ],
    }

    rollout_messages = _build_rollout_messages(
        system_prompt="policy",
        user_prompt="user request",
        tools_used=[],
        final_content="done",
        evaluation_result=reward_info,
        reward=0.0,
        runtime_messages=[
            {"role": "system", "content": "policy"},
            {"role": "user", "content": "user request"},
            {"role": "assistant", "content": "done"},
        ],
    )
    evaluation = _tau2_evaluation(reward=0.0, evaluation_result=reward_info)

    assert not any(message.id == "tau2-reward" for message in rollout_messages)
    assert not any("evaluation report:" in message.content for message in rollout_messages)
    assert [result.criterion_name for result in evaluation.criterion_results] == [
        "task_outcome",
        "environment_state",
        "required_actions",
        "required_communication",
    ]
    assert evaluation.passed is False
    assert evaluation.score == 0.0
    assert evaluation.metadata == {"source": "tau2", "reward": 0.0}
    assert "evaluation_result" not in json.dumps(evaluation.metadata, sort_keys=True)

    by_name = {result.criterion_name: result for result in evaluation.criterion_results}
    assert by_name["environment_state"].passed is True
    assert by_name["environment_state"].score == 1.0
    assert by_name["required_actions"].passed is False
    assert by_name["required_actions"].score == 0.5
    assert any(
        "update_reservation_flights" in evidence and "preserve" in evidence
        for evidence in by_name["required_actions"].evidence
    )
    assert any("send_certificate" in feedback for feedback in by_name["required_actions"].feedback)
    assert by_name["required_communication"].passed is False
    assert by_name["required_communication"].score == 0.0
    assert by_name["required_communication"].feedback == ["Required total was not communicated."]


def test_tau2_evaluation_ignores_malformed_optional_components():
    from benchmark.tau2.train.rollout_executor import _tau2_evaluation

    evaluation = _tau2_evaluation(
        reward="invalid",
        evaluation_result={
            "db_check": "invalid",
            "action_checks": [None, {"action": "invalid"}],
            "communicate_checks": [{"info": "missing met"}],
        },
    )

    assert evaluation.score == 0.0
    assert [result.criterion_name for result in evaluation.criterion_results] == ["task_outcome"]


def test_tau2_litellm_generate_rate_limit_retry_patch(monkeypatch):
    import benchmark.tau2.common.tau2_env.tau2_environment as tau2_environment
    import openviking.utils.model_retry as model_retry

    calls = {"count": 0}
    sleeps = []

    def fake_generate():
        calls["count"] += 1
        if calls["count"] < 5:
            raise RuntimeError("TPM (Tokens Per Minute) limit of the model is exceeded")
        return "ok"

    class FakeLLMUtils:
        generate = staticmethod(fake_generate)

    class FakeUserSimulator:
        generate = staticmethod(fake_generate)

    modules = {
        "tau2.utils.llm_utils": FakeLLMUtils,
        "tau2.user.user_simulator": FakeUserSimulator,
    }

    def fake_import_module(name):
        if name in modules:
            return modules[name]
        raise ImportError(name)

    monkeypatch.setattr(tau2_environment.importlib, "import_module", fake_import_module)
    monkeypatch.setattr(model_retry, "rate_limit_retry_delay", lambda attempt: attempt)
    monkeypatch.setattr(model_retry.time, "sleep", lambda delay: sleeps.append(delay))

    tau2_environment._install_tau2_litellm_rate_limit_retry()

    assert FakeLLMUtils.generate() == "ok"
    assert calls["count"] == 5
    assert sleeps == [1, 2, 3, 4]
    assert FakeUserSimulator.generate is FakeLLMUtils.generate


def test_tau2_litellm_generate_retry_patch_does_not_retry_non_rate_limit(monkeypatch):
    import benchmark.tau2.common.tau2_env.tau2_environment as tau2_environment
    import openviking.utils.model_retry as model_retry

    calls = {"count": 0}

    def fake_generate():
        calls["count"] += 1
        raise RuntimeError("AuthenticationError Unauthorized")

    class FakeLLMUtils:
        generate = staticmethod(fake_generate)

    def fake_import_module(name):
        if name == "tau2.utils.llm_utils":
            return FakeLLMUtils
        raise ImportError(name)

    monkeypatch.setattr(tau2_environment.importlib, "import_module", fake_import_module)

    def fail_on_sleep(_delay):
        raise AssertionError("unexpected sleep")

    monkeypatch.setattr(model_retry.time, "sleep", fail_on_sleep)

    tau2_environment._install_tau2_litellm_rate_limit_retry()

    with pytest.raises(RuntimeError, match="AuthenticationError"):
        FakeLLMUtils.generate()
    assert calls["count"] == 1


def test_tau2_litellm_generate_retry_patch_bounds_other_transient_errors(monkeypatch):
    import benchmark.tau2.common.tau2_env.tau2_environment as tau2_environment
    import openviking.utils.model_retry as model_retry
    from openviking.utils.model_retry import ModelRetryExhaustedError

    calls = {"count": 0}

    def fake_generate():
        calls["count"] += 1
        raise RuntimeError("Error code: 503 - service unavailable")

    class FakeLLMUtils:
        generate = staticmethod(fake_generate)

    def fake_import_module(name):
        if name == "tau2.utils.llm_utils":
            return FakeLLMUtils
        raise ImportError(name)

    monkeypatch.setattr(tau2_environment.importlib, "import_module", fake_import_module)
    monkeypatch.setattr(model_retry.time, "sleep", lambda _delay: None)

    tau2_environment._install_tau2_litellm_rate_limit_retry()

    with pytest.raises(ModelRetryExhaustedError, match="service unavailable"):
        FakeLLMUtils.generate()
    assert calls["count"] == 4


def test_tau2_native_env_reward_handles_required_id_and_tool_call_ids(monkeypatch):
    import benchmark.tau2.common.tau2_env.tau2_environment as tau2_environment
    from benchmark.tau2.common.tau2_env.tau2_environment import Tau2BenchEnv

    monkeypatch.setattr(tau2_environment, "AgentGymEnv", None)
    env = Tau2BenchEnv("airline", "1")
    env.reset()
    env.tool_call("get_user_details", {"user_id": "raj_sanchez_7340"})
    env.tool_call("get_reservation_details", {"reservation_id": "Q69X3R"})

    reward, evaluation = env._impl._get_reward()

    assert reward == 1.0
    assert evaluation.reward == 1.0


def test_tau2_gym_env_passes_seed_to_user_llm_before_reset(monkeypatch):
    import benchmark.tau2.common.tau2_env.tau2_environment as tau2_environment

    calls = {}

    class FakeAgentGymEnv:
        def __init__(self, **kwargs):
            calls["init"] = kwargs

        def reset(self, *, seed=None):
            calls["reset_seed"] = seed
            task = SimpleNamespace(evaluation_criteria=[], user_scenario="scenario")
            return "user: hello", {
                "task": task,
                "simulation_run": None,
                "policy": "policy",
                "tools": [],
            }

    monkeypatch.setattr(tau2_environment, "AgentGymEnv", FakeAgentGymEnv)
    monkeypatch.setattr(tau2_environment, "_install_tau2_litellm_rate_limit_retry", lambda: None)
    monkeypatch.setattr(
        tau2_environment,
        "_install_tau2_litellm_unknown_cost_suppression",
        lambda: None,
    )

    env = tau2_environment._GymTau2BenchEnv("airline", "1")
    env.reset(seed=1234)

    assert calls["init"]["user_llm_args"] == {"temperature": 0.0, "seed": 1234}
    assert calls["reset_seed"] == 1234


def test_tau2_gym_env_waits_for_initial_observation_after_agent_turn(monkeypatch):
    import benchmark.tau2.common.tau2_env.tau2_environment as tau2_environment

    task = SimpleNamespace(evaluation_criteria=[], user_scenario="scenario")

    class FakeAgent:
        def __init__(self):
            self._lock = threading.Lock()
            self._observation = ["published after reset"]

        @property
        def observation(self):
            return self._observation

    class FakeAgentGymEnv:
        def __init__(self, **_kwargs):
            self._agent = FakeAgent()
            self._simulation_done = threading.Event()

        def reset(self, *, seed=None):
            del seed
            return "", {
                "task": task,
                "simulation_run": None,
                "policy": "policy",
                "tools": [],
            }

        @staticmethod
        def _format_observation(messages):
            return "user: hello" if messages else ""

    monkeypatch.setattr(tau2_environment, "AgentGymEnv", FakeAgentGymEnv)
    monkeypatch.setattr(tau2_environment, "_install_tau2_litellm_rate_limit_retry", lambda: None)
    monkeypatch.setattr(
        tau2_environment,
        "_install_tau2_litellm_unknown_cost_suppression",
        lambda: None,
    )

    env = tau2_environment._GymTau2BenchEnv("airline", "1")
    env.reset(seed=1234)

    assert env.user_query == "hello"


def test_tau2_gym_env_bounds_wait_for_missing_initial_observation(monkeypatch):
    import benchmark.tau2.common.tau2_env.tau2_environment as tau2_environment

    task = SimpleNamespace(evaluation_criteria=[], user_scenario="scenario")

    class FakeAgent:
        def __init__(self):
            self._lock = threading.Lock()
            self._observation = []

        @property
        def observation(self):
            return self._observation

    class FakeAgentGymEnv:
        def __init__(self, **_kwargs):
            self._agent = FakeAgent()
            self._simulation_done = threading.Event()

        def reset(self, *, seed=None):
            del seed
            return "", {
                "task": task,
                "simulation_run": None,
                "policy": "policy",
                "tools": [],
            }

        @staticmethod
        def _format_observation(messages):
            del messages
            return ""

    monkeypatch.setattr(tau2_environment, "AgentGymEnv", FakeAgentGymEnv)
    monkeypatch.setattr(tau2_environment, "_GYM_INITIAL_OBSERVATION_TIMEOUT_SECONDS", 0.01)
    monkeypatch.setattr(tau2_environment, "_install_tau2_litellm_rate_limit_retry", lambda: None)
    monkeypatch.setattr(
        tau2_environment,
        "_install_tau2_litellm_unknown_cost_suppression",
        lambda: None,
    )

    env = tau2_environment._GymTau2BenchEnv("airline", "1")

    with pytest.raises(RuntimeError, match="initial observation"):
        env.reset(seed=1234)


def test_tau2_gym_env_allows_slow_initial_observation_startup() -> None:
    import benchmark.tau2.common.tau2_env.tau2_environment as tau2_environment

    assert tau2_environment._GYM_INITIAL_OBSERVATION_TIMEOUT_SECONDS == 120.0


def test_tau2_fixed_first_user_simulator_uses_fixture_only_for_first_turn(monkeypatch):
    from tau2.data_model.message import AssistantMessage, UserMessage
    from tau2.user.user_simulator import UserSimulator

    from benchmark.tau2.common.fixed_first_user import FixedFirstUserSimulator

    generated = []

    def fake_generate(self, message, state):
        del self, message, state
        generated.append(True)
        return UserMessage(role="user", content="generated later")

    monkeypatch.setattr(UserSimulator, "_generate_next_message", fake_generate)
    simulator = FixedFirstUserSimulator(
        fixed_first_message="cached first",
        llm="openai/test-user",
        instructions="scenario",
    )
    state = simulator.get_init_state()

    first = simulator.generate_next_message(
        AssistantMessage(role="assistant", content="start"), state
    )[0]
    second = simulator.generate_next_message(
        AssistantMessage(role="assistant", content="continue"), state
    )[0]

    assert first.content == "cached first"
    assert second.content == "generated later"
    assert generated == [True]


def test_tau2_first_user_cache_records_then_replays(tmp_path):
    from benchmark.tau2.train.first_user_cache import FirstUserMessageCache

    cache = FirstUserMessageCache(tmp_path, enabled=True)
    identity = {
        "task_signature": "tau2:airline:train:7:trial:0",
        "seed": 305,
        "user_model": "openai/test-user",
        "temperature": 0.0,
        "scenario_sha256": "scenario-hash",
    }
    fixed_messages = []

    def reset_user(fixed_first_message):
        fixed_messages.append(fixed_first_message)
        return fixed_first_message or "generated first"

    first = cache.run(identity, reset_user)
    second = cache.run(identity, reset_user)

    assert first.hit is False
    assert second.hit is True
    assert first.message == second.message == "generated first"
    assert fixed_messages == [None, "generated first"]
    assert first.path == second.path
    assert first.path.is_file()


def test_tau2_first_user_cache_off_never_reads_or_writes(tmp_path):
    from benchmark.tau2.train.first_user_cache import FirstUserMessageCache

    cache = FirstUserMessageCache(tmp_path, enabled=False)
    generated = iter(["first", "second"])

    first = cache.run({"task_signature": "case"}, lambda _fixed: next(generated))
    second = cache.run({"task_signature": "case"}, lambda _fixed: next(generated))

    assert first.message == "first"
    assert second.message == "second"
    assert first.hit is second.hit is False
    assert list(tmp_path.rglob("*.json")) == []


def test_tau2_first_user_cache_serializes_same_case_miss(tmp_path):
    import time
    from concurrent.futures import ThreadPoolExecutor

    from benchmark.tau2.train.first_user_cache import FirstUserMessageCache

    cache = FirstUserMessageCache(tmp_path, enabled=True)
    identity = {"task_signature": "tau2:airline:train:7:trial:0"}
    fixed_messages = []

    def reset_user(fixed_first_message):
        fixed_messages.append(fixed_first_message)
        if fixed_first_message is None:
            time.sleep(0.05)
        return fixed_first_message or "generated once"

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _index: cache.run(identity, reset_user), range(2)))

    assert sorted(result.hit for result in results) == [False, True]
    assert fixed_messages == [None, "generated once"]


def test_tau2_native_env_records_communication_as_assistant_text(monkeypatch):
    import benchmark.tau2.common.tau2_env.tau2_environment as tau2_environment
    from benchmark.tau2.common.tau2_env.tau2_environment import Tau2BenchEnv

    monkeypatch.setattr(tau2_environment, "AgentGymEnv", None)
    env = Tau2BenchEnv("airline", "3")
    env.reset()
    env.tool_call("communicate_with_user", {"content": "You may bring 4 suitcases."})

    reward, evaluation = env._impl._get_reward()

    assert reward == 1.0
    assert evaluation.communicate_checks[0].met is True


def test_tau2_final_answer_is_appended_for_native_evaluation(monkeypatch):
    import benchmark.tau2.common.tau2_env.tau2_environment as tau2_environment
    from benchmark.tau2.common.tau2_env.tau2_environment import Tau2BenchEnv
    from benchmark.tau2.train.rollout_executor import _append_final_answer_for_tau2_evaluation

    monkeypatch.setattr(tau2_environment, "AgentGymEnv", None)
    env = Tau2BenchEnv("airline", "3")
    env.reset()
    _append_final_answer_for_tau2_evaluation(env, "You may bring 4 suitcases.")

    reward, evaluation = env._impl._get_reward()

    assert reward == 1.0
    assert evaluation.communicate_checks[0].met is True


def test_tau2_configure_tools_removes_only_openviking_tools():
    from benchmark.tau2.train.rollout_executor import _configure_tools
    from benchmark.tau2.train.rollout_executor_vikingbot import (
        normalize_tau2_experience_loader_mode,
    )

    class FakeTools:
        def __init__(self):
            self.tool_names = [
                "read_file",
                "openviking_search",
                "openviking_memory_commit",
                "web_search",
            ]
            self.unregistered = []
            self.registered = []

        def unregister(self, name):
            self.unregistered.append(name)
            self.tool_names.remove(name)

        def register(self, tool):
            self.registered.append(tool.name)

    class FakeAgent:
        def __init__(self):
            self.tools = FakeTools()

    class FakeProvider:
        def list_openai_tools(self):
            return [
                {
                    "type": "function",
                    "function": {
                        "name": "get_user_details",
                        "description": "get user",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ]

        def call_tool(self, name, args):
            return "ok"

    agent = FakeAgent()

    _configure_tools(agent, FakeProvider(), keep_default_tools=True)

    assert agent.tools.unregistered == ["openviking_search", "openviking_memory_commit"]
    assert agent.tools.tool_names == ["read_file", "web_search"]
    assert agent.tools.registered == [
        "search_experience",
        "read_experience",
        "get_user_details",
    ]

    constraint_agent = FakeAgent()
    _configure_tools(
        constraint_agent,
        FakeProvider(),
        keep_default_tools=True,
        loader_mode="constraint",
    )

    assert constraint_agent.tools.registered == ["get_user_details"]
    assert normalize_tau2_experience_loader_mode("direct_experience") == "direct_experience"


def test_tau2_experience_recall_mode_defaults_to_case_exp_rerank_and_validates():
    from benchmark.tau2.train.rollout_executor_vikingbot import (
        DEFAULT_TAU2_EXPERIENCE_RECALL_MODE,
        DEFAULT_TAU2_EXPERIENCE_RERANK_TOP_N,
        VikingBotTau2RolloutExecutor,
        normalize_tau2_experience_recall_mode,
    )

    assert DEFAULT_TAU2_EXPERIENCE_RECALL_MODE == "case_exp_rerank"
    assert DEFAULT_TAU2_EXPERIENCE_RERANK_TOP_N == 3
    assert VikingBotTau2RolloutExecutor().experience_recall_mode == "case_exp_rerank"
    assert VikingBotTau2RolloutExecutor().experience_rerank_top_n == 3
    assert normalize_tau2_experience_recall_mode(" CASE_ANN ") == "case_ann"
    assert normalize_tau2_experience_recall_mode("exp_ann") == "exp_ann"
    assert normalize_tau2_experience_recall_mode(None) == "case_exp_rerank"
    assert normalize_tau2_experience_recall_mode("hybrid_ann") == "hybrid_ann"
    assert normalize_tau2_experience_recall_mode("case_exp_rerank") == "case_exp_rerank"
    with pytest.raises(ValueError, match="experience_recall_mode"):
        normalize_tau2_experience_recall_mode("semantic")
    with pytest.raises(ValueError, match="experience_rerank_top_n"):
        VikingBotTau2RolloutExecutor(experience_rerank_top_n=0)


def test_tau2_case_exp_rerank_trace_records_scope_counts(monkeypatch):
    import json

    import benchmark.tau2.train.rollout_executor_vikingbot as module

    trace_calls = []
    monkeypatch.setattr(
        module.tracer,
        "info",
        lambda message, console=False: trace_calls.append((message, console)),
    )

    module._trace_experience_recall(
        match_type="case_exp_rerank",
        task_signature=None,
        candidates=[
            {
                "experiences": [
                    {"uri": "viking://user/memories/experiences/a.md"},
                    {"uri": "viking://user/memories/experiences/b.md"},
                ]
            }
        ],
        exact_case_found=False,
        selected_case_count=2,
        scoped_experience_count=5,
    )

    message, console = trace_calls[0]
    assert console is True
    assert json.loads(message) == {
        "event": "experience_recall",
        "match_type": "case_exp_rerank",
        "candidate_count": 1,
        "experience_count": 2,
        "exact_case_found": False,
        "selected_case_count": 2,
        "scoped_experience_count": 5,
    }


def test_tau2_openviking_search_trace_records_server_trace_id(monkeypatch):
    import json

    import benchmark.tau2.train.rollout_executor_vikingbot as module

    trace_calls = []
    monkeypatch.setattr(
        module.tracer,
        "info",
        lambda message, console=False: trace_calls.append((message, console)),
    )

    module._trace_openviking_search(
        recall_stage="case_search",
        target_uri="viking://user/u/memories/cases",
        result={"server_trace_id": "1" * 32},
    )

    message, console = trace_calls[0]
    assert console is True
    assert json.loads(message) == {
        "event": "openviking_search_trace",
        "recall_stage": "case_search",
        "server_trace_id": "1" * 32,
        "target_uri": "viking://user/u/memories/cases",
    }


def test_tau2_configure_tools_binds_case_lookup_to_search_experience(monkeypatch):
    import benchmark.tau2.train.rollout_executor_vikingbot as module

    observed = {}

    class FakeTool:
        def __init__(self, name):
            self.name = name

    def fake_make_search_experience_tool(
        case_lookup=None,
        experience_recall_mode=None,
        experience_rerank_top_n=None,
        client=None,
    ):
        observed["case_lookup"] = case_lookup
        observed["experience_recall_mode"] = experience_recall_mode
        observed["experience_rerank_top_n"] = experience_rerank_top_n
        observed["search_client"] = client
        return FakeTool("search_experience")

    def fake_make_read_experience_tool(*, client=None):
        observed["read_client"] = client
        return FakeTool("read_experience")

    monkeypatch.setattr(module, "_make_search_experience_tool", fake_make_search_experience_tool)
    monkeypatch.setattr(module, "_make_read_experience_tool", fake_make_read_experience_tool)

    class FakeTools:
        tool_names = []

        def unregister(self, name):
            raise AssertionError(f"unexpected unregister: {name}")

        def register(self, tool):
            return None

    class FakeAgent:
        tools = FakeTools()

    class FakeProvider:
        def list_openai_tools(self):
            return []

    case_lookup = _tau2_exact_case_lookup()
    experience_client = object()
    module._configure_tools(
        FakeAgent(),
        FakeProvider(),
        keep_default_tools=True,
        case_lookup=case_lookup,
        experience_recall_mode="exp_ann",
        experience_rerank_top_n=5,
        experience_client=experience_client,
    )

    assert observed == {
        "case_lookup": case_lookup,
        "experience_recall_mode": "exp_ann",
        "experience_rerank_top_n": 5,
        "search_client": experience_client,
        "read_client": experience_client,
    }


@pytest.mark.asyncio
async def test_tau2_experience_tools_reuse_injected_client(monkeypatch):
    import json

    import vikingbot.openviking_mount.ov_server as ov_server

    from benchmark.tau2.train.rollout_executor_vikingbot import (
        _make_read_experience_tool,
        _make_search_experience_tool,
    )

    case_uri = "viking://user/u/memories/cases/case.md"
    experience_uri = "viking://user/u/memories/experiences/experience.md"

    class SharedClient:
        def __init__(self):
            self.close_calls = 0

        def _memory_target_uri(self, user_id):
            assert user_id is None
            return "viking://user/u/memories"

        async def search(self, situation, *, target_uri, limit, score_threshold=None):
            assert situation == "Cancel a reservation"
            assert target_uri == "viking://user/u/memories/cases"
            assert limit == 10
            assert score_threshold == 0.0
            return {"memories": [{"uri": case_uri, "score": 0.9}]}

        async def read_content(self, uri, level="read"):
            assert level == "read"
            if uri == case_uri:
                return f"# Case\n\n## Linked Experiences\n- [experience]({experience_uri})\n"
            if uri == experience_uri:
                return "## Situation\n- Applies to reservation cancellation\n"
            return ""

        async def close(self):
            self.close_calls += 1

    class ForbiddenClientFactory:
        @classmethod
        async def create(cls):
            raise AssertionError("injected tools must not create another VikingClient")

    monkeypatch.setattr(ov_server, "VikingClient", ForbiddenClientFactory)
    shared_client = SharedClient()

    search_result = json.loads(
        await _make_search_experience_tool(
            experience_recall_mode="case_ann",
            client=shared_client,
        ).execute(None, situation="Cancel a reservation", limit=2)
    )
    read_result = await _make_read_experience_tool(client=shared_client).execute(
        None,
        experience_uri=experience_uri,
    )

    assert search_result["candidates"][0]["experiences"] == [
        {
            "uri": experience_uri,
            "situation": "- Applies to reservation cancellation",
        }
    ]
    assert "Applies to reservation cancellation" in read_result
    assert shared_client.close_calls == 0


@pytest.mark.asyncio
async def test_tau2_rollout_experience_client_closes_once_after_success(monkeypatch):
    import vikingbot.openviking_mount.ov_server as ov_server

    import benchmark.tau2.train.rollout_executor_vikingbot as module

    client = SimpleNamespace(close_calls=0)

    async def close():
        client.close_calls += 1

    client.close = close

    class FakeClientFactory:
        @classmethod
        async def create(cls):
            return client

    monkeypatch.setattr(ov_server, "VikingClient", FakeClientFactory)

    async def operation(active_client):
        assert active_client is client
        assert client.close_calls == 0
        return "agent result"

    result = await module._run_with_rollout_experience_client(
        loader_mode="skill",
        operation=operation,
    )

    assert result == "agent result"
    assert client.close_calls == 1


@pytest.mark.asyncio
async def test_tau2_rollout_experience_client_closes_once_after_failure(monkeypatch):
    import vikingbot.openviking_mount.ov_server as ov_server

    import benchmark.tau2.train.rollout_executor_vikingbot as module

    client = SimpleNamespace(close_calls=0)

    async def close():
        client.close_calls += 1

    client.close = close

    class FakeClientFactory:
        @classmethod
        async def create(cls):
            return client

    monkeypatch.setattr(ov_server, "VikingClient", FakeClientFactory)

    async def operation(active_client):
        assert active_client is client
        raise RuntimeError("agent failed")

    with pytest.raises(RuntimeError, match="agent failed"):
        await module._run_with_rollout_experience_client(
            loader_mode="skill",
            operation=operation,
        )

    assert client.close_calls == 1


@pytest.mark.asyncio
async def test_tau2_search_experience_summary_only_exposes_case_name_and_experience_snippets():
    from benchmark.tau2.train.rollout_executor_vikingbot import _experience_search_summary

    case_uri = "viking://user/u/memories/cases/case_1.md"
    exp_uri = "viking://user/u/memories/experiences/keep_scope.md"

    class FakeClient:
        async def read_content(self, uri, level="read"):
            assert level == "read"
            if uri == case_uri:
                return (
                    "# case_1\n\n"
                    "## Task Signature\nsecret task signature\n\n"
                    "## Input\nsecret case input\n\n"
                    "## Linked Experiences\n"
                    f"- [keep_scope]({exp_uri})\n"
                )
            if uri == exp_uri:
                return (
                    "## Situation\n"
                    "- Applies when: 用户请求汇总值且后续写入改变对象集合\n"
                    "- Source binding: 用户请求时的对象范围\n\n"
                    "## Reminder\n- keep original scope\n"
                )
            raise AssertionError(f"unexpected uri: {uri}")

    summary = await _experience_search_summary(
        FakeClient(),
        {"uri": case_uri, "score": 0.99, "abstract": "secret abstract"},
        rank=1,
    )

    assert summary == {
        "rank": 1,
        "case_name": "case_1",
        "experiences": [
            {
                "uri": exp_uri,
                "situation": (
                    "- Applies when: 用户请求汇总值且后续写入改变对象集合 "
                    "- Source binding: 用户请求时的对象范围"
                ),
            }
        ],
    }
    assert "case_uri" not in summary
    assert "case_abstract" not in summary
    assert "task_signature" not in summary
    assert "input_summary" not in summary


def test_tau2_search_experience_response_hides_internal_search_metadata():
    import json

    from benchmark.tau2.train.rollout_executor_vikingbot import _format_search_experience_response

    payload = json.loads(
        _format_search_experience_response(
            situation="The user wants to cancel all upcoming reservations.",
            candidates=[
                {
                    "rank": 1,
                    "case_name": "case_1",
                    "experiences": [
                        {
                            "uri": "viking://user/u/memories/experiences/keep_scope.md",
                            "situation": "- Applies when: scope matters",
                        }
                    ],
                }
            ],
        )
    )

    assert payload == {
        "match_type": "semantic",
        "situation": "The user wants to cancel all upcoming reservations.",
        "candidates": [
            {
                "rank": 1,
                "case_name": "case_1",
                "experiences": [
                    {
                        "uri": "viking://user/u/memories/experiences/keep_scope.md",
                        "situation": "- Applies when: scope matters",
                    }
                ],
            }
        ],
    }
    assert "target_uri" not in payload
    assert "count" not in payload
    assert "query" not in payload


def test_tau2_case_memory_context_includes_exact_case_auto_loaded_experiences():
    import json

    from benchmark.tau2.train.rollout_executor_vikingbot import (
        _case_memory_context_from_tools,
    )

    exp_uri = "viking://user/u/memories/experiences/keep_scope.md"
    result = json.dumps(
        {
            "match_type": "exact_case",
            "candidates": [
                {
                    "experiences": [
                        {"uri": exp_uri, "content": "# Keep scope\n\nUse the original scope."}
                    ]
                }
            ],
        }
    )

    context = _case_memory_context_from_tools(
        [
            {
                "tool_name": "search_experience",
                "args": json.dumps({"task_signature": "tau2:airline:train:39"}),
                "result": result,
            }
        ]
    )

    assert exp_uri in context
    assert "# Keep scope" in context


@pytest.mark.asyncio
async def test_tau2_search_experience_uses_declarative_situation(monkeypatch):
    import json

    import vikingbot.openviking_mount.ov_server as ov_server

    from benchmark.tau2.train.rollout_executor_vikingbot import _make_search_experience_tool

    observed = {}

    class FakeClient:
        @classmethod
        async def create(cls):
            return cls()

        def _memory_target_uri(self, uri):
            assert uri is None
            return "viking://user/u/memories"

        async def search(self, situation, *, target_uri, limit, score_threshold=None):
            observed.update(
                situation=situation,
                target_uri=target_uri,
                limit=limit,
                score_threshold=score_threshold,
            )
            return {"memories": []}

        async def close(self):
            observed["closed"] = True

    monkeypatch.setattr(ov_server, "VikingClient", FakeClient)
    tool = _make_search_experience_tool(experience_recall_mode="case_ann")

    assert tool.parameters["required"] == ["situation"]
    assert "situation" in tool.parameters["properties"]
    assert "task_signature" in tool.parameters["properties"]
    assert "query" not in tool.parameters["properties"]
    description = tool.parameters["properties"]["situation"]["description"]
    assert "current conversation" in description
    assert "keyword" in description

    payload = json.loads(
        await tool.execute(
            None,
            situation="The user wants to cancel all upcoming reservations.",
            limit=3,
        )
    )

    assert observed == {
        "situation": "The user wants to cancel all upcoming reservations.",
        "target_uri": "viking://user/u/memories/cases",
        "limit": 10,
        "score_threshold": 0.0,
        "closed": True,
    }
    assert payload == {
        "match_type": "semantic",
        "situation": "The user wants to cancel all upcoming reservations.",
        "candidates": [],
    }


@pytest.mark.asyncio
async def test_tau2_search_experience_case_ann_searches_ten_and_expands_default_top_two(
    monkeypatch,
):
    import json

    import vikingbot.openviking_mount.ov_server as ov_server

    from benchmark.tau2.train.rollout_executor_vikingbot import _make_search_experience_tool

    case_uris = [f"viking://user/u/memories/cases/case_{index}.md" for index in range(1, 4)]
    exp_uris = [
        f"viking://user/u/memories/experiences/experience_{index}.md" for index in range(1, 4)
    ]

    class FakeClient:
        search_calls = []
        read_uris = []

        @classmethod
        async def create(cls):
            return cls()

        def _memory_target_uri(self, uri):
            assert uri is None
            return "viking://user/u/memories"

        async def search(self, situation, *, target_uri, limit, score_threshold=None):
            self.search_calls.append((situation, target_uri, limit, score_threshold))
            return {
                "memories": [
                    {
                        "uri": case_uri,
                        "score": 1.0 - index / 10,
                        "abstract": f"Case {index}",
                    }
                    for index, case_uri in enumerate(case_uris, start=1)
                ]
            }

        async def read_content(self, uri, level="read"):
            assert level == "read"
            self.read_uris.append(uri)
            if uri in case_uris:
                index = case_uris.index(uri)
                return (
                    f"# case_{index + 1}\n\n"
                    "## Linked Experiences\n"
                    f"- [experience_{index + 1}]({exp_uris[index]})\n"
                )
            if uri in exp_uris:
                index = exp_uris.index(uri)
                return f"## Situation\n- Applies to ranked case {index + 1}\n"
            raise AssertionError(f"unexpected uri: {uri}")

        async def close(self):
            return None

    monkeypatch.setattr(ov_server, "VikingClient", FakeClient)
    tool = _make_search_experience_tool(experience_recall_mode="case_ann")

    payload = json.loads(await tool.execute(None, situation="A ranked airline request"))

    assert FakeClient.search_calls == [
        (
            "A ranked airline request",
            "viking://user/u/memories/cases",
            10,
            0.0,
        )
    ]
    assert [candidate["case_name"] for candidate in payload["candidates"]] == [
        "case_1",
        "case_2",
    ]
    assert case_uris[2] not in FakeClient.read_uris
    assert exp_uris[2] not in FakeClient.read_uris


@pytest.mark.asyncio
async def test_tau2_search_experience_case_exp_rerank_reranks_all_linked_experiences(
    monkeypatch,
):
    import json

    import vikingbot.openviking_mount.ov_server as ov_server

    from benchmark.tau2.train.rollout_executor_vikingbot import _make_search_experience_tool

    situation = "The user needs a multi-step airline change."
    cases_uri = "viking://user/u/memories/cases"
    case_a_uri = f"{cases_uri}/case_a.md"
    case_b_uri = f"{cases_uri}/case_b.md"
    exp_a_uri = "viking://user/u/memories/experiences/exp_a.md"
    exp_shared_uri = "viking://user/u/memories/experiences/exp_shared.md"
    exp_b_uri = "viking://user/u/memories/experiences/exp_b.md"

    class FakeClient:
        search_calls = []
        read_uris = []

        @classmethod
        async def create(cls):
            return cls()

        def _memory_target_uri(self, uri):
            assert uri is None
            return "viking://user/u/memories"

        async def search(
            self,
            query,
            *,
            target_uri,
            limit,
            score_threshold=None,
            filter=None,
        ):
            self.search_calls.append((query, target_uri, limit, score_threshold, filter))
            if target_uri == cases_uri:
                return {
                    "memories": [{"uri": case_a_uri}, {"uri": case_b_uri}],
                    "server_trace_id": "1" * 32,
                }
            assert target_uri == "viking://user/u/memories/experiences"
            return {
                "memories": [
                    {"uri": exp_b_uri},
                    {"uri": exp_a_uri},
                    {"uri": exp_shared_uri},
                ],
                "server_trace_id": "2" * 32,
            }

        async def read_content(self, uri, level="read"):
            assert level == "read"
            self.read_uris.append(uri)
            if uri == case_a_uri:
                return (
                    "# case_a\n\n"
                    "## Linked Experiences\n"
                    f"- [exp_a]({exp_a_uri})\n"
                    f"- [exp_shared]({exp_shared_uri})\n"
                )
            if uri == case_b_uri:
                return (
                    "# case_b\n\n"
                    "## Linked Experiences\n"
                    f"- [exp_shared]({exp_shared_uri})\n"
                    f"- [exp_b]({exp_b_uri})\n"
                )
            if uri == exp_a_uri:
                return "## Situation\n- Applies to exp A\n"
            if uri == exp_b_uri:
                return "## Situation\n- Applies to exp B\n"
            if uri == exp_shared_uri:
                return "## Situation\n- Applies to shared exp\n"
            raise AssertionError(f"unexpected uri: {uri}")

        async def close(self):
            return None

    import benchmark.tau2.train.rollout_executor_vikingbot as module

    trace_messages = []
    monkeypatch.setattr(ov_server, "VikingClient", FakeClient)
    monkeypatch.setattr(
        module.tracer,
        "info",
        lambda message, console=False: trace_messages.append(message),
    )
    tool = _make_search_experience_tool(
        experience_recall_mode="case_exp_rerank",
        experience_rerank_top_n=2,
    )

    payload = json.loads(await tool.execute(None, situation=situation, limit=2))

    assert FakeClient.search_calls == [
        (situation, cases_uri, 10, 0.0, None),
        (
            situation,
            "viking://user/u/memories/experiences",
            3,
            0.0,
            {
                "op": "must",
                "field": "uri",
                "conds": [exp_a_uri, exp_shared_uri, exp_b_uri],
            },
        ),
    ]
    assert payload == {
        "match_type": "case_exp_rerank",
        "situation": situation,
        "candidates": [
            {
                "rank": 1,
                "case_name": "case_exp_rerank",
                "experiences": [
                    {"uri": exp_b_uri, "situation": "- Applies to exp B"},
                    {"uri": exp_a_uri, "situation": "- Applies to exp A"},
                ],
            }
        ],
    }
    assert exp_shared_uri not in FakeClient.read_uris
    search_trace_events = [
        event
        for message in trace_messages
        if (event := json.loads(message)).get("event") == "openviking_search_trace"
    ]
    assert search_trace_events == [
        {
            "event": "openviking_search_trace",
            "recall_stage": "case_search",
            "server_trace_id": "1" * 32,
            "target_uri": cases_uri,
        },
        {
            "event": "openviking_search_trace",
            "recall_stage": "experience_search",
            "server_trace_id": "2" * 32,
            "target_uri": "viking://user/u/memories/experiences",
        },
    ]


@pytest.mark.asyncio
async def test_tau2_search_experience_case_exp_rerank_exact_case_returns_all_linked_content(
    monkeypatch,
):
    import json

    import vikingbot.openviking_mount.ov_server as ov_server

    import benchmark.tau2.train.rollout_executor_vikingbot as module

    situation = "The current train task has an exact Case."
    case_uri = "viking://user/u/memories/cases/tau2_airline_train_22.md"
    exp_a_uri = "viking://user/u/memories/experiences/exp_a.md"
    exp_b_uri = "viking://user/u/memories/experiences/exp_b.md"

    class FakeClient:
        read_uris = []

        @classmethod
        async def create(cls):
            return cls()

        def _memory_target_uri(self, uri):
            assert uri is None
            return "viking://user/u/memories"

        async def search(
            self,
            query,
            *,
            target_uri,
            limit,
            score_threshold=None,
            filter=None,
        ):
            raise AssertionError(
                "deterministic exact Case must not run Case or Experience search/rerank"
            )

        async def read_content(self, uri, level="read"):
            assert level == "read"
            self.read_uris.append(uri)
            if uri == case_uri:
                return _tau2_exact_case_content_with_links([exp_b_uri, exp_a_uri])
            if uri == exp_a_uri:
                return "## Situation\n- Exact exp A\n"
            if uri == exp_b_uri:
                return "## Situation\n- Exact exp B\n"
            return ""

        async def close(self):
            return None

    monkeypatch.setattr(ov_server, "VikingClient", FakeClient)
    tool = module._make_search_experience_tool(
        case_lookup=_tau2_exact_case_lookup(),
        experience_recall_mode="case_exp_rerank",
        experience_rerank_top_n=1,
    )

    payload = json.loads(
        await tool.execute(
            None,
            situation=situation,
            task_signature="tau2:airline:train:39",
            limit=2,
        )
    )

    assert payload["match_type"] == "exact_case"
    assert payload["candidates"][0]["experiences"] == [
        {"uri": exp_a_uri, "content": "## Situation\n- Exact exp A"},
        {"uri": exp_b_uri, "content": "## Situation\n- Exact exp B"},
    ]
    assert FakeClient.read_uris.count(case_uri) == 1


@pytest.mark.asyncio
async def test_tau2_search_experience_exact_case_retries_empty_linked_experience_read(
    monkeypatch,
):
    import json

    import vikingbot.openviking_mount.ov_server as ov_server

    import benchmark.tau2.train.rollout_executor_vikingbot as module

    case_uri = "viking://user/u/memories/cases/tau2_airline_train_22.md"
    exp_uri = "viking://user/u/memories/experiences/exp.md"

    class FakeClient:
        read_counts = {}

        @classmethod
        async def create(cls):
            return cls()

        def _memory_target_uri(self, uri):
            assert uri is None
            return "viking://user/u/memories"

        async def search(self, *args, **kwargs):
            raise AssertionError("exact Case must not search")

        async def read_content(self, uri, level="read"):
            assert level == "read"
            self.read_counts[uri] = self.read_counts.get(uri, 0) + 1
            if uri == case_uri:
                return _tau2_exact_case_content_with_links([exp_uri])
            if uri == exp_uri and self.read_counts[uri] == 1:
                return ""
            if uri == exp_uri:
                return "## Situation\n- Available after retry\n"
            raise AssertionError(f"unexpected uri: {uri}")

        async def close(self):
            return None

    monkeypatch.setattr(ov_server, "VikingClient", FakeClient)
    tool = module._make_search_experience_tool(
        case_lookup=_tau2_exact_case_lookup(),
        experience_recall_mode="case_exp_rerank",
    )

    payload = json.loads(
        await tool.execute(
            None,
            situation="The current train task has an exact Case.",
            task_signature="tau2:airline:train:39",
        )
    )

    assert payload["candidates"][0]["experiences"] == [
        {"uri": exp_uri, "content": "## Situation\n- Available after retry"}
    ]
    assert FakeClient.read_counts == {case_uri: 1, exp_uri: 2}


@pytest.mark.asyncio
async def test_tau2_search_experience_exact_case_traces_persistent_empty_experience_read(
    monkeypatch,
):
    import json

    import vikingbot.openviking_mount.ov_server as ov_server

    import benchmark.tau2.train.rollout_executor_vikingbot as module

    case_uri = "viking://user/u/memories/cases/tau2_airline_train_22.md"
    exp_uri = "viking://user/u/memories/experiences/missing.md"
    trace_messages = []

    class FakeClient:
        read_counts = {}

        @classmethod
        async def create(cls):
            return cls()

        def _memory_target_uri(self, uri):
            assert uri is None
            return "viking://user/u/memories"

        async def search(self, *args, **kwargs):
            raise AssertionError("exact Case must not search")

        async def read_content(self, uri, level="read"):
            assert level == "read"
            self.read_counts[uri] = self.read_counts.get(uri, 0) + 1
            if uri == case_uri:
                return _tau2_exact_case_content_with_links([exp_uri])
            if uri == exp_uri:
                return ""
            raise AssertionError(f"unexpected uri: {uri}")

        async def close(self):
            return None

    monkeypatch.setattr(ov_server, "VikingClient", FakeClient)
    monkeypatch.setattr(
        module.tracer,
        "info",
        lambda message, console=False: trace_messages.append(message),
    )
    tool = module._make_search_experience_tool(
        case_lookup=_tau2_exact_case_lookup(),
        experience_recall_mode="case_exp_rerank",
    )

    payload = json.loads(
        await tool.execute(
            None,
            situation="The current train task has an exact Case.",
            task_signature="tau2:airline:train:39",
        )
    )

    diagnostics = [
        json.loads(message)
        for message in trace_messages
        if json.loads(message).get("event") == "exact_case_experience_read"
    ]
    assert payload["candidates"][0]["experiences"] == []
    assert FakeClient.read_counts == {case_uri: 1, exp_uri: 2}
    assert diagnostics == [
        {
            "event": "exact_case_experience_read",
            "case_uri": case_uri,
            "experience_uri": exp_uri,
            "attempts": 2,
            "outcome": "empty",
        }
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("error", "outcome"),
    [
        (FileNotFoundError("missing"), "not_found"),
        (NotFoundError("missing.md", "file"), "not_found"),
        (RuntimeError("backend unavailable"), "exception"),
    ],
)
async def test_viking_client_read_content_traces_failure_kind(monkeypatch, error, outcome):
    import json

    import vikingbot.openviking_mount.ov_server as ov_server

    uri = "viking://user/u/memories/experiences/missing.md"
    warnings = []

    class FakeReadClient:
        async def read(self, read_uri):
            assert read_uri == uri
            raise error

    fake_client = SimpleNamespace(
        client=FakeReadClient(),
        _owner_user_id_for_uri=lambda _uri: None,
    )
    monkeypatch.setattr(ov_server, "logger", SimpleNamespace(warning=warnings.append))

    content = await ov_server.VikingClient.read_content(fake_client, uri, level="read")

    assert content == ""
    assert [json.loads(message) for message in warnings] == [
        {
            "event": "viking_read_content_failed",
            "uri": uri,
            "level": "read",
            "outcome": outcome,
            "error_type": type(error).__name__,
            "error": str(error),
        }
    ]


@pytest.mark.asyncio
async def test_tau2_search_experience_case_exp_rerank_skips_empty_experience_scope(monkeypatch):
    import json

    import vikingbot.openviking_mount.ov_server as ov_server

    from benchmark.tau2.train.rollout_executor_vikingbot import _make_search_experience_tool

    case_uri = "viking://user/u/memories/cases/no_experience.md"

    class FakeClient:
        search_calls = []

        @classmethod
        async def create(cls):
            return cls()

        def _memory_target_uri(self, uri):
            assert uri is None
            return "viking://user/u/memories"

        async def search(self, query, *, target_uri, limit, score_threshold=None):
            self.search_calls.append((query, target_uri, limit, score_threshold))
            return {"memories": [{"uri": case_uri}]}

        async def read_content(self, uri, level="read"):
            assert uri == case_uri
            assert level == "read"
            return "# no_experience\n\n## Linked Experiences\n"

        async def close(self):
            return None

    monkeypatch.setattr(ov_server, "VikingClient", FakeClient)
    tool = _make_search_experience_tool(experience_recall_mode="case_exp_rerank")

    payload = json.loads(await tool.execute(None, situation="A task without linked experience"))

    assert len(FakeClient.search_calls) == 1
    assert payload == {
        "match_type": "case_exp_rerank",
        "situation": "A task without linked experience",
        "candidates": [],
    }


@pytest.mark.asyncio
async def test_tau2_search_experience_case_exp_rerank_surfaces_scoped_search_error(monkeypatch):
    import vikingbot.openviking_mount.ov_server as ov_server

    from benchmark.tau2.train.rollout_executor_vikingbot import _make_search_experience_tool

    case_uri = "viking://user/u/memories/cases/case_a.md"
    exp_uri = "viking://user/u/memories/experiences/exp_a.md"

    class FakeClient:
        @classmethod
        async def create(cls):
            return cls()

        def _memory_target_uri(self, uri):
            assert uri is None
            return "viking://user/u/memories"

        async def search(
            self,
            query,
            *,
            target_uri,
            limit,
            score_threshold=None,
            filter=None,
        ):
            if filter is not None:
                raise RuntimeError("scoped search unavailable")
            return {"memories": [{"uri": case_uri}]}

        async def read_content(self, uri, level="read"):
            assert uri == case_uri
            assert level == "read"
            return f"# case_a\n\n## Linked Experiences\n- [exp_a]({exp_uri})\n"

        async def close(self):
            return None

    monkeypatch.setattr(ov_server, "VikingClient", FakeClient)
    tool = _make_search_experience_tool(experience_recall_mode="case_exp_rerank")

    result = await tool.execute(None, situation="A task whose scoped search fails")

    assert result == "Error searching experience candidates: scoped search unavailable"


@pytest.mark.asyncio
async def test_tau2_search_experience_exp_ann_searches_experience_tree(monkeypatch):
    import json

    import vikingbot.openviking_mount.ov_server as ov_server

    from benchmark.tau2.train.rollout_executor_vikingbot import _make_search_experience_tool

    exp_uri = "viking://user/u/memories/experiences/direct_hit.md"
    observed = {}

    class FakeClient:
        @classmethod
        async def create(cls):
            return cls()

        def _memory_target_uri(self, uri):
            assert uri is None
            return "viking://user/u/memories"

        async def search(self, situation, *, target_uri, limit):
            observed.update(situation=situation, target_uri=target_uri, limit=limit)
            return {"memories": [{"uri": exp_uri, "score": 0.91}]}

        async def read_content(self, uri, level="read"):
            assert uri == exp_uri
            assert level == "read"
            return "## Situation\n- Applies when: direct Experience ANN matches\n\n## Reminder\nUse it."

        async def close(self):
            observed["closed"] = True

    monkeypatch.setattr(ov_server, "VikingClient", FakeClient)
    tool = _make_search_experience_tool(experience_recall_mode="exp_ann")

    payload = json.loads(await tool.execute(None, situation="A matching situation", limit=2))

    assert observed == {
        "situation": "A matching situation",
        "target_uri": "viking://user/u/memories/experiences",
        "limit": 10,
        "closed": True,
    }
    assert payload == {
        "match_type": "exp_ann",
        "situation": "A matching situation",
        "candidates": [
            {
                "rank": 1,
                "case_name": "exp_ann",
                "experiences": [
                    {
                        "uri": exp_uri,
                        "situation": "- Applies when: direct Experience ANN matches",
                    }
                ],
            }
        ],
    }


@pytest.mark.asyncio
async def test_tau2_search_experience_hybrid_ann_applies_semantic_case_prior(monkeypatch):
    import json

    import vikingbot.openviking_mount.ov_server as ov_server

    from benchmark.tau2.train.rollout_executor_vikingbot import _make_search_experience_tool

    exp_a = "viking://user/u/memories/experiences/direct_first.md"
    exp_b = "viking://user/u/memories/experiences/case_boosted.md"
    case_uri = "viking://user/u/memories/cases/relevant_case.md"
    search_calls = []

    class FakeClient:
        @classmethod
        async def create(cls):
            return cls()

        def _memory_target_uri(self, uri):
            assert uri is None
            return "viking://user/u/memories"

        async def search(self, situation, *, target_uri, limit):
            search_calls.append((situation, target_uri, limit))
            if target_uri.endswith("/experiences"):
                return {"memories": [{"uri": exp_a}, {"uri": exp_b}]}
            if target_uri.endswith("/cases"):
                return {"memories": [{"uri": case_uri}]}
            raise AssertionError(target_uri)

        async def read_content(self, uri, level="read"):
            assert level == "read"
            if uri == case_uri:
                return f"# relevant_case\n\n## Linked Experiences\n- [boosted]({exp_b})\n"
            if uri == exp_a:
                return "## Situation\n- Applies to direct first\n"
            if uri == exp_b:
                return "## Situation\n- Applies to case boosted\n"
            return ""

        async def close(self):
            return None

    monkeypatch.setattr(ov_server, "VikingClient", FakeClient)
    tool = _make_search_experience_tool(experience_recall_mode="hybrid_ann")

    payload = json.loads(await tool.execute(None, situation="A related task", limit=1))

    assert set(search_calls) == {
        ("A related task", "viking://user/u/memories/experiences", 10),
        ("A related task", "viking://user/u/memories/cases", 1),
    }
    assert payload["match_type"] == "hybrid_ann"
    assert payload["candidates"] == [
        {
            "rank": 1,
            "case_name": "hybrid_ann",
            "experiences": [
                {"uri": exp_b, "situation": "- Applies to case boosted"},
                {"uri": exp_a, "situation": "- Applies to direct first"},
            ],
        }
    ]


@pytest.mark.asyncio
async def test_tau2_search_experience_hybrid_ann_exact_case_boosts_linked_experience(
    monkeypatch,
):
    import json

    import vikingbot.openviking_mount.ov_server as ov_server

    import benchmark.tau2.train.rollout_executor_vikingbot as module

    case_uri = "viking://user/u/memories/cases/tau2_airline_train_22.md"
    direct_uri = "viking://user/u/memories/experiences/direct_first.md"
    exact_uri = "viking://user/u/memories/experiences/exact_case.md"
    search_calls = []
    trace_messages = []
    monkeypatch.setattr(
        module,
        "tracer",
        SimpleNamespace(
            info=lambda message, console=False: trace_messages.append(json.loads(message))
        ),
        raising=False,
    )

    class FakeClient:
        @classmethod
        async def create(cls):
            return cls()

        def _memory_target_uri(self, uri):
            assert uri is None
            return "viking://user/u/memories"

        async def search(self, situation, *, target_uri, limit):
            search_calls.append((situation, target_uri, limit))
            assert target_uri.endswith("/experiences")
            return {"memories": [{"uri": direct_uri}]}

        async def read_content(self, uri, level="read"):
            assert level == "read"
            if uri == case_uri:
                return _tau2_exact_case_content(linked_experience_uri=exact_uri)
            if uri == exact_uri:
                return "## Situation\n- Exact Case guidance\n"
            if uri == direct_uri:
                return "## Situation\n- Direct ANN guidance\n"
            return ""

        async def close(self):
            return None

    monkeypatch.setattr(ov_server, "VikingClient", FakeClient)
    tool = module._make_search_experience_tool(
        case_lookup=_tau2_exact_case_lookup(),
        experience_recall_mode="hybrid_ann",
    )

    payload = json.loads(
        await tool.execute(
            None,
            situation="A matching task",
            task_signature="tau2:airline:train:39",
            limit=2,
        )
    )

    assert search_calls == [("A matching task", "viking://user/u/memories/experiences", 10)]
    assert payload["candidates"][0]["experiences"] == [
        {"uri": exact_uri, "content": "## Situation\n- Exact Case guidance"},
        {"uri": direct_uri, "situation": "- Direct ANN guidance"},
    ]
    assert trace_messages[0]["exact_case_found"] is True


def _tau2_exact_case_lookup() -> dict:
    return {
        "strict": True,
        "case_names": ["tau2_airline_train_22"],
        "domain": "airline",
        "split": "train",
        "data_split": "airline_train",
        "task_no": "22",
        "task_id": "39",
        "case_name": "tau2_airline_train_22",
        "task_signature": "tau2:airline:train:39",
        "expected_fields": {
            "input.domain": "airline",
            "input.split": "train",
            "input.data_split": "airline_train",
            "input.task_no": "22",
            "input.task_id": "39",
        },
    }


def _tau2_exact_case_content(*, linked_experience_uri: str | None = None) -> str:
    linked = f"- [cancel_without_refund]({linked_experience_uri})" if linked_experience_uri else ""
    return (
        "# tau2_airline_train_22\n\n"
        "## Task Signature\n"
        "tau2:airline:train:39\n\n"
        "## Input\n"
        '{"domain":"airline","split":"train","data_split":"airline_train",'
        '"task_no":22,"task_id":"39"}\n\n'
        "## Linked Experiences\n"
        f"{linked}\n"
    )


def _tau2_exact_case_content_with_links(linked_experience_uris: list[str]) -> str:
    linked = "\n".join(
        f"- [experience_{index}]({uri})"
        for index, uri in enumerate(linked_experience_uris, start=1)
    )
    return (
        "# tau2_airline_train_22\n\n"
        "## Task Signature\n"
        "tau2:airline:train:39\n\n"
        "## Input\n"
        '{"domain":"airline","split":"train","data_split":"airline_train",'
        '"task_no":22,"task_id":"39"}\n\n'
        "## Linked Experiences\n"
        f"{linked}\n"
    )


@pytest.mark.asyncio
async def test_tau2_search_experience_returns_exact_case_without_semantic_search(monkeypatch):
    import json

    import vikingbot.openviking_mount.ov_server as ov_server

    import benchmark.tau2.train.rollout_executor_vikingbot as module

    trace_messages = []
    monkeypatch.setattr(
        module,
        "tracer",
        SimpleNamespace(info=lambda message, console=False: trace_messages.append(message)),
        raising=False,
    )

    case_uri = "viking://user/u/memories/cases/tau2_airline_train_22.md"
    exp_uri = "viking://user/u/memories/experiences/cancel_without_refund.md"

    class FakeClient:
        search_calls = []

        @classmethod
        async def create(cls):
            return cls()

        def _memory_target_uri(self, uri):
            assert uri is None
            return "viking://user/u/memories"

        async def read_content(self, uri, level="read"):
            assert level == "read"
            if uri == case_uri:
                return _tau2_exact_case_content(linked_experience_uri=exp_uri)
            if uri == exp_uri:
                return "## Situation\n- Applies when: the user accepts no refund\n"
            return ""

        async def search(self, situation, *, target_uri, limit):
            self.search_calls.append((situation, target_uri, limit))
            return {"memories": []}

        async def close(self):
            return None

    monkeypatch.setattr(ov_server, "VikingClient", FakeClient)
    tool = module._make_search_experience_tool(
        case_lookup=_tau2_exact_case_lookup(),
        experience_recall_mode="case_ann",
    )

    payload = json.loads(
        await tool.execute(
            None,
            situation="The user wants to cancel all upcoming reservations.",
            task_signature="tau2:airline:train:39",
        )
    )

    assert payload == {
        "match_type": "exact_case",
        "task_signature": "tau2:airline:train:39",
        "situation": "The user wants to cancel all upcoming reservations.",
        "candidates": [
            {
                "rank": 1,
                "case_name": "tau2_airline_train_22",
                "experiences": [
                    {
                        "uri": exp_uri,
                        "content": "## Situation\n- Applies when: the user accepts no refund",
                    }
                ],
            }
        ],
    }
    assert FakeClient.search_calls == []
    assert [json.loads(message) for message in trace_messages] == [
        {
            "event": "experience_recall",
            "match_type": "exact_case",
            "task_signature": "tau2:airline:train:39",
            "exact_case_found": True,
            "candidate_count": 1,
            "experience_count": 1,
        }
    ]


@pytest.mark.asyncio
async def test_tau2_search_experience_exact_case_loads_unique_experiences_in_uri_order(
    monkeypatch,
):
    import json

    import vikingbot.openviking_mount.ov_server as ov_server

    from benchmark.tau2.train.rollout_executor_vikingbot import _make_search_experience_tool

    case_uri = "viking://user/u/memories/cases/tau2_airline_train_22.md"
    first_uri = "viking://user/u/memories/experiences/a_first.md"
    second_uri = "viking://user/u/memories/experiences/z_second.md"
    reads = []

    class FakeClient:
        @classmethod
        async def create(cls):
            return cls()

        def _memory_target_uri(self, uri):
            assert uri is None
            return "viking://user/u/memories"

        async def read_content(self, uri, level="read"):
            assert level == "read"
            reads.append(uri)
            if uri == case_uri:
                return _tau2_exact_case_content_with_links([second_uri, first_uri, second_uri])
            return {
                first_uri: "# First experience\n\nUse the first rule.",
                second_uri: "# Second experience\n\nUse the second rule.",
            }.get(uri, "")

        async def search(self, situation, *, target_uri, limit):
            raise AssertionError("exact case lookup must not use semantic search")

        async def close(self):
            return None

    monkeypatch.setattr(ov_server, "VikingClient", FakeClient)

    payload = json.loads(
        await _make_search_experience_tool(
            case_lookup=_tau2_exact_case_lookup(),
            experience_recall_mode="case_ann",
        ).execute(
            None,
            situation="Cancel upcoming reservations.",
            task_signature="tau2:airline:train:39",
        )
    )

    assert payload["candidates"][0]["experiences"] == [
        {"uri": first_uri, "content": "# First experience\n\nUse the first rule."},
        {"uri": second_uri, "content": "# Second experience\n\nUse the second rule."},
    ]
    assert reads.count(first_uri) == 1
    assert reads.count(second_uri) == 1


@pytest.mark.asyncio
async def test_tau2_search_experience_returns_exact_empty_case_without_semantic_search(
    monkeypatch,
):
    import json

    import vikingbot.openviking_mount.ov_server as ov_server

    from benchmark.tau2.train.rollout_executor_vikingbot import _make_search_experience_tool

    case_uri = "viking://user/u/memories/cases/tau2_airline_train_22.md"

    class FakeClient:
        search_calls = []

        @classmethod
        async def create(cls):
            return cls()

        def _memory_target_uri(self, uri):
            assert uri is None
            return "viking://user/u/memories"

        async def read_content(self, uri, level="read"):
            assert level == "read"
            return _tau2_exact_case_content() if uri == case_uri else ""

        async def search(self, situation, *, target_uri, limit):
            self.search_calls.append((situation, target_uri, limit))
            return {"memories": []}

        async def close(self):
            return None

    monkeypatch.setattr(ov_server, "VikingClient", FakeClient)
    tool = _make_search_experience_tool(
        case_lookup=_tau2_exact_case_lookup(),
        experience_recall_mode="case_ann",
    )

    payload = json.loads(
        await tool.execute(
            None,
            situation="The user wants to cancel all upcoming reservations.",
            task_signature="tau2:airline:train:39",
        )
    )

    assert payload["match_type"] == "exact_case"
    assert payload["candidates"] == [
        {
            "rank": 1,
            "case_name": "tau2_airline_train_22",
            "experiences": [],
        }
    ]
    assert FakeClient.search_calls == []


@pytest.mark.asyncio
async def test_tau2_search_experience_falls_back_when_task_signature_case_file_is_missing(
    monkeypatch,
):
    import json

    import vikingbot.openviking_mount.ov_server as ov_server

    from benchmark.tau2.train.rollout_executor_vikingbot import _make_search_experience_tool

    semantic_case_uri = "viking://user/u/memories/cases/semantic_case.md"

    class FakeClient:
        search_calls = []

        @classmethod
        async def create(cls):
            return cls()

        def _memory_target_uri(self, uri):
            assert uri is None
            return "viking://user/u/memories"

        async def read_content(self, uri, level="read"):
            assert level == "read"
            if uri == semantic_case_uri:
                return "# semantic_case\n\n## Linked Experiences\n"
            return ""

        async def search(self, situation, *, target_uri, limit, score_threshold=None):
            self.search_calls.append((situation, target_uri, limit, score_threshold))
            return {"memories": [{"uri": semantic_case_uri, "score": 0.8}]}

        async def close(self):
            return None

    monkeypatch.setattr(ov_server, "VikingClient", FakeClient)
    tool = _make_search_experience_tool(
        case_lookup=_tau2_exact_case_lookup(),
        experience_recall_mode="case_ann",
    )

    payload = json.loads(
        await tool.execute(
            None,
            situation="The user wants to cancel all upcoming reservations.",
            task_signature="tau2:airline:train:39",
            limit=2,
        )
    )

    assert payload["match_type"] == "semantic"
    assert payload["fallback_reason"] == "task_signature_not_found"
    assert payload["candidates"][0]["case_name"] == "semantic_case"
    assert FakeClient.search_calls == [
        (
            "The user wants to cancel all upcoming reservations.",
            "viking://user/u/memories/cases",
            10,
            0.0,
        )
    ]


@pytest.mark.asyncio
async def test_tau2_search_experience_deduplicates_experiences_across_semantic_cases(
    monkeypatch,
):
    import json

    import vikingbot.openviking_mount.ov_server as ov_server

    from benchmark.tau2.train.rollout_executor_vikingbot import _make_search_experience_tool

    case_uris = [
        "viking://user/u/memories/cases/case_one.md",
        "viking://user/u/memories/cases/case_two.md",
    ]
    exp_uri = "viking://user/u/memories/experiences/shared.md"

    class FakeClient:
        @classmethod
        async def create(cls):
            return cls()

        def _memory_target_uri(self, uri):
            assert uri is None
            return "viking://user/u/memories"

        async def read_content(self, uri, level="read"):
            assert level == "read"
            if uri in case_uris:
                return f"# case\n\n## Linked Experiences\n- [shared]({exp_uri})\n"
            if uri == exp_uri:
                return "## Situation\n- Applies to both cases\n"
            return ""

        async def search(self, situation, *, target_uri, limit, score_threshold=None):
            assert score_threshold == 0.0
            return {"memories": [{"uri": uri, "score": 0.9} for uri in case_uris]}

        async def close(self):
            return None

    monkeypatch.setattr(ov_server, "VikingClient", FakeClient)
    tool = _make_search_experience_tool(experience_recall_mode="case_ann")

    payload = json.loads(await tool.execute(None, situation="A related situation", limit=2))

    returned_experiences = [
        experience for candidate in payload["candidates"] for experience in candidate["experiences"]
    ]
    assert returned_experiences == [
        {
            "uri": exp_uri,
            "situation": "- Applies to both cases",
        }
    ]


@pytest.mark.asyncio
async def test_tau2_search_experience_returns_error_when_client_creation_fails(monkeypatch):
    import vikingbot.openviking_mount.ov_server as ov_server

    from benchmark.tau2.train.rollout_executor_vikingbot import _make_search_experience_tool

    class FakeClient:
        @classmethod
        async def create(cls):
            raise RuntimeError("service unavailable")

    monkeypatch.setattr(ov_server, "VikingClient", FakeClient)

    result = await _make_search_experience_tool(
        case_lookup=_tau2_exact_case_lookup(),
        experience_recall_mode="case_ann",
    ).execute(
        None,
        situation="The user wants to cancel all upcoming reservations.",
        task_signature="tau2:airline:train:39",
    )

    assert result == "Error searching experience candidates: service unavailable"


def test_tau2_rollout_backend_factory_defaults_to_native():
    from benchmark.tau2.train.rollout_executor import (
        NativeTau2RolloutExecutor,
        make_tau2_rollout_executor,
        normalize_tau2_rollout_backend,
    )

    executor = make_tau2_rollout_executor(
        options={"keep_default_tools": False, "max_iterations": 7},
        concurrency=3,
    )

    assert normalize_tau2_rollout_backend(None) == "native"
    assert isinstance(executor, NativeTau2RolloutExecutor)
    assert executor.concurrency == 3
    assert executor.memory_enabled is False
    assert executor.max_steps == 7
    assert executor.show_progress is False


def test_tau2_native_rollout_resolves_non_empty_llms(monkeypatch):
    from benchmark.tau2.train.rollout_executor_native import (
        NativeTau2RolloutExecutor,
        _resolve_llm_runtime_config,
    )

    monkeypatch.delenv("TAU2_AGENT_LLM", raising=False)
    monkeypatch.delenv("TAU2_USER_LLM", raising=False)

    agent_llm, agent_args, user_llm, user_args = _resolve_llm_runtime_config(
        NativeTau2RolloutExecutor(
            agent_llm_args={"temperature": 0.2},
            user_llm_args={"top_p": 0.9},
        )
    )

    assert agent_llm
    assert user_llm
    assert agent_args["temperature"] == 0.2
    assert user_args["temperature"] == 0.0
    assert user_args["top_p"] == 0.9


def test_tau2_native_rollout_uses_env_llm_when_options_omit_model(monkeypatch):
    from benchmark.tau2.train.rollout_executor_native import (
        NativeTau2RolloutExecutor,
        _resolve_llm_runtime_config,
    )

    monkeypatch.setenv("TAU2_AGENT_LLM", "openai/test-agent")
    monkeypatch.setenv("TAU2_USER_LLM", "openai/test-user")

    agent_llm, _agent_args, user_llm, _user_args = _resolve_llm_runtime_config(
        NativeTau2RolloutExecutor()
    )

    assert agent_llm == "openai/test-agent"
    assert user_llm == "openai/test-user"


def test_tau2_rollout_backend_factory_selects_vikingbot(monkeypatch):
    import benchmark.tau2.train.rollout_executor as module

    created = {}

    class FakeVikingBotExecutor:
        def __init__(self, **kwargs):
            created.update(kwargs)

    monkeypatch.setattr(module, "VikingBotTau2RolloutExecutor", FakeVikingBotExecutor)

    executor = module.make_tau2_rollout_executor(
        backend="vikingbot",
        options={
            "config_path": "/tmp/ov.conf",
            "max_iterations": 9,
            "session_log_root": "/tmp/tau2-session-logs",
        },
        concurrency=2,
        rollout_language="zh",
    )

    assert isinstance(executor, FakeVikingBotExecutor)
    assert created == {
        "config_path": "/tmp/ov.conf",
        "concurrency": 2,
        "keep_default_tools": True,
        "max_iterations": 9,
        "seed": 300,
        "rollout_language": "zh",
        "loader_mode": "skill",
        "experience_recall_mode": "case_exp_rerank",
        "experience_rerank_top_n": 3,
        "system_prompt_profile": "minimal",
        "direct_experience_content": None,
        "direct_experience_name": None,
        "direct_experience_uri": None,
        "first_user_cache": True,
        "first_user_cache_dir": None,
        "session_log_root": "/tmp/tau2-session-logs",
    }

    module.make_tau2_rollout_executor(
        backend="vikingbot",
        options={
            "config_path": "/tmp/ov.conf",
            "system_prompt_profile": "minimal",
        },
        concurrency=1,
    )
    assert created["system_prompt_profile"] == "minimal"

    module.make_tau2_rollout_executor(
        backend="vikingbot",
        options={
            "config_path": "/tmp/ov.conf",
            "experience_rerank_top_n": 5,
        },
        concurrency=1,
    )
    assert created["experience_rerank_top_n"] == 5


def test_tau2_vikingbot_seed_is_stable_for_task_and_trial():
    from benchmark.tau2.train.rollout_executor_vikingbot import _stable_case_seed

    assert _stable_case_seed(300, task_no=22, trial=4) == 400_322
    assert _stable_case_seed(300, task_no=22, trial="4") == 400_322


def test_tau2_vikingbot_build_agent_uses_configured_temperature(monkeypatch, tmp_path):
    import benchmark.tau2.train.rollout_executor_vikingbot as module

    config = SimpleNamespace(
        bot_data_path=tmp_path / "bot-data",
        workspace_path=tmp_path / "workspace",
        agents=SimpleNamespace(
            model="test-model",
            temperature=0.0,
            memory_window=12,
            gen_image_model="test-image-model",
        ),
        tools=SimpleNamespace(
            web=SimpleNamespace(search=SimpleNamespace(api_key="")),
            exec=SimpleNamespace(),
        ),
    )
    captured = {}

    def fake_agent_loop(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(
        module,
        "_vikingbot_imports",
        lambda: {
            "ensure_config": lambda _path: config,
            "_init_bot_data": lambda _config: None,
            "MessageBus": object,
            "SessionManager": lambda _path: object(),
            "get_source_workspace_path": lambda: tmp_path / "source",
            "SandboxManager": lambda *_args: object(),
            "_make_provider": lambda _config: object(),
            "AgentLoop": fake_agent_loop,
        },
    )

    module._build_agent(None, max_iterations=9)

    assert captured["temperature"] == 0.0
    assert captured["max_iterations"] == 9


def test_tau2_service_rollout_backend_option_overrides_default(monkeypatch):
    import benchmark.tau2.train.service_app as service_app

    calls = []

    def fake_create_dataset_service_app(**kwargs):
        calls.append(kwargs)
        return kwargs

    class FakeExecutor:
        pass

    def fake_make_tau2_rollout_executor(**kwargs):
        calls.append({"factory": kwargs})
        return FakeExecutor()

    monkeypatch.setattr(service_app, "create_dataset_service_app", fake_create_dataset_service_app)
    monkeypatch.setattr(service_app, "make_tau2_rollout_executor", fake_make_tau2_rollout_executor)

    app = service_app.create_app(
        rollout_backend="native",
        experience_recall_mode="case_ann",
        experience_rerank_top_n=4,
        first_user_cache=False,
    )
    executor = app["make_rollout_executor"]({"rollout_backend": "vikingbot", "max_iterations": 5})

    assert isinstance(executor, FakeExecutor)
    assert calls[-1]["factory"]["backend"] == "vikingbot"
    assert calls[-1]["factory"]["options"]["max_iterations"] == 5
    assert calls[-1]["factory"]["options"]["show_progress"] is False
    assert calls[-1]["factory"]["options"]["first_user_cache"] is False
    assert calls[-1]["factory"]["options"]["experience_recall_mode"] == "case_ann"
    assert calls[-1]["factory"]["options"]["experience_rerank_top_n"] == 4

    app["make_rollout_executor"](
        {
            "rollout_backend": "native",
            "show_progress": True,
            "experience_rerank_top_n": 5,
        }
    )
    assert calls[-1]["factory"]["options"]["show_progress"] is True
    assert calls[-1]["factory"]["options"]["experience_rerank_top_n"] == 5

    default_app = service_app.create_app(rollout_backend="native")
    default_app["make_rollout_executor"]({})
    assert calls[-1]["factory"]["options"]["first_user_cache"] is True
    assert calls[-1]["factory"]["options"]["experience_recall_mode"] == "case_exp_rerank"
    assert calls[-1]["factory"]["options"]["experience_rerank_top_n"] == 3


def test_tau2_service_cli_recall_mode_default_ignores_environment(monkeypatch):
    import benchmark.tau2.train.service_app as service_app

    monkeypatch.setenv("TAU2_EXPERIENCE_RECALL_MODE", "case_ann")
    monkeypatch.setattr(sys, "argv", ["service_app.py"])

    assert service_app.parse_args().experience_recall_mode == "case_exp_rerank"
    assert service_app.parse_args().experience_rerank_top_n == 3

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "service_app.py",
            "--experience-recall-mode",
            "case_ann",
            "--experience-rerank-top-n",
            "5",
        ],
    )
    args = service_app.parse_args()
    assert args.experience_recall_mode == "case_ann"
    assert args.experience_rerank_top_n == 5


@pytest.mark.asyncio
async def test_tau2_vikingbot_rollout_runs_on_current_event_loop():
    from benchmark.tau2.train.rollout_executor_vikingbot import VikingBotTau2RolloutExecutor

    expected_loop = asyncio.get_running_loop()
    expected_thread = threading.get_ident()
    observed = {}

    class FakeVikingBotExecutor(VikingBotTau2RolloutExecutor):
        async def _execute_one_async(self, case, context):
            del context
            observed["loop"] = asyncio.get_running_loop()
            observed["thread"] = threading.get_ident()
            await asyncio.sleep(0)
            return case.name

    executor = FakeVikingBotExecutor()

    result = await executor._execute_one(
        _case(),
        ExecutionContext(policy_snapshot_id="snapshot", metadata={}),
    )

    assert result == "case-1"
    assert observed["loop"] is expected_loop
    assert observed["thread"] == expected_thread


@pytest.mark.asyncio
async def test_tau2_vikingbot_rollout_writes_run_local_session_log(tmp_path):
    from loguru import logger as loguru_logger

    import benchmark.tau2.train.rollout_executor_vikingbot as module
    from benchmark.tau2.train.rollout_executor_vikingbot import VikingBotTau2RolloutExecutor
    from openviking.session.train import Rollout

    class FakeVikingBotExecutor(VikingBotTau2RolloutExecutor):
        async def _execute_one_async(self, case, context):
            module.logger.warning("python-inside-rollout case=%s", case.name)
            loguru_logger.warning("loguru-inside-rollout case={}", case.name)
            return Rollout(
                case=case,
                messages=[],
                policy_snapshot_id=context.policy_snapshot_id,
                metadata={},
            )

    executor = FakeVikingBotExecutor(session_log_root=str(tmp_path))
    context = ExecutionContext(
        policy_snapshot_id="snapshot",
        metadata={"epoch": 2, "rollout_stage": "epoch_test_rollout"},
    )

    rollout = await executor._execute_one(_case(), context)

    expected = tmp_path / "epoch_test_rollout" / "epoch_2" / "case-1.log"
    assert rollout.metadata["vikingbot_log_path"] == str(expected.resolve())
    content = expected.read_text(encoding="utf-8")
    assert "tau2 vikingbot rollout start" in content
    assert "python-inside-rollout case=case-1" in content
    assert "loguru-inside-rollout case=case-1" in content
    assert "tau2 vikingbot rollout success" in content


@pytest.mark.asyncio
async def test_tau2_vikingbot_rollout_without_log_root_keeps_metadata_unchanged():
    from benchmark.tau2.train.rollout_executor_vikingbot import VikingBotTau2RolloutExecutor
    from openviking.session.train import Rollout

    class FakeVikingBotExecutor(VikingBotTau2RolloutExecutor):
        async def _execute_one_async(self, case, context):
            return Rollout(
                case=case,
                messages=[],
                policy_snapshot_id=context.policy_snapshot_id,
                metadata={"existing": True},
            )

    rollout = await FakeVikingBotExecutor()._execute_one(
        _case(),
        ExecutionContext(policy_snapshot_id="snapshot", metadata={}),
    )

    assert rollout.metadata == {"existing": True}


@pytest.mark.asyncio
async def test_tau2_prepare_experience_loader_skill_writes_static_required_skill(tmp_path):
    import benchmark.tau2.train.rollout_executor_vikingbot as module

    class FakeSandbox:
        def __init__(self):
            self.writes = []

        async def read_file(self, path):
            target = tmp_path / path
            if not target.is_file():
                raise FileNotFoundError(path)
            return target.read_text(encoding="utf-8")

        async def write_file(self, path, content):
            self.writes.append((path, content))
            target = tmp_path / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")

    fake_sandbox = FakeSandbox()

    class FakeSandboxManager:
        def get_workspace_path(self, session_key):
            return tmp_path

        def to_workspace_id(self, session_key):
            return "workspace"

        async def get_sandbox(self, session_key):
            return fake_sandbox

    class FakeAgent:
        sandbox_manager = FakeSandboxManager()
        context = SimpleNamespace(workspace=tmp_path)

    context_builder = await module._prepare_experience_loader_skill(
        agent=FakeAgent(),
        session_key=SimpleNamespace(),
    )

    skill_path = tmp_path / "skills" / "experience_loader" / "SKILL.md"
    content = skill_path.read_text(encoding="utf-8")
    assert context_builder.workspace == tmp_path
    assert "name: experience_loader" in content
    assert "search_experience" in content
    assert "read_experience" in content
    assert "search_experience(situation, task_signature=None, limit=2)" in content
    assert "search_experience(query" not in content
    assert "## Runtime Case context" not in content
    assert "tau2:airline:train:39" not in content
    assert "keyword list" in content
    assert "current conversation" in content
    assert "Situation snippets" in content or "`situation` as a filter only" in content
    assert "Read by default" in content
    assert "call `read_experience` unless" in content
    assert "already loaded inline" in content
    assert "do not call `read_experience` again" in content
    assert "later boundary in the same task" in content
    assert "Re-search on new subtasks" in content
    assert "case_name" in content
    assert "case URI" not in content
    assert "RETURN_COMPLETED" not in content
    assert "RETURN_BLOCKED" not in content
    assert "RETURN_NOT_APPLICABLE" not in content
    assert fake_sandbox.writes
    assert fake_sandbox.writes[0][0] == "skills/experience_loader/SKILL.md"
    assert context_builder.latest_experience_loader_skill_content == content


@pytest.mark.asyncio
async def test_tau2_prepare_experience_loader_skill_skips_identical_content(tmp_path):
    import benchmark.tau2.train.rollout_executor_vikingbot as module

    skill_path = tmp_path / module.EXPERIENCE_LOADER_SKILL_PATH
    skill_path.parent.mkdir(parents=True)
    skill_path.write_text(
        module._read_experience_loader_template_file("SKILL.md"),
        encoding="utf-8",
    )

    class FakeSandbox:
        def __init__(self):
            self.writes = []

        async def read_file(self, path):
            return (tmp_path / path).read_text(encoding="utf-8")

        async def write_file(self, path, content):
            self.writes.append((path, content))
            (tmp_path / path).write_text(content, encoding="utf-8")

    fake_sandbox = FakeSandbox()

    class FakeSandboxManager:
        def get_workspace_path(self, session_key):
            return tmp_path

        async def get_sandbox(self, session_key):
            return fake_sandbox

    class FakeAgent:
        sandbox_manager = FakeSandboxManager()
        context = SimpleNamespace(workspace=tmp_path)

    await module._prepare_experience_loader_skill(
        agent=FakeAgent(),
        session_key=SimpleNamespace(),
    )

    assert fake_sandbox.writes == []


@pytest.mark.asyncio
async def test_tau2_prepare_experience_loader_skill_replaces_changed_content(tmp_path):
    import benchmark.tau2.train.rollout_executor_vikingbot as module

    skill_path = tmp_path / module.EXPERIENCE_LOADER_SKILL_PATH
    skill_path.parent.mkdir(parents=True)
    skill_path.write_text("old skill", encoding="utf-8")

    class FakeSandbox:
        def __init__(self):
            self.writes = []

        async def read_file(self, path):
            return (tmp_path / path).read_text(encoding="utf-8")

        async def write_file(self, path, content):
            self.writes.append((path, content))
            (tmp_path / path).write_text(content, encoding="utf-8")

    fake_sandbox = FakeSandbox()

    class FakeSandboxManager:
        def get_workspace_path(self, session_key):
            return tmp_path

        async def get_sandbox(self, session_key):
            return fake_sandbox

    class FakeAgent:
        sandbox_manager = FakeSandboxManager()
        context = SimpleNamespace(workspace=tmp_path)

    await module._prepare_experience_loader_skill(
        agent=FakeAgent(),
        session_key=SimpleNamespace(),
    )

    expected = module._read_experience_loader_template_file("SKILL.md")
    assert fake_sandbox.writes == [(module.EXPERIENCE_LOADER_SKILL_PATH, expected)]
    assert skill_path.read_text(encoding="utf-8") == expected


@pytest.mark.asyncio
async def test_tau2_prepare_experience_loader_skill_serializes_worker_thread_installs(tmp_path):
    import benchmark.tau2.train.rollout_executor_vikingbot as module

    class FakeSandbox:
        def __init__(self):
            self.writes = []

        async def read_file(self, path):
            target = tmp_path / path
            if not target.is_file():
                await asyncio.sleep(0.05)
                if not target.is_file():
                    raise FileNotFoundError(path)
            return target.read_text(encoding="utf-8")

        async def write_file(self, path, content):
            self.writes.append((path, content))
            target = tmp_path / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")

    fake_sandbox = FakeSandbox()

    class FakeSandboxManager:
        def get_workspace_path(self, session_key):
            return tmp_path

        async def get_sandbox(self, session_key):
            return fake_sandbox

    class FakeAgent:
        sandbox_manager = FakeSandboxManager()
        context = SimpleNamespace(workspace=tmp_path)

    def prepare_in_worker_thread():
        asyncio.run(
            module._prepare_experience_loader_skill(
                agent=FakeAgent(),
                session_key=SimpleNamespace(),
            )
        )

    await asyncio.gather(*(asyncio.to_thread(prepare_in_worker_thread) for _ in range(8)))

    assert len(fake_sandbox.writes) == 1


@pytest.mark.asyncio
async def test_tau2_prepare_experience_loader_skill_rejects_unverified_content(tmp_path):
    import benchmark.tau2.train.rollout_executor_vikingbot as module

    class FakeSandbox:
        async def read_file(self, path):
            return "corrupted skill"

        async def write_file(self, path, content):
            return None

    class FakeSandboxManager:
        def get_workspace_path(self, session_key):
            return tmp_path

        async def get_sandbox(self, session_key):
            return FakeSandbox()

    class FakeAgent:
        sandbox_manager = FakeSandboxManager()
        context = SimpleNamespace(workspace=tmp_path)

    with pytest.raises(RuntimeError, match="experience_loader skill verification failed"):
        await module._prepare_experience_loader_skill(
            agent=FakeAgent(),
            session_key=SimpleNamespace(),
        )


@pytest.mark.asyncio
async def test_tau2_experience_loader_skill_is_required_with_relative_read_path(tmp_path):
    from vikingbot.config.schema import SessionKey

    import benchmark.tau2.train.rollout_executor_vikingbot as module

    module._write_experience_loader_files(
        workspace_path=tmp_path,
        skill_content="# experience_loader\n\nUse search_experience then read_experience.",
    )

    from vikingbot.agent.context import ContextBuilder

    context_builder = ContextBuilder(tmp_path, eval=True)
    system_prompt = await context_builder.build_system_prompt(
        SessionKey(type="cli", channel_id="tau2", chat_id="case"),
        ov_tools_enable=False,
    )

    assert "Required skill: before taking any task action" in system_prompt
    assert "`skills/experience_loader/SKILL.md`" in system_prompt
    assert "<location>skills/experience_loader/SKILL.md</location>" in system_prompt
    assert f"<location>{tmp_path}" not in system_prompt


@pytest.mark.asyncio
async def test_tau2_vikingbot_blocking_setup_and_reward_are_offloaded(monkeypatch, tmp_path):
    import benchmark.tau2.train.rollout_executor_vikingbot as module
    from benchmark.tau2.train.rollout_executor_vikingbot import VikingBotTau2RolloutExecutor

    event_loop_thread = threading.get_ident()
    calls = []

    class FakeEnv:
        def _get_reward(self):
            calls.append(("reward", threading.get_ident()))
            return 1.0, {"ok": True}

    class FakeTau2BenchToolProvider:
        def __init__(self, domain, task_id, data_root=None):
            self.domain = domain
            self.task_id = task_id
            self.data_root = data_root
            self.env = FakeEnv()
            self.policy = "policy"
            self.user_query = "user query"

        def reset(self, *, seed=None, fixed_first_user_message=None):
            calls.append(("reset", threading.get_ident()))
            calls.append(("reset_seed", seed))
            calls.append(("fixed_first_user_message", fixed_first_user_message))

        def list_openai_tools(self):
            return []

    class FakeAgent:
        def __init__(self):
            calls.append(("build_agent", threading.get_ident()))

    async def fake_run_agent(**kwargs):
        calls.append(("run_agent", threading.get_ident()))
        calls.append(("case_lookup", kwargs.get("case_lookup")))
        return (
            "final",
            None,
            [],
            {},
            1,
            None,
            None,
            None,
            [{"role": "user", "content": "user query"}, {"role": "assistant", "content": "final"}],
        )

    monkeypatch.setattr(module, "_tool_provider_cls", lambda: FakeTau2BenchToolProvider)
    monkeypatch.setattr(module, "_build_agent", lambda *args, **kwargs: FakeAgent())
    monkeypatch.setattr(module, "_configure_tools", lambda *args, **kwargs: None)
    monkeypatch.setattr(module, "_run_agent", fake_run_agent)

    case = Case(
        name="tau2_case",
        task_signature="tau2:airline:train:0",
        input={
            "domain": "airline",
            "split": "train",
            "task_id": "0",
            "task_no": 0,
            "data_split": "airline_train",
        },
        rubric=Rubric(name="rubric", description="", criteria=[]),
    )
    executor = VikingBotTau2RolloutExecutor(first_user_cache_dir=str(tmp_path))

    rollout = await executor._execute_one(
        case,
        ExecutionContext(policy_snapshot_id="snapshot", metadata={}),
    )

    assert rollout.metadata["reward"] == 1.0
    assert rollout.metadata["evaluation_result"] == {"ok": True}
    assert rollout.metadata["seed"] == 300
    assert rollout.metadata["first_user_cache_enabled"] is True
    assert rollout.metadata["first_user_cache_hit"] is False
    assert Path(rollout.metadata["first_user_cache_path"]).is_file()
    assert "evaluation_result" not in rollout.evaluation.metadata
    call_values = dict(calls)
    assert call_values["case_lookup"] == {
        "benchmark": "tau2",
        "strict": True,
        "case_names": ["tau2_case", "tau2_airline_train_0"],
        "domain": "airline",
        "split": "train",
        "data_split": "airline_train",
        "task_no": 0,
        "task_id": "0",
        "case_name": "tau2_case",
        "task_signature": "tau2:airline:train:0",
        "original_case_name": None,
        "expected_fields": {
            "input.domain": "airline",
            "input.split": "train",
            "input.data_split": "airline_train",
            "input.task_no": 0,
            "input.task_id": "0",
        },
    }
    call_threads = call_values
    assert call_threads["reset"] != event_loop_thread
    assert call_threads["reset_seed"] == 300
    assert call_threads["build_agent"] != event_loop_thread
    assert call_threads["reward"] != event_loop_thread
    assert call_threads["run_agent"] == event_loop_thread


@pytest.mark.asyncio
async def test_tau2_run_agent_force_loads_experience_loader_skill_before_task_actions(monkeypatch):
    from vikingbot.providers.base import LLMResponse, ToolCallRequest

    import benchmark.tau2.train.rollout_executor_vikingbot as module

    observed = {}
    real_imports = module._vikingbot_imports()

    class FakeSandbox:
        content = ""

        async def read_file(self, path):
            observed.setdefault("sandbox_reads", []).append(path)
            return self.content

        async def write_file(self, path, content):
            observed.setdefault("sandbox_writes", []).append((path, content))
            type(self).content = content

    class FakeSandboxManager:
        def get_workspace_path(self, session_key):
            return Path("/tmp/fake-workspace")

        def to_workspace_id(self, session_key):
            return "workspace"

        async def get_sandbox(self, session_key):
            return FakeSandbox()

    class FakeContextBuilder:
        def __init__(self, workspace, *, sandbox_manager=None, eval=False, **kwargs):
            self.workspace = workspace
            self.sandbox_manager = sandbox_manager
            self.latest_experience_loader_skill_content = ""

        async def build_messages(self, **kwargs):
            return [
                {"role": "system", "content": "ctx system"},
                {"role": "user", "content": kwargs["current_message"]},
            ]

        def add_assistant_message(self, messages, content, tool_calls=None, reasoning_content=None):
            msg = {"role": "assistant", "content": content or "[tool call]"}
            if tool_calls:
                msg["tool_calls"] = tool_calls
            if reasoning_content:
                msg["reasoning_content"] = reasoning_content
            messages.append(msg)
            return messages

        def add_tool_result(self, messages, tool_call_id, tool_name, result):
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "name": tool_name,
                    "content": result,
                }
            )
            return messages

    class FakeProvider:
        async def chat(self, messages, tools=None, **kwargs):
            observed["llm_messages"] = list(messages)
            return LLMResponse(
                content=None,
                tool_calls=[ToolCallRequest("call-1", "done", {}, 0)],
            )

        async def chat_stream(self, **kwargs):
            from vikingbot.providers.base import LLMStreamEvent

            yield LLMStreamEvent(type="response", response=await self.chat(**kwargs))

        def get_default_model(self):
            return "fake"

    class FakeAgent:
        def __init__(self):
            from vikingbot.agent.tools.filesystem import ReadFileTool
            from vikingbot.agent.tools.registry import ToolRegistry

            self.sandbox_manager = FakeSandboxManager()
            self.context = FakeContextBuilder(Path("/tmp/fake-workspace"))
            self.tools = ToolRegistry()
            self.tools.register(ReadFileTool())
            self.tools.register(_DoneTool())
            self.provider = FakeProvider()
            self.model = "fake"
            self.temperature = None
            self.max_iterations = 1

        _chat_with_stream_events = real_imports["AgentLoop"]._chat_with_stream_events
        _run_agent_loop = real_imports["AgentLoop"]._run_agent_loop

    class _DoneTool:
        @property
        def name(self):
            return "done"

        @property
        def description(self):
            return "done"

        @property
        def parameters(self):
            return {"type": "object", "properties": {}}

        def to_schema(self):
            return {
                "type": "function",
                "function": {
                    "name": self.name,
                    "description": self.description,
                    "parameters": self.parameters,
                },
            }

        def validate_params(self, params):
            return []

        async def execute(self, tool_context, **kwargs):
            return ""

    monkeypatch.setattr(
        module,
        "_vikingbot_imports",
        lambda: {**real_imports, "ContextBuilder": FakeContextBuilder},
    )

    result = await module._run_agent(
        agent=FakeAgent(),
        system_prompt="tau2 policy",
        user_prompt="user query",
        session_key=SimpleNamespace(safe_name=lambda: "session"),
        sender_id="tau2_user",
        keep_default_tools=True,
        loader_mode="skill",
        case_lookup={
            "benchmark": "tau2",
            "strict": True,
            "case_name": "case",
            "task_signature": "tau2:airline:train:39",
        },
    )

    tools_used = result[2]
    messages = observed["llm_messages"]
    read_call_index = next(
        i
        for i, msg in enumerate(messages)
        if msg.get("role") == "assistant" and "read_file" in str(msg.get("tool_calls"))
    )
    tool_result_index = next(
        i
        for i, msg in enumerate(messages)
        if msg.get("role") == "tool" and msg.get("name") == "read_file"
    )

    assert observed["sandbox_writes"][0][0] == "skills/experience_loader/SKILL.md"
    assert observed["sandbox_reads"] == [
        "skills/experience_loader/SKILL.md",  # install-time comparison
        "skills/experience_loader/SKILL.md",  # post-write verification
        "skills/experience_loader/SKILL.md",  # required agent-visible read
    ]
    assert read_call_index < tool_result_index
    assert "search_experience" in messages[tool_result_index]["content"]
    assert "read_experience" in messages[tool_result_index]["content"]
    runtime_prompt = next(
        message["content"]
        for message in messages
        if message.get("role") == "system" and "tau2 policy" in message.get("content", "")
    )
    assert "## Runtime Case context" in runtime_prompt
    assert "`task_signature`: `tau2:airline:train:39`" in runtime_prompt
    assert "tau2:airline:train:39" not in messages[tool_result_index]["content"]
    assert "tau2:airline:train:39" not in result[7]
    assert tools_used[0]["tool_name"] == "read_file"
    assert tools_used[0]["required_skill"] == "experience_loader"


@pytest.mark.asyncio
async def test_tau2_run_agent_constraint_mode_does_not_force_load_experience_loader_skill(
    monkeypatch,
):
    from vikingbot.providers.base import LLMResponse, ToolCallRequest

    import benchmark.tau2.train.rollout_executor_vikingbot as module

    observed = {}
    real_imports = module._vikingbot_imports()

    class FakeSandboxManager:
        def get_workspace_path(self, session_key):
            return Path("/tmp/fake-workspace")

        def to_workspace_id(self, session_key):
            return "workspace"

        async def get_sandbox(self, session_key):
            raise AssertionError("constraint mode should not request sandbox for skill install")

    class FakeContextBuilder:
        def __init__(self, workspace, *, sandbox_manager=None, eval=False, **kwargs):
            self.workspace = workspace
            self.sandbox_manager = sandbox_manager
            self.latest_experience_loader_skill_content = ""

        async def build_messages(self, **kwargs):
            return [
                {"role": "system", "content": "ctx system"},
                {"role": "user", "content": kwargs["current_message"]},
            ]

        def add_assistant_message(self, messages, content, tool_calls=None, reasoning_content=None):
            msg = {"role": "assistant", "content": content or "[tool call]"}
            if tool_calls:
                msg["tool_calls"] = tool_calls
            messages.append(msg)
            return messages

        def add_tool_result(self, messages, tool_call_id, tool_name, result):
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "name": tool_name,
                    "content": result,
                }
            )
            return messages

    class FakeProvider:
        async def chat(self, messages, tools=None, **kwargs):
            observed["llm_messages"] = list(messages)
            return LLMResponse(
                content=None,
                tool_calls=[ToolCallRequest("call-1", "done", {}, 0)],
            )

        async def chat_stream(self, **kwargs):
            from vikingbot.providers.base import LLMStreamEvent

            yield LLMStreamEvent(type="response", response=await self.chat(**kwargs))

        def get_default_model(self):
            return "fake"

    class FakeAgent:
        def __init__(self):
            from vikingbot.agent.tools.registry import ToolRegistry

            self.sandbox_manager = FakeSandboxManager()
            self.context = FakeContextBuilder(Path("/tmp/fake-workspace"))
            self.tools = ToolRegistry()
            self.tools.register(_DoneTool())
            self.provider = FakeProvider()
            self.model = "fake"
            self.temperature = None
            self.max_iterations = 1

        _chat_with_stream_events = real_imports["AgentLoop"]._chat_with_stream_events
        _run_agent_loop = real_imports["AgentLoop"]._run_agent_loop

    class _DoneTool:
        @property
        def name(self):
            return "done"

        @property
        def description(self):
            return "done"

        @property
        def parameters(self):
            return {"type": "object", "properties": {}}

        def to_schema(self):
            return {
                "type": "function",
                "function": {
                    "name": self.name,
                    "description": self.description,
                    "parameters": self.parameters,
                },
            }

        def validate_params(self, params):
            return []

        async def execute(self, tool_context, **kwargs):
            return ""

    monkeypatch.setattr(
        module,
        "_vikingbot_imports",
        lambda: {**real_imports, "ContextBuilder": FakeContextBuilder},
    )

    result = await module._run_agent(
        agent=FakeAgent(),
        system_prompt="tau2 policy",
        user_prompt="user query",
        session_key=SimpleNamespace(safe_name=lambda: "session"),
        sender_id="tau2_user",
        keep_default_tools=True,
        loader_mode="constraint",
        case_lookup={"benchmark": "tau2", "strict": True, "case_name": "case"},
    )

    tools_used = result[2]
    assert not any(tool.get("required_skill") == "experience_loader" for tool in tools_used)
    assert all("experience_loader" not in str(message) for message in observed["llm_messages"])


@pytest.mark.asyncio
async def test_tau2_run_agent_direct_experience_injects_reminder_without_skill():
    import benchmark.tau2.train.rollout_executor_vikingbot as module

    observed = {}

    class FakeContextBuilder:
        async def build_messages(self, **kwargs):
            return [
                {"role": "system", "content": "ctx system"},
                {"role": "user", "content": kwargs["current_message"]},
            ]

    class FakeLoopResult:
        def __init__(self, messages):
            self.messages = list(messages)

        def __iter__(self):
            yield "final"
            yield None
            yield []
            yield {}
            yield 1

    class FakeAgent:
        context = FakeContextBuilder()
        bus = None

        async def _run_agent_loop(self, **kwargs):
            observed["messages"] = list(kwargs["messages"])
            return FakeLoopResult(kwargs["messages"])

    result = await module._run_agent(
        agent=FakeAgent(),
        system_prompt="tau2 policy",
        user_prompt="user query",
        session_key=SimpleNamespace(safe_name=lambda: "session"),
        sender_id="tau2_user",
        keep_default_tools=True,
        loader_mode="direct_experience",
        direct_experience_content="When the user asks for an aggregate, freeze the object set.",
        direct_experience_name="freeze_aggregate_request_scope",
        case_lookup={"benchmark": "tau2", "strict": True, "case_name": "case"},
    )

    messages = observed["messages"]
    reminder = next(
        msg["content"]
        for msg in messages
        if msg.get("role") == "user" and "[Experience Reminder]" in msg.get("content", "")
    )
    assert messages[0] == {"role": "system", "content": "ctx system"}
    assert messages[1] == {"role": "system", "content": "tau2 policy"}
    assert messages[2]["content"] == reminder
    assert "### freeze_aggregate_request_scope" in reminder
    assert "direct://experience/freeze_aggregate_request_scope" in reminder
    assert "freeze the object set" in reminder
    assert result[5] is not None
    assert "## Experience Memories" in result[5]
    assert result[6] == reminder
    assert result[7] is None


def test_tau2_rollout_messages_preserve_runtime_user_messages():
    from benchmark.tau2.train.rollout_executor import _build_rollout_messages
    from openviking.message import TextPart, ToolPart

    rollout_messages = _build_rollout_messages(
        system_prompt="policy",
        user_prompt="user request",
        tools_used=[],
        final_content=None,
        evaluation_result=None,
        reward=0.0,
        runtime_messages=[
            {"role": "system", "content": "policy"},
            {"role": "user", "content": "user request"},
            {"role": "user", "content": "## Situation\n- Check cancellation eligibility."},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {
                            "name": "get_reservation_details",
                            "arguments": '{"reservation_id": "XEHM4B"}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call-1",
                "name": "get_reservation_details",
                "content": '{"reservation_id": "XEHM4B"}',
            },
        ],
    )

    user_texts = [
        part.text
        for message in rollout_messages
        if message.role == "user"
        for part in message.parts
        if isinstance(part, TextPart)
    ]
    assert "## Situation\n- Check cancellation eligibility." in user_texts
    tool_parts = [
        part for message in rollout_messages for part in message.parts if isinstance(part, ToolPart)
    ]
    assert tool_parts[0].tool_name == "get_reservation_details"
    assert tool_parts[0].tool_input == {"reservation_id": "XEHM4B"}
