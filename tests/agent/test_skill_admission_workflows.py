"""Workflow-style scenario tests for strict skill admission."""

from types import SimpleNamespace
from unittest.mock import patch

from agent.skill_admission import (
    DECISION_PROMOTE_NEW,
    DECISION_REJECT,
    ORIGIN_AUTOMATIC_REVIEW,
    maybe_generate_automatic_candidate,
    run_admission_review,
)

_MAIN_RUNTIME = {
    "provider": "openrouter",
    "model": "openai/gpt-5",
    "base_url": "https://openrouter.ai/api/v1",
    "api_key": "test-key",
    "api_mode": "chat_completions",
}


def _response(content: str):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
    )


def _candidate_from_generated(generated: dict) -> dict:
    return {
        "origin": ORIGIN_AUTOMATIC_REVIEW,
        "proposed_name": generated["name"],
        "category": generated.get("category", ""),
        "scope": generated["scope"],
        "why_created": generated["why_created"],
        "reusability_rationale": generated.get("reusability_rationale", ""),
        "known_limits": generated.get("known_limits", ""),
        "verification_summary": generated["verification_summary"],
        "tool_calls": generated.get("tool_calls", []),
        "commands_run": generated.get("commands_run", []),
        "tests_run": generated.get("tests_run", []),
        "skill_content": generated["content"],
    }


def _pytest_fix_messages():
    return [
        {"role": "user", "content": "fix the failing plugin tool tests and keep the new strict skill flow working"},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "function": {
                        "name": "terminal",
                        "arguments": '{"command":"python -m pytest tests/test_model_tools.py -q -k plugin_tool_dispatch"}',
                    }
                }
            ],
        },
        {
            "role": "tool",
            "content": "1 failed, 19 deselected\\nAssertionError: plugin tool should be blocked under strict mode",
        },
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "function": {
                        "name": "terminal",
                        "arguments": '{"command":"rg -n \\"plugin tool\\" model_tools.py tests/test_model_tools.py"}',
                    }
                }
            ],
        },
        {"role": "tool", "content": "model_tools.py:211\\ntests/test_model_tools.py:83"},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "function": {
                        "name": "terminal",
                        "arguments": '{"command":"python -m pytest tests/test_model_tools.py -q -k plugin_tool_dispatch"}',
                    }
                }
            ],
        },
        {"role": "tool", "content": "1 passed, 19 deselected"},
        {
            "role": "assistant",
            "content": "I fixed the plugin tool gating and re-ran the focused pytest target successfully.",
        },
    ]


def _trivial_branch_check_messages():
    return [
        {"role": "user", "content": "what branch am I on?"},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "function": {
                        "name": "terminal",
                        "arguments": '{"command":"git branch --show-current"}',
                    }
                }
            ],
        },
        {"role": "tool", "content": "codex/skill-admission-gate"},
        {"role": "assistant", "content": "You are on codex/skill-admission-gate."},
    ]


def _cache_workaround_messages():
    return [
        {"role": "user", "content": "make the flaky pytest file pass locally"},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "function": {
                        "name": "terminal",
                        "arguments": '{"command":"python -m pytest tests/test_run_agent.py -q -k strict_skill"}',
                    }
                }
            ],
        },
        {"role": "tool", "content": "1 failed, 210 deselected"},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "function": {
                        "name": "terminal",
                        "arguments": '{"command":"rm -rf .pytest_cache && python -m pytest tests/test_run_agent.py -q -k strict_skill"}',
                    }
                }
            ],
        },
        {"role": "tool", "content": "1 passed, 210 deselected"},
        {
            "role": "assistant",
            "content": "Clearing the local pytest cache made the flaky test pass again.",
        },
    ]


def test_automatic_workflow_strong_reusable_fix_reaches_promote_review():
    proposal_response = _response(
        """{
          "save": true,
          "name": "pytest-focused-regression-rerun",
          "description": "Re-run a focused pytest regression while iterating on a fix.",
          "category": "",
          "scope": "repo-specific debugging for focused pytest regressions",
          "why_created": "Captures the repeatable workflow for isolating and re-running a failing pytest target while editing strict-mode logic.",
          "reusability_rationale": "This same focused rerun workflow is useful whenever a strict-mode regression is localized to a small pytest target.",
          "known_limits": "Applies to Python test debugging in this repo.",
          "verification_summary": "Validated by re-running the targeted pytest command until it passed."
        }"""
    )
    draft_response = _response(
        """# Workflow

1. Run the focused pytest target first.
2. Inspect the failing code path.
3. Re-run the same focused pytest target after the fix.
"""
    )
    review_response = _response(
        """{
          "scores": {
            "reusability": 2,
            "verification": 2,
            "non_triviality": 2,
            "scope_quality": 1,
            "actionability": 2
          },
          "decision_reason": "Strong reusable debugging workflow with explicit validation.",
          "summary_for_user": "Save this focused pytest regression workflow for future strict-mode debugging.",
          "target_skill": ""
        }"""
    )

    with patch("agent.skill_admission.call_llm", side_effect=[proposal_response, draft_response]):
        generated = maybe_generate_automatic_candidate(
            _pytest_fix_messages(),
            main_runtime=_MAIN_RUNTIME,
        )

    assert generated["created"] is True
    assert generated["tests_run"] == [
        "python -m pytest tests/test_model_tools.py -q -k plugin_tool_dispatch",
        "python -m pytest tests/test_model_tools.py -q -k plugin_tool_dispatch",
    ]

    with patch("agent.skill_admission.call_llm", return_value=review_response):
        verdict = run_admission_review(
            _candidate_from_generated(generated),
            main_runtime=_MAIN_RUNTIME,
        )

    assert verdict["decision"] == DECISION_PROMOTE_NEW
    assert verdict["total_score"] == 9
    assert verdict["summary_for_user"]


def test_automatic_workflow_trivial_lookup_stays_out_of_skill_pipeline():
    proposal_response = _response('{"save": false}')

    with patch("agent.skill_admission.call_llm", return_value=proposal_response):
        generated = maybe_generate_automatic_candidate(
            _trivial_branch_check_messages(),
            main_runtime=_MAIN_RUNTIME,
        )

    assert generated["success"] is True
    assert generated["created"] is False
    assert generated["reason"] == "reviewer opted out"


def test_automatic_workflow_local_workaround_can_be_generated_but_rejected():
    proposal_response = _response(
        """{
          "save": true,
          "name": "clear-pytest-cache-for-flaky-local-reruns",
          "description": "Clear .pytest_cache before rerunning flaky local tests.",
          "category": "",
          "scope": "local repo-specific flaky pytest reruns",
          "why_created": "Documents a local workaround that made the flaky test pass.",
          "reusability_rationale": "Might help when local pytest cache corruption causes odd failures.",
          "known_limits": "This is a local workaround and may not address the real root cause.",
          "verification_summary": "Validated by deleting .pytest_cache and rerunning the focused test once."
        }"""
    )
    draft_response = _response(
        """# Workflow

1. Remove `.pytest_cache`.
2. Re-run the focused test.
"""
    )
    review_response = _response(
        """{
          "scores": {
            "reusability": 1,
            "verification": 1,
            "non_triviality": 1,
            "scope_quality": 1,
            "actionability": 1
          },
          "decision_reason": "Potentially useful, but narrow and not strong enough for automatic promotion.",
          "summary_for_user": "This looks more like a local workaround than a broadly reusable skill.",
          "target_skill": ""
        }"""
    )

    with patch("agent.skill_admission.call_llm", side_effect=[proposal_response, draft_response]):
        generated = maybe_generate_automatic_candidate(
            _cache_workaround_messages(),
            main_runtime=_MAIN_RUNTIME,
        )

    assert generated["created"] is True
    assert generated["commands_run"][1].startswith("rm -rf .pytest_cache")

    with patch("agent.skill_admission.call_llm", return_value=review_response):
        verdict = run_admission_review(
            _candidate_from_generated(generated),
            main_runtime=_MAIN_RUNTIME,
        )

    assert verdict["decision"] == DECISION_REJECT
    assert verdict["total_score"] == 5
    assert verdict["threshold_used"] == 7
