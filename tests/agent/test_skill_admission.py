"""Tests for strict skill admission helpers."""

from types import SimpleNamespace
from unittest.mock import patch

from agent.skill_admission import (
    DECISION_PROMOTE_NEW,
    DECISION_REJECT,
    ORIGIN_AUTOMATIC_REVIEW,
    RUBRIC_FIELDS,
    maybe_generate_automatic_candidate,
    run_admission_review,
    run_admission_rule_filter,
)

_MAIN_RUNTIME = {
    "provider": "openrouter",
    "model": "openai/gpt-5",
    "base_url": "https://openrouter.ai/api/v1",
    "api_key": "test-key",
    "api_mode": "chat_completions",
}


def _candidate(origin: str = ORIGIN_AUTOMATIC_REVIEW, *, evidence: bool = True):
    return {
        "origin": origin,
        "proposed_name": "test-skill",
        "category": "",
        "scope": "repo-specific",
        "why_created": "This workflow fixed a tricky test failure.",
        "reusability_rationale": "Applies to similar failing pytest runs.",
        "known_limits": "",
        "verification_summary": "Validated with a passing targeted pytest run.",
        "tool_calls": ["terminal"] if evidence else [],
        "commands_run": ["pytest tests/test_skill_admission.py -q"] if evidence else [],
        "tests_run": ["pytest tests/test_skill_admission.py -q"] if evidence else [],
        "skill_content": "---\nname: test-skill\ndescription: desc\n---\n\n# Body\n",
    }


def _response(content: str):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
    )


def _reasoning_response(content: str = "", *, reasoning_content: str | None = None):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=content,
                    reasoning_content=reasoning_content,
                )
            )
        ]
    )


def test_rule_filter_rejects_non_automatic_origin():
    verdict = run_admission_rule_filter(_candidate("user_requested"))
    assert verdict is not None
    assert verdict["decision"] == DECISION_REJECT
    assert "only handles automatic" in verdict["decision_reason"]


def test_rule_filter_rejects_automatic_review_without_evidence():
    verdict = run_admission_rule_filter(_candidate(evidence=False))
    assert verdict is not None
    assert verdict["decision"] == DECISION_REJECT
    assert "concrete tool or command evidence" in verdict["decision_reason"]
    assert verdict["threshold_used"] == 7


def test_valid_scores_promote_automatic_at_threshold():
    candidate = _candidate()
    fake_response = _response(
        """{
          "scores": {
            "reusability": 2,
            "verification": 2,
            "non_triviality": 1,
            "scope_quality": 1,
            "actionability": 1
          },
          "decision_reason": "Structured and sufficiently reusable.",
          "summary_for_user": "Save this bounded pytest workflow.",
          "target_skill": ""
        }"""
    )
    with patch("agent.skill_admission.call_llm", return_value=fake_response):
        verdict = run_admission_review(
            candidate,
            main_runtime=_MAIN_RUNTIME,
        )

    assert verdict["decision"] == DECISION_PROMOTE_NEW
    assert verdict["total_score"] == 7
    assert verdict["threshold_used"] == 7
    assert verdict["scores"]["actionability"] == 1


def test_invalid_scores_fall_back_to_reject():
    candidate = _candidate()
    fake_response = _response(
        """{
          "scores": {
            "reusability": 3,
            "verification": 1,
            "non_triviality": 1,
            "scope_quality": 1,
            "actionability": 1
          },
          "decision_reason": "bad scores",
          "summary_for_user": "",
          "target_skill": ""
        }"""
    )
    with patch("agent.skill_admission.call_llm", return_value=fake_response):
        verdict = run_admission_review(
            candidate,
            main_runtime=_MAIN_RUNTIME,
        )

    assert verdict["decision"] == DECISION_REJECT
    assert "valid rubric scores" in verdict["decision_reason"]
    assert set(verdict["scores"]) == set(RUBRIC_FIELDS)


def test_stringified_scores_are_accepted():
    candidate = _candidate()
    fake_response = _response(
        """{
          "scores": {
            "reusability": "2",
            "verification": "2",
            "non_triviality": "1",
            "scope_quality": "1",
            "actionability": "1"
          },
          "decision_reason": "Structured and sufficiently reusable.",
          "summary_for_user": "Save this bounded pytest workflow.",
          "target_skill": ""
        }"""
    )
    with patch("agent.skill_admission.call_llm", return_value=fake_response):
        verdict = run_admission_review(
            candidate,
            main_runtime=_MAIN_RUNTIME,
        )

    assert verdict["decision"] == DECISION_PROMOTE_NEW
    assert verdict["scores"]["reusability"] == 2
    assert verdict["scores"]["actionability"] == 1


def test_reviewer_json_in_reasoning_content_is_accepted():
    candidate = _candidate()
    fake_response = _reasoning_response(
        reasoning_content="""{
          "scores": {
            "reusability": 2,
            "verification": 2,
            "non_triviality": 1,
            "scope_quality": 1,
            "actionability": 1
          },
          "decision_reason": "Structured and sufficiently reusable.",
          "summary_for_user": "Save this bounded pytest workflow.",
          "target_skill": ""
        }"""
    )
    with patch("agent.skill_admission.call_llm", return_value=fake_response):
        verdict = run_admission_review(
            candidate,
            main_runtime=_MAIN_RUNTIME,
        )

    assert verdict["decision"] == DECISION_PROMOTE_NEW
    assert verdict["total_score"] == 7


def test_reviewer_reasoning_content_with_prose_and_json_block_is_repaired():
    candidate = _candidate()
    fake_response = _reasoning_response(
        reasoning_content="""I will score this candidate conservatively.

```json
{
  "scores": {
    "reusability": 2,
    "verification": 2,
    "non_triviality": 1,
    "scope_quality": 1,
    "actionability": 1,
  },
  "decision_reason": "Structured and sufficiently reusable.",
  "summary_for_user": "Save this bounded pytest workflow.",
  "target_skill": "",
}
```"""
    )
    with patch("agent.skill_admission.call_llm", return_value=fake_response):
        verdict = run_admission_review(
            candidate,
            main_runtime=_MAIN_RUNTIME,
        )

    assert verdict["decision"] == DECISION_PROMOTE_NEW
    assert verdict["total_score"] == 7


def test_reviewer_json_with_trailing_commas_is_repaired():
    candidate = _candidate()
    fake_response = _response(
        """```json
        {
          "scores": {
            "reusability": 2,
            "verification": 2,
            "non_triviality": 1,
            "scope_quality": 1,
            "actionability": 1,
          },
          "decision_reason": "Structured and sufficiently reusable.",
          "summary_for_user": "Save this bounded pytest workflow.",
          "target_skill": "",
        }
        ```"""
    )
    with patch("agent.skill_admission.call_llm", return_value=fake_response):
        verdict = run_admission_review(
            candidate,
            main_runtime=_MAIN_RUNTIME,
        )

    assert verdict["decision"] == DECISION_PROMOTE_NEW
    assert verdict["total_score"] == 7


def test_no_single_dimension_gate_if_total_meets_threshold():
    candidate = _candidate()
    candidate["tool_calls"] = ["terminal"]
    fake_response = _response(
        """{
          "scores": {
            "reusability": 2,
            "verification": 0,
            "non_triviality": 2,
            "scope_quality": 2,
            "actionability": 1
          },
          "decision_reason": "One weak dimension, but overall strong.",
          "summary_for_user": "Save this as a reusable workflow.",
          "target_skill": ""
        }"""
    )
    with patch("agent.skill_admission.call_llm", return_value=fake_response):
        verdict = run_admission_review(
            candidate,
            main_runtime=_MAIN_RUNTIME,
        )

    assert verdict["decision"] == DECISION_PROMOTE_NEW
    assert verdict["scores"]["verification"] == 0
    assert verdict["total_score"] == 7
    assert verdict["threshold_used"] == 7


def test_automatic_candidate_json_with_trailing_commas_is_repaired():
    messages_snapshot = [
        {"role": "user", "content": "figure out how to rerun the failing pytest subset"},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "function": {
                        "name": "terminal",
                        "arguments": '{"command": "pytest tests/test_model_tools.py -q"}',
                    }
                }
            ],
        },
    ]
    proposal_response = _response(
        """{
          "save": true,
          "name": "pytest-subset-rerun",
          "description": "Rerun a focused pytest subset.",
          "category": "",
          "scope": "repo-specific pytest debugging",
          "why_created": "Captures a repeatable way to rerun the relevant failing subset.",
          "reusability_rationale": "Useful for future targeted pytest investigations in this repo.",
          "known_limits": "",
          "verification_summary": "Validated with a targeted pytest run.",
        }"""
    )
    draft_response = _response(
        """```markdown
        # Steps

        - Run the targeted pytest command.
        ```"""
    )
    with patch("agent.skill_admission.call_llm", side_effect=[proposal_response, draft_response]) as mock_call_llm:
        proposal = maybe_generate_automatic_candidate(
            messages_snapshot,
            main_runtime=_MAIN_RUNTIME,
        )

    assert proposal["created"] is True
    assert proposal["name"] == "pytest-subset-rerun"
    assert proposal["commands_run"] == ["pytest tests/test_model_tools.py -q"]
    assert mock_call_llm.call_count == 2
    assert proposal["content"].startswith("---\nname: pytest-subset-rerun\n")
    assert "description: Rerun a focused pytest subset." in proposal["content"]


def test_automatic_candidate_accepts_reasoning_content_json_and_markdown():
    messages_snapshot = [
        {"role": "user", "content": "figure out how to rerun the failing pytest subset"},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "function": {
                        "name": "terminal",
                        "arguments": '{"command": "pytest tests/test_model_tools.py -q"}',
                    }
                }
            ],
        },
    ]
    proposal_response = _reasoning_response(
        reasoning_content="""Here is the normalized proposal.

```json
{
  "save": true,
  "name": "pytest-subset-rerun",
  "description": "Rerun a focused pytest subset.",
  "category": "",
  "scope": "repo-specific pytest debugging",
  "why_created": "Captures a repeatable way to rerun the relevant failing subset.",
  "reusability_rationale": "Useful for future targeted pytest investigations in this repo.",
  "known_limits": "",
  "verification_summary": "Validated with a targeted pytest run.",
}
```"""
    )
    draft_response = _reasoning_response(
        reasoning_content="""```markdown
# Steps

- Run the targeted pytest command.
```"""
    )
    with patch("agent.skill_admission.call_llm", side_effect=[proposal_response, draft_response]):
        proposal = maybe_generate_automatic_candidate(
            messages_snapshot,
            main_runtime=_MAIN_RUNTIME,
        )

    assert proposal["created"] is True
    assert proposal["name"] == "pytest-subset-rerun"
    assert "# Steps" in proposal["content"]


def test_automatic_candidate_passes_live_main_runtime_to_aux_calls():
    messages_snapshot = [
        {"role": "user", "content": "capture the reusable pytest rerun workflow"},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "function": {
                        "name": "terminal",
                        "arguments": '{"command": "pytest tests/test_model_tools.py -q"}',
                    }
                }
            ],
        },
    ]
    proposal_response = _response(
        """{
          "save": true,
          "name": "pytest-subset-rerun",
          "description": "Rerun a focused pytest subset.",
          "category": "",
          "scope": "repo-specific pytest debugging",
          "why_created": "Captures a repeatable way to rerun the relevant failing subset.",
          "reusability_rationale": "Useful for future targeted pytest investigations in this repo.",
          "known_limits": "",
          "verification_summary": "Validated with a targeted pytest run."
        }"""
    )
    draft_response = _response(
        """# Steps

- Run the targeted pytest command.
"""
    )
    with patch("agent.skill_admission.call_llm", side_effect=[proposal_response, draft_response]) as mock_call_llm:
        result = maybe_generate_automatic_candidate(
            messages_snapshot,
            main_runtime=_MAIN_RUNTIME,
        )

    assert result["created"] is True
    assert mock_call_llm.call_count == 2
    assert all(call.kwargs["main_runtime"] == _MAIN_RUNTIME for call in mock_call_llm.call_args_list)


def test_automatic_candidate_save_false_skips_draft_generation():
    messages_snapshot = [
        {"role": "user", "content": "what branch am I on?"},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "function": {
                        "name": "terminal",
                        "arguments": '{"command": "git branch --show-current"}',
                    }
                }
            ],
        },
    ]
    proposal_response = _response('{"save": false}')
    with patch("agent.skill_admission.call_llm", return_value=proposal_response) as mock_call_llm:
        result = maybe_generate_automatic_candidate(
            messages_snapshot,
            main_runtime=_MAIN_RUNTIME,
        )

    assert result["created"] is False
    assert result["reason"] == "reviewer opted out"
    assert mock_call_llm.call_count == 1


def test_automatic_candidate_malformed_proposal_returns_not_created():
    messages_snapshot = [
        {"role": "user", "content": "capture a reusable workflow"},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "function": {
                        "name": "terminal",
                        "arguments": '{"command": "pytest tests/test_model_tools.py -q"}',
                    }
                }
            ],
        },
    ]
    with patch("agent.skill_admission.call_llm", return_value=_response("not valid json")):
        result = maybe_generate_automatic_candidate(
            messages_snapshot,
            main_runtime=_MAIN_RUNTIME,
        )

    assert result["created"] is False
    assert "proposal generation failed" in result["reason"]


def test_malformed_reviewer_reasoning_content_still_rejects():
    candidate = _candidate()
    fake_response = _reasoning_response(
        reasoning_content="I cannot produce structured output for this candidate."
    )
    with patch("agent.skill_admission.call_llm", return_value=fake_response):
        verdict = run_admission_review(
            candidate,
            main_runtime=_MAIN_RUNTIME,
        )

    assert verdict["decision"] == DECISION_REJECT
    assert "Malformed reviewer output" in verdict["decision_reason"]


def test_automatic_candidate_missing_proposal_metadata_returns_not_created():
    messages_snapshot = [
        {"role": "user", "content": "capture a reusable workflow"},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "function": {
                        "name": "terminal",
                        "arguments": '{"command": "pytest tests/test_model_tools.py -q"}',
                    }
                }
            ],
        },
    ]
    proposal_response = _response(
        """{
          "save": true,
          "name": "pytest-subset-rerun",
          "description": "",
          "category": "",
          "scope": "repo-specific pytest debugging",
          "why_created": "Captures a repeatable way to rerun the relevant failing subset.",
          "reusability_rationale": "Useful for future targeted pytest investigations in this repo.",
          "known_limits": "",
          "verification_summary": "Validated with a targeted pytest run."
        }"""
    )
    with patch("agent.skill_admission.call_llm", return_value=proposal_response) as mock_call_llm:
        result = maybe_generate_automatic_candidate(
            messages_snapshot,
            main_runtime=_MAIN_RUNTIME,
        )

    assert result["created"] is False
    assert result["reason"] == "proposal missing required field: description"
    assert mock_call_llm.call_count == 1


def test_automatic_candidate_transcript_excludes_system_messages():
    messages_snapshot = [
        {"role": "system", "content": "system prompt override with policy and loaded skills"},
        {"role": "user", "content": "capture the reusable pytest rerun workflow"},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "function": {
                        "name": "terminal",
                        "arguments": '{"command": "pytest tests/test_model_tools.py -q"}',
                    }
                }
            ],
        },
    ]
    proposal_response = _response(
        """{
          "save": true,
          "name": "pytest-subset-rerun",
          "description": "Rerun a focused pytest subset.",
          "category": "",
          "scope": "repo-specific pytest debugging",
          "why_created": "Captures a repeatable way to rerun the relevant failing subset.",
          "reusability_rationale": "Useful for future targeted pytest investigations in this repo.",
          "known_limits": "",
          "verification_summary": "Validated with a targeted pytest run."
        }"""
    )
    draft_response = _response(
        """```markdown
        # Steps

        - Run the targeted pytest command.
        ```"""
    )
    captured_user_payloads = []

    def _fake_call_llm(*, messages, **kwargs):
        captured_user_payloads.append(messages[1]["content"])
        return proposal_response if len(captured_user_payloads) == 1 else draft_response

    with patch("agent.skill_admission.call_llm", side_effect=_fake_call_llm):
        result = maybe_generate_automatic_candidate(
            messages_snapshot,
            main_runtime=_MAIN_RUNTIME,
        )

    assert result["created"] is True
    assert len(captured_user_payloads) == 2
    assert all("system prompt override with policy and loaded skills" not in payload for payload in captured_user_payloads)
    assert "[USER] capture the reusable pytest rerun workflow" in captured_user_payloads[0]


def test_automatic_candidate_empty_draft_returns_not_created():
    messages_snapshot = [
        {"role": "user", "content": "capture a reusable workflow"},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "function": {
                        "name": "terminal",
                        "arguments": '{"command": "pytest tests/test_model_tools.py -q"}',
                    }
                }
            ],
        },
    ]
    proposal_response = _response(
        """{
          "save": true,
          "name": "pytest-subset-rerun",
          "description": "Rerun a focused pytest subset.",
          "category": "",
          "scope": "repo-specific pytest debugging",
          "why_created": "Captures a repeatable way to rerun the relevant failing subset.",
          "reusability_rationale": "Useful for future targeted pytest investigations in this repo.",
          "known_limits": "",
          "verification_summary": "Validated with a targeted pytest run."
        }"""
    )
    draft_response = _response("   ")
    with patch("agent.skill_admission.call_llm", side_effect=[proposal_response, draft_response]):
        result = maybe_generate_automatic_candidate(
            messages_snapshot,
            main_runtime=_MAIN_RUNTIME,
        )

    assert result["created"] is False
    assert result["reason"] == "draft generation returned empty content"


def test_configurable_automatic_threshold_is_used():
    candidate = _candidate()
    fake_response = _response(
        """{
          "scores": {
            "reusability": 1,
            "verification": 1,
            "non_triviality": 1,
            "scope_quality": 1,
            "actionability": 0
          },
          "decision_reason": "Barely sufficient under the default threshold.",
          "summary_for_user": "",
          "target_skill": ""
        }"""
    )
    with (
        patch("agent.skill_admission.call_llm", return_value=fake_response),
        patch(
            "hermes_cli.config.load_config",
            return_value={"skills": {"automatic_min_score": 9}},
        ),
    ):
        verdict = run_admission_review(
            candidate,
            main_runtime=_MAIN_RUNTIME,
        )

    assert verdict["decision"] == DECISION_REJECT
    assert verdict["threshold_used"] == 9


def test_admission_review_passes_live_main_runtime_to_aux_call():
    candidate = _candidate()
    fake_response = _response(
        """{
          "scores": {
            "reusability": 2,
            "verification": 2,
            "non_triviality": 2,
            "scope_quality": 1,
            "actionability": 1
          },
          "decision_reason": "Structured and sufficiently reusable.",
          "summary_for_user": "Save this bounded pytest workflow.",
          "target_skill": ""
        }"""
    )
    with patch("agent.skill_admission.call_llm", return_value=fake_response) as mock_call_llm:
        verdict = run_admission_review(
            candidate,
            main_runtime=_MAIN_RUNTIME,
        )

    assert verdict["decision"] == DECISION_PROMOTE_NEW
    assert mock_call_llm.call_args.kwargs["main_runtime"] == _MAIN_RUNTIME
