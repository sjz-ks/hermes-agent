"""
Strict skill-creation helpers: candidate generation + admission review.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Iterable, List, Optional

from agent.auxiliary_client import call_llm, extract_content_or_reasoning
from json_repair import loads as repair_json_loads
import yaml
from tools.skill_manager_tool import _validate_content_size, _validate_frontmatter
from tools.skill_candidate_tool import (
    DECISION_PROMOTE_NEW,
    DECISION_REJECT,
    ORIGIN_AUTOMATIC_REVIEW,
)

RUBRIC_FIELDS = (
    "reusability",
    "verification",
    "non_triviality",
    "scope_quality",
    "actionability",
)
DEFAULT_AUTOMATIC_MIN_SCORE = 7
_CREATE_RULE_REQUIRED_FIELDS = ("scope", "why_created", "verification_summary")
_AUTOMATIC_PROPOSAL_REQUIRED_FIELDS = (
    "name",
    "description",
    "scope",
    "why_created",
    "reusability_rationale",
    "verification_summary",
)
_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(\{.*\})\s*```", re.S)
_MARKDOWN_BLOCK_RE = re.compile(r"^```(?:markdown|md)?\s*(.*?)\s*```$", re.S)


def _extract_json_object(text: str) -> Dict[str, Any]:
    raw = (text or "").strip()
    if not raw:
        raise ValueError("empty reviewer output")
    match = _JSON_BLOCK_RE.search(raw)
    if match:
        raw = match.group(1).strip()
    candidates: List[str] = []
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1:
        candidates.append(raw[start:])
        if end != -1 and end >= start:
            candidates.append(raw[start : end + 1])
    candidates.append(raw)

    seen: set[str] = set()
    last_error = "no JSON object found"
    for candidate in candidates:
        candidate = candidate.strip()
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        try:
            payload = repair_json_loads(candidate)
        except Exception as exc:
            last_error = str(exc)
            continue
        if isinstance(payload, dict):
            return payload
        last_error = f"decoded JSON was {type(payload).__name__}, expected object"

    raise ValueError(last_error)


def _render_message_preview(message: Dict[str, Any], limit: int = 500) -> str:
    content = message.get("content")
    if isinstance(content, list):
        parts: List[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text") or ""))
        text = "\n".join(parts)
    else:
        text = str(content or "")
    text = text.strip()
    if len(text) > limit:
        return text[:limit] + "..."
    return text


def extract_skill_evidence(messages_snapshot: Iterable[Dict[str, Any]]) -> Dict[str, List[str]]:
    tool_calls: List[str] = []
    commands_run: List[str] = []
    tests_run: List[str] = []
    for msg in messages_snapshot:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        if role == "assistant":
            raw_calls = msg.get("tool_calls") or []
            for tc in raw_calls if isinstance(raw_calls, list) else []:
                try:
                    if isinstance(tc, dict):
                        fn = ((tc.get("function") or {}).get("name") or "").strip()
                        args = (tc.get("function") or {}).get("arguments")
                    else:
                        fn = str(getattr(getattr(tc, "function", None), "name", "") or "").strip()
                        args = getattr(getattr(tc, "function", None), "arguments", None)
                    if not fn:
                        continue
                    tool_calls.append(fn)
                    parsed_args = json.loads(args) if isinstance(args, str) and args.strip() else (args or {})
                    if isinstance(parsed_args, dict) and fn == "terminal":
                        cmd = str(parsed_args.get("command") or "").strip()
                        if cmd:
                            commands_run.append(cmd)
                            lowered = cmd.lower()
                            if any(
                                tok in lowered
                                for tok in (
                                    "pytest",
                                    "npm test",
                                    "pnpm test",
                                    "yarn test",
                                    "cargo test",
                                    "go test",
                                )
                            ):
                                tests_run.append(cmd)
                except Exception:
                    continue
    return {
        "tool_calls": tool_calls,
        "commands_run": commands_run,
        "tests_run": tests_run,
    }


def _render_review_transcript(messages_snapshot: Iterable[Dict[str, Any]], max_chars: int = 12000) -> str:
    lines: List[str] = []
    for msg in list(messages_snapshot)[-60:]:
        if not isinstance(msg, dict):
            continue
        role = str(msg.get("role") or "unknown").upper()
        # Deliberately exclude injected system prompts so automatic skill review
        # only sees the user/assistant/tool exchange from the actual workflow.
        if role == "SYSTEM":
            continue
        if role == "TOOL":
            tool_id = str(msg.get("tool_call_id") or "")
            preview = _render_message_preview(msg, limit=350)
            if preview:
                lines.append(f"[{role} {tool_id}] {preview}")
        else:
            preview = _render_message_preview(msg, limit=500)
            if preview:
                lines.append(f"[{role}] {preview}")
    rendered = "\n\n".join(lines)
    if len(rendered) > max_chars:
        return rendered[-max_chars:]
    return rendered


def _build_automatic_proposal_messages(
    messages_snapshot: Iterable[Dict[str, Any]],
    evidence: Dict[str, List[str]],
) -> List[Dict[str, Any]]:
    transcript = _render_review_transcript(messages_snapshot)
    evidence_block = json.dumps(evidence, ensure_ascii=False, indent=2)
    return [
        {
            "role": "system",
            "content": (
                "You are reviewing a completed Hermes agent conversation for a potential new skill candidate. "
                "Return JSON only.\n\n"
                "Your job is to decide whether this conversation is worth turning into a skill candidate for "
                "downstream review. Do not apply the standard for a final trusted skill here.\n\n"
                "Return exactly one JSON object in one of these shapes:\n"
                "{\"save\": false}\n"
                "or\n"
                "{"
                "\"save\": true, "
                "\"name\": \"skill-name\", "
                "\"description\": \"one-line description\", "
                "\"category\": \"optional-category-or-empty\", "
                "\"scope\": \"where this skill applies\", "
                "\"why_created\": \"why this should be saved\", "
                "\"reusability_rationale\": \"why the procedure is reusable\", "
                "\"known_limits\": \"optional limits or caveats\", "
                "\"verification_summary\": \"how the approach was validated\""
                "}\n"
                "Use save=false for trivial lookups, one-off results, conversations without a reusable workflow, "
                "or cases with no meaningful evidence.\n"
                "Use save=true when the conversation shows a non-trivial workflow, especially if it includes "
                "multiple steps, trial and error, changing course, useful pitfalls or constraints, explicit "
                "validation, or a workflow that might be reused even if its scope is somewhat narrow.\n"
                "A later admission review and explicit user approval will decide whether the candidate becomes "
                "a trusted skill. Do not draft the skill body."
            ),
        },
        {
            "role": "user",
            "content": (
                "Review this conversation and decide whether it warrants a new skill candidate for downstream "
                "review.\n\n"
                f"Conversation:\n{transcript}\n\n"
                f"Evidence:\n{evidence_block}"
            ),
        },
    ]


def _build_automatic_draft_messages(
    messages_snapshot: Iterable[Dict[str, Any]],
    evidence: Dict[str, List[str]],
    proposal: Dict[str, Any],
) -> List[Dict[str, Any]]:
    transcript = _render_review_transcript(messages_snapshot)
    evidence_block = json.dumps(evidence, ensure_ascii=False, indent=2)
    proposal_block = json.dumps(proposal, ensure_ascii=False, indent=2)
    return [
        {
            "role": "system",
            "content": (
                "You are drafting the body of a Hermes skill from a completed agent conversation.\n\n"
                "Write only the markdown body of the skill. Do not include YAML frontmatter. "
                "Focus on the reusable workflow, concrete steps, important pitfalls, validation, and scope limits "
                "when relevant. Do not decide whether the skill should be saved. Do not return JSON."
            ),
        },
        {
            "role": "user",
            "content": (
                "Draft the markdown body for this skill candidate.\n\n"
                f"Accepted proposal:\n{proposal_block}\n\n"
                f"Conversation:\n{transcript}\n\n"
                f"Evidence:\n{evidence_block}"
            ),
        },
    ]


def _extract_markdown_body(text: str) -> str:
    raw = (text or "").strip()
    if not raw:
        return ""
    match = _MARKDOWN_BLOCK_RE.match(raw)
    if match:
        raw = match.group(1).strip()
    if raw.startswith("---"):
        end_match = re.search(r"\n---\s*\n", raw[3:])
        if end_match:
            raw = raw[end_match.end() + 3 :].strip()
    return raw.strip()


def _assemble_skill_content(name: str, description: str, body: str) -> str:
    yaml_content = yaml.safe_dump(
        {"name": name, "description": description},
        allow_unicode=True,
        sort_keys=False,
    ).strip()
    return f"---\n{yaml_content}\n---\n\n{body.strip()}\n"


def _generate_automatic_proposal(
    messages_snapshot: Iterable[Dict[str, Any]],
    evidence: Dict[str, List[str]],
    *,
    main_runtime: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    # Reuse the live main runtime so strict side-calls honor the same
    # provider wire mode (Responses, Anthropic messages, Bedrock, etc.).
    response = call_llm(
        main_runtime=main_runtime,
        messages=_build_automatic_proposal_messages(messages_snapshot, evidence),
        temperature=0,
        max_tokens=1100,
    )
    # auxiliary_client owns provider/response-shape compatibility; strict
    # skill admission owns the structured-output parsing and repair policy.
    content = extract_content_or_reasoning(response)
    payload = _extract_json_object(content)
    if not payload.get("save"):
        return {"success": True, "created": False, "reason": "reviewer opted out"}

    normalized = {
        "name": str(payload.get("name") or "").strip(),
        "description": str(payload.get("description") or "").strip(),
        "category": str(payload.get("category") or "").strip(),
        "scope": str(payload.get("scope") or "").strip(),
        "why_created": str(payload.get("why_created") or "").strip(),
        "reusability_rationale": str(payload.get("reusability_rationale") or "").strip(),
        "known_limits": str(payload.get("known_limits") or "").strip(),
        "verification_summary": str(payload.get("verification_summary") or "").strip(),
    }
    for field in _AUTOMATIC_PROPOSAL_REQUIRED_FIELDS:
        if not normalized[field]:
            return {"success": True, "created": False, "reason": f"proposal missing required field: {field}"}

    return {"success": True, "created": True, "proposal": normalized}


def _generate_automatic_skill_draft(
    messages_snapshot: Iterable[Dict[str, Any]],
    evidence: Dict[str, List[str]],
    proposal: Dict[str, Any],
    *,
    main_runtime: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    response = call_llm(
        main_runtime=main_runtime,
        messages=_build_automatic_draft_messages(messages_snapshot, evidence, proposal),
        temperature=0,
        max_tokens=1400,
    )
    content = extract_content_or_reasoning(response)
    body = _extract_markdown_body(content)
    if not body:
        return {"success": True, "created": False, "reason": "draft generation returned empty content"}

    skill_content = _assemble_skill_content(
        proposal["name"],
        proposal["description"],
        body,
    )
    err = _validate_frontmatter(skill_content) or _validate_content_size(skill_content)
    if err:
        return {"success": True, "created": False, "reason": f"draft assembly produced invalid SKILL.md: {err}"}
    return {"success": True, "created": True, "content": skill_content}


def maybe_generate_automatic_candidate(
    messages_snapshot: Iterable[Dict[str, Any]],
    *,
    main_runtime: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    evidence = extract_skill_evidence(messages_snapshot)
    if not evidence["tool_calls"]:
        return {"success": True, "created": False, "reason": "no tool evidence"}

    try:
        proposal_result = _generate_automatic_proposal(
            messages_snapshot,
            evidence,
            main_runtime=main_runtime,
        )
    except Exception as exc:
        return {"success": True, "created": False, "reason": f"proposal generation failed: {exc}"}
    if not proposal_result.get("success") or not proposal_result.get("created"):
        return proposal_result

    proposal = dict(proposal_result.get("proposal") or {})
    try:
        draft_result = _generate_automatic_skill_draft(
            messages_snapshot,
            evidence,
            proposal,
            main_runtime=main_runtime,
        )
    except Exception as exc:
        return {"success": True, "created": False, "reason": f"draft generation failed: {exc}"}
    if not draft_result.get("success") or not draft_result.get("created"):
        return draft_result

    return {
        "success": True,
        "created": True,
        "name": proposal["name"],
        "category": proposal["category"],
        "scope": proposal["scope"],
        "why_created": proposal["why_created"],
        "reusability_rationale": proposal["reusability_rationale"],
        "known_limits": proposal["known_limits"],
        "verification_summary": proposal["verification_summary"],
        "content": draft_result["content"],
        "tool_calls": evidence["tool_calls"],
        "commands_run": evidence["commands_run"],
        "tests_run": evidence["tests_run"],
    }


def _default_scores() -> Dict[str, int]:
    return {field: 0 for field in RUBRIC_FIELDS}


def _load_automatic_score_threshold() -> int:
    try:
        from hermes_cli.config import load_config

        config = load_config()
        skills_cfg = config.get("skills", {}) if isinstance(config, dict) else {}
        return int(skills_cfg.get("automatic_min_score", DEFAULT_AUTOMATIC_MIN_SCORE))
    except Exception:
        return DEFAULT_AUTOMATIC_MIN_SCORE


def _reject_result(
    reason: str,
    *,
    summary_for_user: str = "",
    target_skill: str = "",
    scores: Optional[Dict[str, int]] = None,
) -> Dict[str, Any]:
    normalized_scores = dict(_default_scores())
    if scores:
        for field in RUBRIC_FIELDS:
            normalized_scores[field] = int(scores.get(field, 0))
    return {
        "scores": normalized_scores,
        "total_score": sum(normalized_scores.values()),
        "threshold_used": _load_automatic_score_threshold(),
        "decision": DECISION_REJECT,
        "decision_reason": reason,
        "summary_for_user": summary_for_user,
        "target_skill": target_skill,
    }


def run_admission_rule_filter(candidate: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    origin = str(candidate.get("origin") or "")
    if origin != ORIGIN_AUTOMATIC_REVIEW:
        return _reject_result(
            "Admission review only handles automatic skill candidates.",
        )
    for field in _CREATE_RULE_REQUIRED_FIELDS:
        if not str(candidate.get(field) or "").strip():
            return _reject_result(
                f"Missing required field: {field}.",
            )

    has_evidence = any(candidate.get(key) for key in ("tool_calls", "commands_run", "tests_run"))
    if not has_evidence:
        return _reject_result(
            "Automatic candidates need concrete tool or command evidence.",
        )

    return None


def _build_admission_messages(candidate: Dict[str, Any]) -> List[Dict[str, Any]]:
    origin_policy = (
        "This candidate was proposed automatically. Score conservatively. Only clearly reusable, well-evidenced, "
        "non-trivial workflows should score highly enough to pass."
    )
    payload = {
        "proposed_name": candidate.get("proposed_name", ""),
        "category": candidate.get("category", ""),
        "scope": candidate.get("scope", ""),
        "why_created": candidate.get("why_created", ""),
        "reusability_rationale": candidate.get("reusability_rationale", ""),
        "known_limits": candidate.get("known_limits", ""),
        "verification_summary": candidate.get("verification_summary", ""),
        "tool_calls": candidate.get("tool_calls", []),
        "commands_run": candidate.get("commands_run", []),
        "tests_run": candidate.get("tests_run", []),
        "skill_content": candidate.get("skill_content", ""),
    }
    rubric = (
        "Scoring rubric (0-2 each):\n"
        "- reusability: 0 one-off/specific, 1 partially reusable, 2 clearly reusable workflow\n"
        "- verification: 0 little/no validation, 1 partial validation, 2 strong explicit validation\n"
        "- non_triviality: 0 obvious/simple, 1 some procedure value, 2 meaningful trial-and-error or pitfall value\n"
        "- scope_quality: 0 vague or over-broad, 1 partly bounded, 2 clear scope and boundaries\n"
        "- actionability: 0 too abstract, 1 somewhat actionable, 2 concrete and directly usable\n"
    )
    return [
        {
            "role": "system",
            "content": (
                "You are the admission reviewer for Hermes skill candidates. "
                "Return JSON only. Do not make the final promote/reject decision; only score the candidate.\n\n"
                f"{rubric}\n"
                "Output schema:\n"
                "{"
                "\"scores\": {"
                "\"reusability\": 0, "
                "\"verification\": 0, "
                "\"non_triviality\": 0, "
                "\"scope_quality\": 0, "
                "\"actionability\": 0"
                "}, "
                "\"decision_reason\": \"short rationale for the score profile\", "
                "\"summary_for_user\": \"brief human-facing summary for pending manual review\", "
                "\"target_skill\": \"\""
                "}\n\n"
                f"{origin_policy}"
            ),
        },
        {
            "role": "user",
            "content": "Review this skill candidate:\n\n" + json.dumps(payload, ensure_ascii=False, indent=2),
        },
    ]


def _parse_scores(payload: Dict[str, Any]) -> Optional[Dict[str, int]]:
    scores = payload.get("scores")
    if not isinstance(scores, dict):
        return None
    parsed: Dict[str, int] = {}
    for field in RUBRIC_FIELDS:
        value = scores.get(field)
        if isinstance(value, str):
            value = value.strip()
            if value in {"0", "1", "2"}:
                value = int(value)
        if value not in {0, 1, 2}:
            return None
        parsed[field] = int(value)
    return parsed


def run_admission_review(
    candidate: Dict[str, Any],
    *,
    main_runtime: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Score an automatic skill candidate before it is shown for manual approval."""
    filtered = run_admission_rule_filter(candidate)
    if filtered is not None:
        return filtered

    threshold_used = _load_automatic_score_threshold()

    response = call_llm(
        main_runtime=main_runtime,
        messages=_build_admission_messages(candidate),
        temperature=0,
        max_tokens=5000,
    )
    content = extract_content_or_reasoning(response)
    try:
        payload = _extract_json_object(content)
    except Exception as exc:
        # Deliberate tradeoff: until strict review supports a surfaced
        # retryable-failure state, malformed reviewer output is treated as a
        # rejection rather than leaving the candidate in limbo.
        return _reject_result(
            f"Malformed reviewer output: {exc}.",
        )

    scores = _parse_scores(payload)
    if scores is None:
        return _reject_result(
            "Reviewer output missing valid rubric scores (expected integers 0-2 for all five dimensions).",
        )

    total_score = sum(scores.values())
    decision = DECISION_PROMOTE_NEW if total_score >= threshold_used else DECISION_REJECT
    reviewer_rationale = str(payload.get("decision_reason") or "").strip()
    if decision == DECISION_PROMOTE_NEW:
        decision_reason = (
            f"Promoted because total score {total_score}/10 met the threshold {threshold_used}/10."
            + (f" Reviewer rationale: {reviewer_rationale}" if reviewer_rationale else "")
        )
    else:
        decision_reason = (
            f"Rejected because total score {total_score}/10 was below the threshold {threshold_used}/10."
            + (f" Reviewer rationale: {reviewer_rationale}" if reviewer_rationale else "")
        )

    return {
        "scores": scores,
        "total_score": total_score,
        "threshold_used": threshold_used,
        "decision": decision,
        "decision_reason": decision_reason,
        "summary_for_user": str(payload.get("summary_for_user") or "").strip(),
        "target_skill": str(payload.get("target_skill") or "").strip(),
    }
