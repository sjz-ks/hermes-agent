#!/usr/bin/env python3
"""
Skill Candidate Tool -- strict-mode staging area for new skill creation.

In ``skills.strict_creation_mode``, new skills no longer land directly in the
trusted skill store. They are first written as candidates under
``$HERMES_HOME/skill-candidates/`` and later promoted into trusted skills after
manual approval. Automatic proposals also pass an admission review before they
enter the manual review queue.
"""

from __future__ import annotations

import json
import re
import shutil
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from hermes_constants import get_hermes_home
from tools.skill_manager_tool import (
    _create_skill,
    _find_skill,
    _resolve_skill_target,
    _security_scan_skill,
    _validate_category,
    _validate_content_size,
    _validate_file_path,
    _validate_frontmatter,
    _validate_name,
)

SCHEMA_VERSION = 1
CANDIDATES_DIR = get_hermes_home() / "skill-candidates"

STATUS_CANDIDATE = "candidate"
STATUS_REVIEWED = "reviewed"
STATUS_APPROVED = "approved_by_user"
STATUS_PROMOTED = "promoted"
STATUS_REJECTED = "rejected"

DECISION_PROMOTE_NEW = "promote_new"
DECISION_REJECT = "reject"
_VALID_DECISIONS = {DECISION_PROMOTE_NEW, DECISION_REJECT}
_REVIEW_SCORE_FIELDS = (
    "reusability",
    "verification",
    "non_triviality",
    "scope_quality",
    "actionability",
)

ORIGIN_USER_REQUESTED = "user_requested"
ORIGIN_AUTOMATIC_REVIEW = "automatic_review"
_VALID_ORIGINS = {
    ORIGIN_USER_REQUESTED,
    ORIGIN_AUTOMATIC_REVIEW,
}
_CANDIDATE_ID_RE = re.compile(r"^\d{8}_\d{6}_[0-9a-f]{8}$")
_SUPPORTING_FILE_SUBDIRS = ("assets", "references", "scripts", "templates")


def _error_result(message: str) -> Dict[str, Any]:
    return {"success": False, "error": str(message)}


def _ensure_candidates_dir() -> Path:
    CANDIDATES_DIR.mkdir(parents=True, exist_ok=True)
    return CANDIDATES_DIR


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        delete=False,
        dir=str(path.parent),
        prefix=f".{path.name}.",
        suffix=".tmp",
    ) as tmp:
        tmp.write(content)
        tmp_path = Path(tmp.name)
    tmp_path.replace(path)


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    _atomic_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


def _candidate_dir(candidate_id: str) -> Path:
    candidate_id = str(candidate_id or "").strip()
    if not _CANDIDATE_ID_RE.fullmatch(candidate_id):
        raise FileNotFoundError(f"Candidate '{candidate_id}' not found.")
    return _ensure_candidates_dir() / candidate_id


def _candidate_skill_path(candidate_id: str) -> Path:
    return _candidate_dir(candidate_id) / "SKILL.md"


def _candidate_meta_path(candidate_id: str) -> Path:
    return _candidate_dir(candidate_id) / "candidate.json"


def _list_candidate_supporting_files(candidate_id: str) -> List[str]:
    candidate_dir = _candidate_dir(candidate_id)
    files: List[str] = []
    for subdir in _SUPPORTING_FILE_SUBDIRS:
        root = candidate_dir / subdir
        if not root.exists():
            continue
        for path in sorted(root.rglob("*")):
            if path.is_file():
                files.append(str(path.relative_to(candidate_dir)))
    return files


def _copy_candidate_supporting_files(candidate_id: str, skill_dir: Path) -> int:
    candidate_dir = _candidate_dir(candidate_id)
    copied = 0
    # Known limitation: /skillreviews view shows metadata-backed supporting_files,
    # while promotion copies allowed candidate subdirs from disk; direct writes
    # into the candidate dir can therefore bypass the displayed list.
    for subdir in _SUPPORTING_FILE_SUBDIRS:
        source_root = candidate_dir / subdir
        if not source_root.exists():
            continue
        for source in sorted(source_root.rglob("*")):
            relative = source.relative_to(candidate_dir)
            target = skill_dir / relative
            if source.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            if source.is_symlink() or not source.is_file():
                raise OSError(f"Unsupported staged candidate file type: {relative}")
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            copied += 1
    return copied


def _normalize_str_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _extract_frontmatter_name(content: str) -> str:
    if not content.startswith("---"):
        return ""
    end_idx = content.find("\n---", 3)
    if end_idx == -1:
        return ""
    yaml_content = content[3:end_idx]
    try:
        parsed = yaml.safe_load(yaml_content) or {}
    except Exception:
        return ""
    if not isinstance(parsed, dict):
        return ""
    return str(parsed.get("name") or "").strip()


def _candidate_sort_key(path: Path) -> tuple[str, str]:
    try:
        data = json.loads((path / "candidate.json").read_text(encoding="utf-8"))
        return (str(data.get("created_at") or ""), path.name)
    except Exception:
        return ("", path.name)


def _find_live_candidate_by_name(name: str) -> Optional[Dict[str, Any]]:
    live_statuses = {
        STATUS_CANDIDATE,
        STATUS_REVIEWED,
        STATUS_APPROVED,
    }
    root = _ensure_candidates_dir()
    for child in sorted(root.iterdir(), key=_candidate_sort_key):
        if not child.is_dir():
            continue
        meta_path = child / "candidate.json"
        if not meta_path.exists():
            continue
        try:
            data = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if str(data.get("proposed_name") or "").strip() != name:
            continue
        if str(data.get("status") or "").strip() not in live_statuses:
            continue
        return data
    return None


def find_live_candidate_by_name(name: str) -> Optional[Dict[str, Any]]:
    return _find_live_candidate_by_name((name or "").strip())


def _load_candidate(candidate_id: str) -> Dict[str, Any]:
    meta_path = _candidate_meta_path(candidate_id)
    skill_path = _candidate_skill_path(candidate_id)
    if not meta_path.exists():
        raise FileNotFoundError(f"Candidate '{candidate_id}' not found.")
    data = json.loads(meta_path.read_text(encoding="utf-8"))
    if skill_path.exists():
        data["skill_content"] = skill_path.read_text(encoding="utf-8")
    return data


def _save_candidate(candidate: Dict[str, Any]) -> None:
    candidate_id = str(candidate["id"])
    candidate_dir = _candidate_dir(candidate_id)
    candidate_dir.mkdir(parents=True, exist_ok=True)

    skill_content = str(candidate.get("skill_content") or "")
    meta = dict(candidate)
    meta.pop("skill_content", None)
    meta["updated_at"] = _utc_now()

    _atomic_write_text(_candidate_skill_path(candidate_id), skill_content)
    _atomic_write_json(_candidate_meta_path(candidate_id), meta)


def _base_candidate_payload(
    *,
    candidate_id: str,
    proposed_name: str,
    category: Optional[str],
    skill_content: str,
    origin: str,
    scope: str,
    why_created: str,
    verification_summary: str,
    reusability_rationale: Optional[str],
    known_limits: Optional[str],
    source_session_id: Optional[str],
    source_turn_range: Optional[List[int]],
    tool_calls: Optional[List[str]],
    commands_run: Optional[List[str]],
    tests_run: Optional[List[str]],
) -> Dict[str, Any]:
    now = _utc_now()
    return {
        "schema_version": SCHEMA_VERSION,
        "id": candidate_id,
        "status": STATUS_CANDIDATE,
        "created_at": now,
        "updated_at": now,
        "origin": origin,
        "source_session_id": source_session_id or "",
        "source_turn_range": source_turn_range or [],
        "proposed_name": proposed_name,
        "category": category or "",
        "scope": scope,
        "why_created": why_created,
        "reusability_rationale": reusability_rationale or "",
        "known_limits": known_limits or "",
        "verification_summary": verification_summary,
        "tool_calls": _normalize_str_list(tool_calls),
        "commands_run": _normalize_str_list(commands_run),
        "tests_run": _normalize_str_list(tests_run),
        "supporting_files": [],
        "review": {
            "scores": {field: 0 for field in _REVIEW_SCORE_FIELDS},
            "total_score": 0,
            "threshold_used": 0,
            "decision": "",
            "decision_reason": "",
            "summary_for_user": "",
            "target_skill": "",
        },
        "skill_content": skill_content,
    }


def create_candidate(
    *,
    name: str,
    content: str,
    category: str | None = None,
    scope: str,
    why_created: str,
    verification_summary: str,
    origin: str = ORIGIN_USER_REQUESTED,
    reusability_rationale: str | None = None,
    known_limits: str | None = None,
    source_session_id: str | None = None,
    source_turn_range: Optional[List[int]] = None,
    tool_calls: Optional[List[str]] = None,
    commands_run: Optional[List[str]] = None,
    tests_run: Optional[List[str]] = None,
) -> Dict[str, Any]:
    name = (name or "").strip()
    origin = (origin or ORIGIN_USER_REQUESTED).strip()
    scope = (scope or "").strip()
    why_created = (why_created or "").strip()
    verification_summary = (verification_summary or "").strip()

    if origin not in _VALID_ORIGINS:
        return _error_result(
            f"invalid origin '{origin}'. Use: {', '.join(sorted(_VALID_ORIGINS))}"
        )

    for label, value in (
        ("scope", scope),
        ("why_created", why_created),
        ("verification_summary", verification_summary),
    ):
        if not value:
            return _error_result(f"{label} is required for candidate creation.")

    for err in (
        _validate_name(name),
        _validate_category(category),
        _validate_frontmatter(content),
        _validate_content_size(content),
    ):
        if err:
            return _error_result(err)

    frontmatter_name = _extract_frontmatter_name(content)
    if frontmatter_name and frontmatter_name != name:
        return _error_result(
            f"Name mismatch: argument name='{name}' but SKILL.md frontmatter name='{frontmatter_name}'.",
        )

    existing = _find_skill(name)
    if existing:
        return _error_result(
            f"A skill named '{name}' already exists at {existing['path']}."
        )

    existing_candidate = _find_live_candidate_by_name(name)
    if existing_candidate:
        return _error_result(
            "A pending skill candidate named "
            f"'{name}' already exists (candidate_id={existing_candidate.get('id', '')})."
        )

    candidate_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    candidate = _base_candidate_payload(
        candidate_id=candidate_id,
        proposed_name=name,
        category=category,
        skill_content=content,
        origin=origin,
        scope=scope,
        why_created=why_created,
        verification_summary=verification_summary,
        reusability_rationale=reusability_rationale,
        known_limits=known_limits,
        source_session_id=source_session_id,
        source_turn_range=source_turn_range,
        tool_calls=tool_calls,
        commands_run=commands_run,
        tests_run=tests_run,
    )
    _save_candidate(candidate)
    return {
        "success": True,
        "message": f"Skill candidate '{name}' created.",
        "candidate_id": candidate_id,
        "status": STATUS_CANDIDATE,
        "path": str(_candidate_dir(candidate_id)),
    }


def list_candidates(status: str | None = None, pending_only: bool = False) -> Dict[str, Any]:
    root = _ensure_candidates_dir()
    candidates: List[Dict[str, Any]] = []
    for child in sorted(root.iterdir(), key=_candidate_sort_key):
        if not child.is_dir():
            continue
        meta_path = child / "candidate.json"
        if not meta_path.exists():
            continue
        try:
            data = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        current_status = str(data.get("status") or "")
        decision = str((data.get("review") or {}).get("decision") or "")
        if status and current_status != status:
            continue
        if pending_only and not (current_status == STATUS_REVIEWED and decision == DECISION_PROMOTE_NEW):
            continue
        candidates.append(
            {
                "id": data.get("id", child.name),
                "status": current_status,
                "origin": data.get("origin", ""),
                "proposed_name": data.get("proposed_name", ""),
                "category": data.get("category", ""),
                "scope": data.get("scope", ""),
                "created_at": data.get("created_at", ""),
                "summary_for_user": (data.get("review") or {}).get("summary_for_user", ""),
            }
        )
    return {"success": True, "count": len(candidates), "candidates": candidates}


def view_candidate(candidate_id: str) -> Dict[str, Any]:
    try:
        data = _load_candidate(candidate_id)
    except FileNotFoundError as exc:
        return _error_result(str(exc))
    return {"success": True, "candidate": data}


def write_candidate_file(candidate_id: str, file_path: str, file_content: str) -> Dict[str, Any]:
    # Candidate writes intentionally mirror skill_manage(write_file) validation,
    # but land under the staging directory until a user approves the candidate.
    err = _validate_file_path(file_path)
    if err:
        return _error_result(err)
    if file_content is None:
        return _error_result("file_content is required.")

    try:
        from tools import skill_manager_tool as _skill_manager_tool

        max_bytes = int(getattr(_skill_manager_tool, "MAX_SKILL_FILE_BYTES", 1_048_576))
    except Exception:
        max_bytes = 1_048_576
    content_bytes = len(str(file_content).encode("utf-8"))
    if content_bytes > max_bytes:
        return _error_result(
            f"File content is {content_bytes:,} bytes "
            f"(limit: {max_bytes:,} bytes / 1 MiB). Consider splitting into smaller files."
        )
    err = _validate_content_size(str(file_content), label=file_path)
    if err:
        return _error_result(err)

    try:
        candidate = _load_candidate(candidate_id)
    except FileNotFoundError as exc:
        return _error_result(str(exc))
    if candidate.get("status") not in {STATUS_CANDIDATE, STATUS_REVIEWED, STATUS_APPROVED}:
        return _error_result(
            f"Candidate '{candidate_id}' is not writable (status={candidate.get('status')!r})."
        )

    candidate_dir = _candidate_dir(candidate_id)
    target, err = _resolve_skill_target(candidate_dir, file_path)
    if err:
        return _error_result(err)
    _atomic_write_text(target, str(file_content))
    candidate["supporting_files"] = _list_candidate_supporting_files(candidate_id)
    _save_candidate(candidate)
    return {
        "success": True,
        "message": (
            f"File '{file_path}' staged with skill candidate '{candidate.get('proposed_name', '')}'. "
            "It will be installed when the candidate is approved."
        ),
        "candidate_id": candidate_id,
        "path": str(target),
    }


def delete_candidate(candidate_id: str) -> Dict[str, Any]:
    try:
        candidate_dir = _candidate_dir(candidate_id)
    except FileNotFoundError as exc:
        return _error_result(str(exc))
    if not candidate_dir.exists():
        return _error_result(f"Candidate '{candidate_id}' not found.")
    try:
        shutil.rmtree(candidate_dir)
    except Exception as exc:
        return _error_result(f"Failed to delete candidate '{candidate_id}': {exc}")
    return {
        "success": True,
        "message": f"Candidate '{candidate_id}' deleted.",
        "candidate_id": candidate_id,
    }


def update_candidate_review(
    candidate_id: str,
    *,
    scores: Dict[str, int],
    total_score: int,
    threshold_used: int,
    decision: str,
    decision_reason: str,
    summary_for_user: str,
    target_skill: str = "",
    status: str | None = None,
) -> Dict[str, Any]:
    if decision not in _VALID_DECISIONS:
        return _error_result(
            f"invalid review decision '{decision}'. Use: {', '.join(sorted(_VALID_DECISIONS))}",
        )
    try:
        candidate = _load_candidate(candidate_id)
    except FileNotFoundError as exc:
        return _error_result(str(exc))

    candidate.setdefault("review", {})
    candidate["review"].update(
        {
            "scores": {
                field: int(scores.get(field, 0))
                for field in _REVIEW_SCORE_FIELDS
            },
            "total_score": int(total_score),
            "threshold_used": int(threshold_used),
            "decision": decision,
            "decision_reason": decision_reason or "",
            "summary_for_user": summary_for_user or "",
            "target_skill": target_skill or "",
        }
    )
    candidate["status"] = status or (STATUS_REVIEWED if decision == DECISION_PROMOTE_NEW else STATUS_REJECTED)
    _save_candidate(candidate)
    return {
        "success": True,
        "message": f"Candidate '{candidate_id}' reviewed with decision '{decision}'.",
        "candidate_id": candidate_id,
        "status": candidate["status"],
    }


def promote_candidate(candidate_id: str) -> Dict[str, Any]:
    try:
        candidate = _load_candidate(candidate_id)
    except FileNotFoundError as exc:
        return _error_result(str(exc))

    review = candidate.get("review") or {}
    if review.get("decision") != DECISION_PROMOTE_NEW:
        return _error_result(
            f"Candidate '{candidate_id}' is not approved for promotion (decision={review.get('decision')!r})."
        )
    if candidate.get("status") not in {STATUS_REVIEWED, STATUS_APPROVED}:
        return _error_result(
            f"Candidate '{candidate_id}' is not ready for promotion (status={candidate.get('status')!r})."
        )

    candidate["status"] = STATUS_APPROVED
    _save_candidate(candidate)

    try:
        result = _create_skill(
            str(candidate.get("proposed_name") or ""),
            str(candidate.get("skill_content") or ""),
            str(candidate.get("category") or "") or None,
        )
    except Exception:
        candidate["status"] = STATUS_REVIEWED
        _save_candidate(candidate)
        raise
    if not result.get("success"):
        candidate["status"] = STATUS_REVIEWED
        _save_candidate(candidate)
        return result

    skill_md = str(result.get("skill_md") or "")
    if not skill_md:
        candidate["status"] = STATUS_REVIEWED
        _save_candidate(candidate)
        return _error_result("Skill promotion succeeded without a SKILL.md path.")
    skill_dir = Path(skill_md).parent
    try:
        copied_count = _copy_candidate_supporting_files(candidate_id, skill_dir)
        if copied_count:
            scan_error = _security_scan_skill(skill_dir)
            if scan_error:
                shutil.rmtree(skill_dir, ignore_errors=True)
                candidate["status"] = STATUS_REVIEWED
                _save_candidate(candidate)
                return _error_result(scan_error)
    except Exception as exc:
        shutil.rmtree(skill_dir, ignore_errors=True)
        candidate["status"] = STATUS_REVIEWED
        _save_candidate(candidate)
        return _error_result(f"Failed to promote staged supporting files: {exc}")

    candidate["status"] = STATUS_PROMOTED
    _save_candidate(candidate)

    try:
        from agent.prompt_builder import clear_skills_system_prompt_cache

        clear_skills_system_prompt_cache(clear_snapshot=True)
    except Exception:
        pass

    return {
        "success": True,
        "message": f"Candidate '{candidate_id}' promoted to trusted skill '{candidate.get('proposed_name', '')}'.",
        "candidate_id": candidate_id,
        "status": STATUS_PROMOTED,
        "promoted_name": candidate.get("proposed_name", ""),
    }


def reject_candidate(candidate_id: str, reason: str | None = None) -> Dict[str, Any]:
    try:
        candidate = _load_candidate(candidate_id)
    except FileNotFoundError as exc:
        return _error_result(str(exc))

    # Promoted candidates are terminal audit records; rejecting them later would
    # rewrite history without removing the already trusted skill.
    if candidate.get("status") == STATUS_PROMOTED:
        return _error_result(f"Candidate '{candidate_id}' has already been promoted and cannot be rejected.")

    candidate.setdefault("review", {})
    if reason:
        candidate["review"]["decision_reason"] = reason
    candidate["review"]["decision"] = DECISION_REJECT
    candidate["status"] = STATUS_REJECTED
    _save_candidate(candidate)
    return {
        "success": True,
        "message": f"Candidate '{candidate_id}' rejected.",
        "candidate_id": candidate_id,
        "status": STATUS_REJECTED,
    }
