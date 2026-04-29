"""Tests for tools/skill_candidate_tool.py."""

from contextlib import contextmanager
from unittest.mock import patch

import pytest

from tools.skill_candidate_tool import (
    STATUS_CANDIDATE,
    DECISION_PROMOTE_NEW,
    STATUS_PROMOTED,
    STATUS_REVIEWED,
    create_candidate,
    delete_candidate,
    promote_candidate,
    reject_candidate,
    update_candidate_review,
    view_candidate,
    write_candidate_file,
)


VALID_SKILL_CONTENT = """\
---
name: test-skill
description: Candidate test skill.
---

# Test Skill

1. Do the thing.
"""


@contextmanager
def _candidate_dirs(tmp_path):
    candidates_dir = tmp_path / "skill-candidates"
    skills_dir = tmp_path / "skills"
    with (
        patch("tools.skill_candidate_tool.CANDIDATES_DIR", candidates_dir),
        patch("tools.skill_manager_tool.SKILLS_DIR", skills_dir),
        patch("agent.skill_utils.get_all_skills_dirs", return_value=[skills_dir]),
    ):
        yield candidates_dir, skills_dir


def _mark_reviewed(candidate_id):
    return update_candidate_review(
        candidate_id,
        scores={
            "reusability": 2,
            "verification": 2,
            "non_triviality": 1,
            "scope_quality": 2,
            "actionability": 1,
        },
        total_score=8,
        threshold_used=7,
        decision=DECISION_PROMOTE_NEW,
        decision_reason="Looks reusable.",
        summary_for_user="Save this as a reusable pytest workflow.",
        status=STATUS_REVIEWED,
    )


def test_create_candidate_requires_scope(tmp_path):
    with _candidate_dirs(tmp_path):
        result = create_candidate(
            name="test-skill",
            content=VALID_SKILL_CONTENT,
            scope="",
            why_created="Useful workflow.",
            verification_summary="Ran pytest.",
        )

    assert result["success"] is False
    assert "scope is required" in result["error"]


def test_create_candidate_rejects_name_collision_with_existing_skill(tmp_path):
    with _candidate_dirs(tmp_path) as (candidates_dir, skills_dir):
        existing_dir = skills_dir / "test-skill"
        existing_dir.mkdir(parents=True, exist_ok=True)
        (existing_dir / "SKILL.md").write_text(VALID_SKILL_CONTENT, encoding="utf-8")

        result = create_candidate(
            name="test-skill",
            content=VALID_SKILL_CONTENT,
            scope="general",
            why_created="Useful workflow.",
            verification_summary="Validated manually.",
        )

    assert result["success"] is False
    assert "already exists at" in result["error"]
    assert not candidates_dir.exists() or not any(candidates_dir.iterdir())


def test_create_candidate_rejects_duplicate_live_candidate_name(tmp_path):
    with _candidate_dirs(tmp_path) as (candidates_dir, _):
        created = create_candidate(
            name="test-skill",
            content=VALID_SKILL_CONTENT,
            scope="general",
            why_created="Useful workflow.",
            verification_summary="Validated manually.",
        )
        assert created["success"] is True

        result = create_candidate(
            name="test-skill",
            content=VALID_SKILL_CONTENT,
            scope="general",
            why_created="Useful workflow.",
            verification_summary="Validated manually.",
        )
        candidate = view_candidate(created["candidate_id"])["candidate"]

    assert result["success"] is False
    assert "pending skill candidate named 'test-skill' already exists" in result["error"]
    candidate_dirs = [p for p in candidates_dir.iterdir() if p.is_dir()]
    assert len(candidate_dirs) == 1
    assert candidate["status"] == STATUS_CANDIDATE


def test_candidate_create_does_not_clear_cache_and_promotion_does(tmp_path):
    with _candidate_dirs(tmp_path) as (_, skills_dir), \
         patch("agent.prompt_builder.clear_skills_system_prompt_cache") as clear_cache:
        created = create_candidate(
            name="test-skill",
            content=VALID_SKILL_CONTENT,
            scope="repo-specific",
            why_created="Useful workflow.",
            verification_summary="Ran pytest tests/test_model_tools.py -q successfully.",
        )
        assert created["success"] is True
        candidate_id = created["candidate_id"]
        assert clear_cache.call_count == 0

        reviewed = update_candidate_review(
            candidate_id,
            scores={
                "reusability": 2,
                "verification": 2,
                "non_triviality": 1,
                "scope_quality": 2,
                "actionability": 1,
            },
                total_score=8,
                threshold_used=7,
                decision=DECISION_PROMOTE_NEW,
                decision_reason="Looks reusable.",
                summary_for_user="Save this as a reusable pytest workflow.",
                status=STATUS_REVIEWED,
            )
        assert reviewed["success"] is True
        assert clear_cache.call_count == 0

        promoted = promote_candidate(candidate_id)
        assert promoted["success"] is True
        assert clear_cache.call_count == 1
        assert (skills_dir / "test-skill" / "SKILL.md").exists()

        candidate = view_candidate(candidate_id)["candidate"]
        assert candidate["review"]["scores"]["verification"] == 2
        assert candidate["review"]["total_score"] == 8
        assert candidate["review"]["threshold_used"] == 7


def test_view_candidate_returns_skill_content(tmp_path):
    with _candidate_dirs(tmp_path):
        created = create_candidate(
            name="test-skill",
            content=VALID_SKILL_CONTENT,
            scope="general",
            why_created="Useful workflow.",
            verification_summary="Validated manually.",
        )
        candidate_id = created["candidate_id"]

        result = view_candidate(candidate_id)

    assert result["success"] is True
    assert result["candidate"]["skill_content"] == VALID_SKILL_CONTENT


def test_write_candidate_file_stages_supporting_file(tmp_path):
    with _candidate_dirs(tmp_path) as (candidates_dir, _):
        created = create_candidate(
            name="test-skill",
            content=VALID_SKILL_CONTENT,
            scope="general",
            why_created="Useful workflow.",
            verification_summary="Validated manually.",
        )

        result = write_candidate_file(
            created["candidate_id"],
            "references/notes.md",
            "supporting notes",
        )
        candidate = view_candidate(created["candidate_id"])["candidate"]

    assert result["success"] is True
    assert (candidates_dir / created["candidate_id"] / "references" / "notes.md").read_text(
        encoding="utf-8"
    ) == "supporting notes"
    assert candidate["supporting_files"] == ["references/notes.md"]


@pytest.mark.parametrize("file_path", ["SKILL.md", "docs/notes.md", "../escape.md"])
def test_write_candidate_file_rejects_unsupported_paths(tmp_path, file_path):
    with _candidate_dirs(tmp_path):
        created = create_candidate(
            name="test-skill",
            content=VALID_SKILL_CONTENT,
            scope="general",
            why_created="Useful workflow.",
            verification_summary="Validated manually.",
        )

        result = write_candidate_file(created["candidate_id"], file_path, "content")

    assert result["success"] is False


def test_promote_candidate_copies_staged_supporting_files_only(tmp_path):
    with _candidate_dirs(tmp_path) as (candidates_dir, skills_dir):
        created = create_candidate(
            name="test-skill",
            content=VALID_SKILL_CONTENT,
            scope="general",
            why_created="Useful workflow.",
            verification_summary="Validated manually.",
        )
        candidate_id = created["candidate_id"]
        assert write_candidate_file(candidate_id, "references/notes.md", "notes")["success"] is True
        unknown_dir = candidates_dir / candidate_id / "scratch"
        unknown_dir.mkdir(parents=True)
        (unknown_dir / "internal.txt").write_text("do not copy", encoding="utf-8")
        assert _mark_reviewed(candidate_id)["success"] is True

        promoted = promote_candidate(candidate_id)

    assert promoted["success"] is True
    assert (skills_dir / "test-skill" / "SKILL.md").exists()
    assert (skills_dir / "test-skill" / "references" / "notes.md").read_text(
        encoding="utf-8"
    ) == "notes"
    assert not (skills_dir / "test-skill" / "candidate.json").exists()
    assert not (skills_dir / "test-skill" / "scratch").exists()


def test_promote_candidate_rolls_back_supporting_files_on_scan_failure(tmp_path):
    with _candidate_dirs(tmp_path) as (_, skills_dir):
        created = create_candidate(
            name="test-skill",
            content=VALID_SKILL_CONTENT,
            scope="general",
            why_created="Useful workflow.",
            verification_summary="Validated manually.",
        )
        candidate_id = created["candidate_id"]
        assert write_candidate_file(candidate_id, "scripts/run.sh", "echo bad")["success"] is True
        assert _mark_reviewed(candidate_id)["success"] is True

        with patch("tools.skill_candidate_tool._security_scan_skill", return_value="blocked"):
            result = promote_candidate(candidate_id)
        candidate = view_candidate(candidate_id)["candidate"]

    assert result["success"] is False
    assert "blocked" in result["error"]
    assert candidate["status"] == STATUS_REVIEWED
    assert not (skills_dir / "test-skill").exists()


def test_reject_candidate_does_not_rewrite_promoted_candidate(tmp_path):
    with _candidate_dirs(tmp_path):
        created = create_candidate(
            name="test-skill",
            content=VALID_SKILL_CONTENT,
            scope="general",
            why_created="Useful workflow.",
            verification_summary="Validated manually.",
        )
        candidate_id = created["candidate_id"]

        reviewed = update_candidate_review(
            candidate_id,
            scores={
                "reusability": 2,
                "verification": 2,
                "non_triviality": 1,
                "scope_quality": 2,
                "actionability": 1,
            },
            total_score=8,
            threshold_used=7,
            decision=DECISION_PROMOTE_NEW,
            decision_reason="Looks reusable.",
            summary_for_user="Save this as a reusable pytest workflow.",
            status=STATUS_REVIEWED,
        )
        assert reviewed["success"] is True
        assert promote_candidate(candidate_id)["success"] is True

        rejected = reject_candidate(candidate_id, reason="Changed my mind.")
        candidate = view_candidate(candidate_id)["candidate"]

    assert rejected["success"] is False
    assert "already been promoted" in rejected["error"]
    assert candidate["status"] == STATUS_PROMOTED
    assert candidate["review"]["decision"] == DECISION_PROMOTE_NEW
    assert candidate["review"]["decision_reason"] == "Looks reusable."


@pytest.mark.parametrize("candidate_id", ["../skills/test-skill", "/tmp/test-skill", "not-a-candidate"])
@pytest.mark.parametrize(
    "operation",
    [
        view_candidate,
        promote_candidate,
        reject_candidate,
        delete_candidate,
    ],
)
def test_public_candidate_helpers_reject_invalid_candidate_ids(tmp_path, candidate_id, operation):
    with _candidate_dirs(tmp_path) as (candidates_dir, _):
        outside_dir = tmp_path / "skills" / "test-skill"
        outside_dir.mkdir(parents=True)
        (outside_dir / "candidate.json").write_text("{}", encoding="utf-8")

        result = operation(candidate_id)

    assert result["success"] is False
    assert "not found" in result["error"]
    assert outside_dir.exists()
    assert not candidates_dir.exists() or not any(candidates_dir.iterdir())


def test_update_candidate_review_rejects_invalid_candidate_id(tmp_path):
    with _candidate_dirs(tmp_path) as (candidates_dir, _):
        result = update_candidate_review(
            "../skills/test-skill",
            scores={},
            total_score=0,
            threshold_used=0,
            decision=DECISION_PROMOTE_NEW,
            decision_reason="",
            summary_for_user="",
        )

    assert result["success"] is False
    assert "not found" in result["error"]
    assert not candidates_dir.exists() or not any(candidates_dir.iterdir())


def test_promote_candidate_rolls_back_approved_state_on_create_exception(tmp_path):
    with _candidate_dirs(tmp_path):
        created = create_candidate(
            name="test-skill",
            content=VALID_SKILL_CONTENT,
            scope="general",
            why_created="Useful workflow.",
            verification_summary="Validated manually.",
        )
        candidate_id = created["candidate_id"]

        reviewed = update_candidate_review(
            candidate_id,
            scores={
                "reusability": 2,
                "verification": 2,
                "non_triviality": 1,
                "scope_quality": 2,
                "actionability": 1,
            },
            total_score=8,
            threshold_used=7,
            decision=DECISION_PROMOTE_NEW,
            decision_reason="Looks reusable.",
            summary_for_user="Save this as a reusable pytest workflow.",
            status=STATUS_REVIEWED,
        )
        assert reviewed["success"] is True

        with (
            patch("tools.skill_candidate_tool._create_skill", side_effect=OSError("disk full")),
            pytest.raises(OSError, match="disk full"),
        ):
            promote_candidate(candidate_id)

        candidate = view_candidate(candidate_id)["candidate"]

    assert candidate["status"] == STATUS_REVIEWED
