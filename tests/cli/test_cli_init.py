"""Tests for HermesCLI initialization -- catches configuration bugs
that only manifest at runtime (not in mocked unit tests)."""

import os
import sys
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _make_cli(env_overrides=None, config_overrides=None, **kwargs):
    """Create a HermesCLI instance with minimal mocking."""
    import importlib

    _clean_config = {
        "model": {
            "default": "anthropic/claude-opus-4.6",
            "base_url": "https://openrouter.ai/api/v1",
            "provider": "auto",
        },
        "display": {"compact": False, "tool_progress": "all"},
        "agent": {},
        "terminal": {"env_type": "local"},
    }
    if config_overrides:
        _clean_config.update(config_overrides)
    clean_env = {"LLM_MODEL": "", "HERMES_MAX_ITERATIONS": ""}
    if env_overrides:
        clean_env.update(env_overrides)
    prompt_toolkit_stubs = {
        "prompt_toolkit": MagicMock(),
        "prompt_toolkit.history": MagicMock(),
        "prompt_toolkit.styles": MagicMock(),
        "prompt_toolkit.patch_stdout": MagicMock(),
        "prompt_toolkit.application": MagicMock(),
        "prompt_toolkit.layout": MagicMock(),
        "prompt_toolkit.layout.processors": MagicMock(),
        "prompt_toolkit.filters": MagicMock(),
        "prompt_toolkit.layout.dimension": MagicMock(),
        "prompt_toolkit.layout.menus": MagicMock(),
        "prompt_toolkit.widgets": MagicMock(),
        "prompt_toolkit.key_binding": MagicMock(),
        "prompt_toolkit.completion": MagicMock(),
        "prompt_toolkit.formatted_text": MagicMock(),
        "prompt_toolkit.auto_suggest": MagicMock(),
    }
    with patch.dict(sys.modules, prompt_toolkit_stubs), \
         patch.dict("os.environ", clean_env, clear=False):
        import cli as _cli_mod
        _cli_mod = importlib.reload(_cli_mod)
        with patch.object(_cli_mod, "get_tool_definitions", return_value=[]), \
             patch.dict(_cli_mod.__dict__, {"CLI_CONFIG": _clean_config}):
            return _cli_mod.HermesCLI(**kwargs)


class TestMaxTurnsResolution:
    """max_turns must always resolve to a positive integer, never None."""

    def test_default_max_turns_is_integer(self):
        cli = _make_cli()
        assert isinstance(cli.max_turns, int)
        assert cli.max_turns == 90

    def test_explicit_max_turns_honored(self):
        cli = _make_cli(max_turns=25)
        assert cli.max_turns == 25

    def test_none_max_turns_gets_default(self):
        cli = _make_cli(max_turns=None)
        assert isinstance(cli.max_turns, int)
        assert cli.max_turns == 90

    def test_env_var_max_turns(self):
        """Env var is used when config file doesn't set max_turns."""
        cli_obj = _make_cli(env_overrides={"HERMES_MAX_ITERATIONS": "42"})
        assert cli_obj.max_turns == 42

    def test_legacy_root_max_turns_is_used_when_agent_key_exists_without_value(self):
        cli_obj = _make_cli(config_overrides={"agent": {}, "max_turns": 77})
        assert cli_obj.max_turns == 77

    def test_max_turns_never_none_for_agent(self):
        """The value passed to AIAgent must never be None (causes TypeError in run_conversation)."""
        cli = _make_cli()
        assert isinstance(cli.max_turns, int) and cli.max_turns == 90


class TestVerboseAndToolProgress:
    def test_default_verbose_is_bool(self):
        cli = _make_cli()
        assert isinstance(cli.verbose, bool)

    def test_tool_progress_mode_is_string(self):
        cli = _make_cli()
        assert isinstance(cli.tool_progress_mode, str)
        assert cli.tool_progress_mode in ("off", "new", "all", "verbose")


class TestBusyInputMode:
    def test_default_busy_input_mode_is_interrupt(self):
        cli = _make_cli()
        assert cli.busy_input_mode == "interrupt"

    def test_busy_input_mode_queue_is_honored(self):
        cli = _make_cli(config_overrides={"display": {"busy_input_mode": "queue"}})
        assert cli.busy_input_mode == "queue"

    def test_unknown_busy_input_mode_falls_back_to_interrupt(self):
        cli = _make_cli(config_overrides={"display": {"busy_input_mode": "bogus"}})
        assert cli.busy_input_mode == "interrupt"

    def test_queue_command_works_while_busy(self):
        """When agent is running, /queue should still put the prompt in _pending_input."""
        cli = _make_cli()
        cli._agent_running = True
        cli.process_command("/queue follow up")
        assert cli._pending_input.get_nowait() == "follow up"

    def test_queue_command_works_while_idle(self):
        """When agent is idle, /queue should still queue (not reject)."""
        cli = _make_cli()
        cli._agent_running = False
        cli.process_command("/queue follow up")
        assert cli._pending_input.get_nowait() == "follow up"

    def test_queue_mode_routes_busy_enter_to_pending(self):
        """In queue mode, Enter while busy should go to _pending_input, not _interrupt_queue."""
        cli = _make_cli(config_overrides={"display": {"busy_input_mode": "queue"}})
        cli._agent_running = True
        # Simulate what handle_enter does for non-command input while busy
        text = "follow up"
        if cli.busy_input_mode == "queue":
            cli._pending_input.put(text)
        else:
            cli._interrupt_queue.put(text)
        assert cli._pending_input.get_nowait() == "follow up"
        assert cli._interrupt_queue.empty()

    def test_interrupt_mode_routes_busy_enter_to_interrupt(self):
        """In interrupt mode (default), Enter while busy goes to _interrupt_queue."""
        cli = _make_cli()
        cli._agent_running = True
        text = "redirect"
        if cli.busy_input_mode == "queue":
            cli._pending_input.put(text)
        else:
            cli._interrupt_queue.put(text)
        assert cli._interrupt_queue.get_nowait() == "redirect"
        assert cli._pending_input.empty()


class TestSingleQueryState:
    def test_voice_and_interrupt_state_initialized_before_run(self):
        """Single-query mode calls chat() without going through run()."""
        cli = _make_cli()
        assert cli._voice_tts is False
        assert cli._voice_mode is False
        assert cli._voice_tts_done.is_set()
        assert hasattr(cli, "_interrupt_queue")
        assert hasattr(cli, "_pending_input")


class TestStrictSkillReviewQueue:
    VALID_SKILL_CONTENT = """\
---
name: demo-skill
description: Demo skill.
---

# Demo

1. Do the thing.
"""

    def _candidate_dirs(self, tmp_path):
        candidates_dir = tmp_path / "skill-candidates"
        skills_dir = tmp_path / "skills"
        return candidates_dir, skills_dir

    def test_poll_pending_skill_review_candidates_enqueues_reviewed_candidate(self, tmp_path):
        from tools.skill_candidate_tool import (
            DECISION_PROMOTE_NEW,
            STATUS_REVIEWED,
            create_candidate,
            update_candidate_review,
        )

        cli = _make_cli(config_overrides={"skills": {"strict_creation_mode": True}})
        candidates_dir, skills_dir = self._candidate_dirs(tmp_path)
        with (
            patch("tools.skill_candidate_tool.CANDIDATES_DIR", candidates_dir),
            patch("tools.skill_manager_tool.SKILLS_DIR", skills_dir),
            patch("agent.skill_utils.get_all_skills_dirs", return_value=[skills_dir]),
        ):
            created = create_candidate(
                name="demo-skill",
                content=self.VALID_SKILL_CONTENT,
                scope="repo-specific",
                why_created="Useful workflow.",
                verification_summary="Ran targeted tests successfully.",
            )
            update_candidate_review(
                created["candidate_id"],
                scores={
                    "reusability": 2,
                    "verification": 1,
                    "non_triviality": 1,
                    "scope_quality": 1,
                    "actionability": 1,
                },
                total_score=6,
                threshold_used=4,
                decision=DECISION_PROMOTE_NEW,
                decision_reason="Looks good.",
                summary_for_user="Save this as a reusable workflow.",
                status=STATUS_REVIEWED,
            )

            cli._poll_pending_skill_review_candidates()

        assert created["candidate_id"] in cli._pending_skill_review_ids

    def test_skillreviews_view_shows_staged_skill_content(self, tmp_path):
        import cli as cli_mod
        from tools.skill_candidate_tool import create_candidate, write_candidate_file

        cli = _make_cli(config_overrides={"skills": {"strict_creation_mode": True}})
        candidates_dir, skills_dir = self._candidate_dirs(tmp_path)
        with (
            patch("tools.skill_candidate_tool.CANDIDATES_DIR", candidates_dir),
            patch("tools.skill_manager_tool.SKILLS_DIR", skills_dir),
            patch("agent.skill_utils.get_all_skills_dirs", return_value=[skills_dir]),
        ):
            created = create_candidate(
                name="demo-skill",
                content=self.VALID_SKILL_CONTENT,
                scope="repo-specific",
                why_created="Useful workflow.",
                verification_summary="Ran targeted tests successfully.",
            )
            write_candidate_file(created["candidate_id"], "references/notes.md", "notes")

            printed = []
            with patch.object(cli_mod, "_cprint", side_effect=printed.append):
                cli._handle_skillreviews_command(f"/skillreviews view {created['candidate_id']}")

        output = "\n".join(str(line) for line in printed)
        assert "Skill proposal: demo-skill" in output
        assert f"Candidate ID: {created['candidate_id']}" in output
        assert str(candidates_dir / created["candidate_id"]) in output
        assert "Staged supporting files:" in output
        assert "references/notes.md" in output
        assert "--- SKILL.md ---" in output
        assert "# Demo" in output
        assert "1. Do the thing." in output

    def test_skillreviews_view_truncates_long_staged_skill_content(self, tmp_path):
        import cli as cli_mod
        from tools.skill_candidate_tool import create_candidate

        cli = _make_cli(config_overrides={"skills": {"strict_creation_mode": True}})
        candidates_dir, skills_dir = self._candidate_dirs(tmp_path)
        long_content = (
            "---\nname: demo-skill\ndescription: Demo skill.\n---\n\n"
            + "\n".join(f"line {idx}" for idx in range(1, 130))
        )
        with (
            patch("tools.skill_candidate_tool.CANDIDATES_DIR", candidates_dir),
            patch("tools.skill_manager_tool.SKILLS_DIR", skills_dir),
            patch("agent.skill_utils.get_all_skills_dirs", return_value=[skills_dir]),
        ):
            created = create_candidate(
                name="demo-skill",
                content=long_content,
                scope="repo-specific",
                why_created="Useful workflow.",
                verification_summary="Ran targeted tests successfully.",
            )

            printed = []
            with patch.object(cli_mod, "_cprint", side_effect=printed.append):
                cli._handle_skillreviews_command(f"/skillreviews view {created['candidate_id']}")

        output = "\n".join(str(line) for line in printed)
        assert "line 95" in output
        assert "line 101" not in output
        assert "truncated after 100 lines" in output
        assert str(candidates_dir / created["candidate_id"] / "SKILL.md") in output

    def test_skillreviews_approve_promotes_candidate(self, tmp_path):
        from tools.skill_candidate_tool import (
            DECISION_PROMOTE_NEW,
            STATUS_REVIEWED,
            create_candidate,
            update_candidate_review,
        )

        cli = _make_cli(config_overrides={"skills": {"strict_creation_mode": True}})
        candidates_dir, skills_dir = self._candidate_dirs(tmp_path)
        with (
            patch("tools.skill_candidate_tool.CANDIDATES_DIR", candidates_dir),
            patch("tools.skill_manager_tool.SKILLS_DIR", skills_dir),
            patch("agent.skill_utils.get_all_skills_dirs", return_value=[skills_dir]),
            patch("agent.prompt_builder.clear_skills_system_prompt_cache"),
        ):
            created = create_candidate(
                name="demo-skill",
                content=self.VALID_SKILL_CONTENT,
                scope="repo-specific",
                why_created="Useful workflow.",
                verification_summary="Ran targeted tests successfully.",
            )
            update_candidate_review(
                created["candidate_id"],
                scores={
                    "reusability": 2,
                    "verification": 1,
                    "non_triviality": 1,
                    "scope_quality": 1,
                    "actionability": 1,
                },
                total_score=6,
                threshold_used=4,
                decision=DECISION_PROMOTE_NEW,
                decision_reason="Looks good.",
                summary_for_user="Save this as a reusable workflow.",
                status=STATUS_REVIEWED,
            )

            cli._handle_skillreviews_command(f"/skillreviews approve {created['candidate_id']}")

        assert (skills_dir / "demo-skill" / "SKILL.md").exists()

    def test_skillreviews_approve_refreshes_skill_command_cache(self, tmp_path):
        import cli as cli_mod
        from tools.skill_candidate_tool import (
            DECISION_PROMOTE_NEW,
            STATUS_REVIEWED,
            create_candidate,
            update_candidate_review,
        )

        cli = _make_cli(config_overrides={"skills": {"strict_creation_mode": True}})
        candidates_dir, skills_dir = self._candidate_dirs(tmp_path)
        with (
            patch("tools.skill_candidate_tool.CANDIDATES_DIR", candidates_dir),
            patch("tools.skill_manager_tool.SKILLS_DIR", skills_dir),
            patch("agent.skill_utils.get_all_skills_dirs", return_value=[skills_dir]),
            patch("agent.prompt_builder.clear_skills_system_prompt_cache"),
        ):
            created = create_candidate(
                name="demo-skill",
                content=self.VALID_SKILL_CONTENT,
                scope="repo-specific",
                why_created="Useful workflow.",
                verification_summary="Ran targeted tests successfully.",
            )
            update_candidate_review(
                created["candidate_id"],
                scores={
                    "reusability": 2,
                    "verification": 1,
                    "non_triviality": 1,
                    "scope_quality": 1,
                    "actionability": 1,
                },
                total_score=6,
                threshold_used=4,
                decision=DECISION_PROMOTE_NEW,
                decision_reason="Looks good.",
                summary_for_user="Save this as a reusable workflow.",
                status=STATUS_REVIEWED,
            )

            cli_mod._skill_commands = {}
            refreshed = {"/demo-skill": {"name": "demo-skill"}}
            with patch.object(cli_mod, "scan_skill_commands", return_value=refreshed) as mock_scan:
                cli._handle_skillreviews_command(f"/skillreviews approve {created['candidate_id']}")

        mock_scan.assert_called_once()
        assert cli_mod._skill_commands == refreshed

    def test_skillreviews_approve_reports_promotion_exception(self):
        cli = _make_cli(config_overrides={"skills": {"strict_creation_mode": True}})

        printed = []
        globals_to_patch = cli._handle_skillreviews_command.__globals__
        mock_scan = MagicMock()
        with (
            patch("tools.skill_candidate_tool.view_candidate", return_value={"success": True}),
            patch("tools.skill_candidate_tool.promote_candidate", side_effect=OSError("disk full")),
            patch.dict(globals_to_patch, {"_cprint": printed.append, "scan_skill_commands": mock_scan}),
        ):
            cli._handle_skillreviews_command("/skillreviews approve cand-123")

        mock_scan.assert_not_called()
        assert "Could not promote candidate: disk full" in "\n".join(
            str(line) for line in printed
        )


class TestHistoryDisplay:
    def test_history_numbers_only_visible_messages_and_summarizes_tools(self, capsys):
        cli = _make_cli()
        cli.conversation_history = [
            {"role": "system", "content": "system prompt"},
            {"role": "user", "content": "Hello"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"id": "call_1"}, {"id": "call_2"}],
            },
            {"role": "tool", "content": "tool output 1"},
            {"role": "tool", "content": "tool output 2"},
            {"role": "assistant", "content": "All set."},
            {"role": "user", "content": "A" * 250},
        ]

        cli.show_history()
        output = capsys.readouterr().out

        assert "[You #1]" in output
        assert "[Hermes #2]" in output
        assert "(requested 2 tool calls)" in output
        assert "[Tools]" in output
        assert "(2 tool messages hidden)" in output
        assert "[Hermes #3]" in output
        assert "[You #4]" in output
        assert "[You #5]" not in output
        assert "A" * 250 in output
        assert "A" * 250 + "..." not in output

    def test_history_shows_recent_sessions_when_current_chat_is_empty(self, capsys):
        cli = _make_cli()
        cli.session_id = "current"
        cli._session_db = MagicMock()
        cli._session_db.list_sessions_rich.return_value = [
            {
                "id": "current",
                "title": "Current",
                "preview": "Current preview",
                "last_active": 0,
            },
            {
                "id": "20260401_201329_d85961",
                "title": "Checking Running Hermes Agent",
                "preview": "check running gateways for hermes agent",
                "last_active": 0,
            },
        ]

        cli.show_history()
        output = capsys.readouterr().out

        assert "No messages in the current chat yet" in output
        assert "Checking Running Hermes Agent" in output
        assert "20260401_201329_d85961" in output
        assert "/resume" in output
        assert "Current preview" not in output

    def test_resume_without_target_lists_recent_sessions(self, capsys):
        cli = _make_cli()
        cli.session_id = "current"
        cli._session_db = MagicMock()
        cli._session_db.list_sessions_rich.return_value = [
            {
                "id": "current",
                "title": "Current",
                "preview": "Current preview",
                "last_active": 0,
            },
            {
                "id": "20260401_201329_d85961",
                "title": "Checking Running Hermes Agent",
                "preview": "check running gateways for hermes agent",
                "last_active": 0,
            },
        ]

        cli._handle_resume_command("/resume")
        output = capsys.readouterr().out

        assert "Recent sessions" in output
        assert "Checking Running Hermes Agent" in output
        assert "Use /resume <session id or title> to continue" in output


class TestRootLevelProviderOverride:
    """Root-level provider/base_url in config.yaml must NOT override model.provider."""

    def test_model_provider_wins_over_root_provider(self, tmp_path, monkeypatch):
        """model.provider takes priority — root-level provider is only a fallback."""
        import yaml

        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))

        config_path = hermes_home / "config.yaml"
        config_path.write_text(yaml.safe_dump({
            "provider": "opencode-go",  # stale root-level key
            "model": {
                "default": "google/gemini-3-flash-preview",
                "provider": "openrouter",  # correct canonical key
            },
        }))

        import cli
        monkeypatch.setattr(cli, "_hermes_home", hermes_home)
        cfg = cli.load_cli_config()

        assert cfg["model"]["provider"] == "openrouter"

    def test_root_provider_ignored_when_default_model_provider_exists(self, tmp_path, monkeypatch):
        """Even when model.provider is the default 'auto', root-level provider is ignored."""
        import yaml

        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))

        config_path = hermes_home / "config.yaml"
        config_path.write_text(yaml.safe_dump({
            "provider": "opencode-go",  # stale root key
            "model": {
                "default": "google/gemini-3-flash-preview",
                # no explicit model.provider — defaults provide "auto"
            },
        }))

        import cli
        monkeypatch.setattr(cli, "_hermes_home", hermes_home)
        cfg = cli.load_cli_config()

        # Root-level "opencode-go" must NOT leak through
        assert cfg["model"]["provider"] != "opencode-go"

    def test_normalize_root_model_keys_moves_to_model(self):
        """_normalize_root_model_keys migrates root keys into model section."""
        from hermes_cli.config import _normalize_root_model_keys

        config = {
            "provider": "opencode-go",
            "base_url": "https://example.com/v1",
            "model": {
                "default": "some-model",
            },
        }
        result = _normalize_root_model_keys(config)
        # Root keys removed
        assert "provider" not in result
        assert "base_url" not in result
        # Migrated into model section
        assert result["model"]["provider"] == "opencode-go"
        assert result["model"]["base_url"] == "https://example.com/v1"

    def test_normalize_root_model_keys_does_not_override_existing(self):
        """Existing model.provider is never overridden by root-level key."""
        from hermes_cli.config import _normalize_root_model_keys

        config = {
            "provider": "stale-provider",
            "model": {
                "default": "some-model",
                "provider": "correct-provider",
            },
        }
        result = _normalize_root_model_keys(config)
        assert result["model"]["provider"] == "correct-provider"
        assert "provider" not in result  # root key still cleaned up


class TestProviderResolution:
    def test_api_key_is_string_or_none(self):
        cli = _make_cli()
        assert cli.api_key is None or isinstance(cli.api_key, str)

    def test_base_url_is_string(self):
        cli = _make_cli()
        assert isinstance(cli.base_url, str)
        assert cli.base_url.startswith("http")

    def test_model_is_string(self):
        cli = _make_cli()
        assert isinstance(cli.model, str)
        assert isinstance(cli.model, str) and '/' in cli.model
