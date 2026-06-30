"""Coverage tests for agents/onboarding.py — missing lines."""

from __future__ import annotations

import shutil
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml


def _setup(tmp_path, monkeypatch):
    import onemancompany.core.config as config_mod
    import onemancompany.agents.onboarding as ob_mod
    monkeypatch.setattr(config_mod, "EMPLOYEES_DIR", tmp_path / "employees")
    monkeypatch.setattr(ob_mod, "EMPLOYEES_DIR", tmp_path / "employees")
    (tmp_path / "employees").mkdir(parents=True, exist_ok=True)
    return tmp_path


# ---------------------------------------------------------------------------
# _load_nickname_pool — file not found (line 115-116)
# ---------------------------------------------------------------------------

class TestLoadNicknamePool:
    def test_pool_file_not_found(self, tmp_path, monkeypatch):
        import onemancompany.agents.onboarding as ob_mod
        import onemancompany.core.config as config_mod
        monkeypatch.setattr(ob_mod, "_NICKNAMES_FILE", tmp_path / "nonexistent.txt")
        monkeypatch.setattr(config_mod, "DATA_ROOT", tmp_path / "no_data")
        pool = ob_mod._load_nickname_pool()
        assert pool == []

    def test_pool_file_exists(self, tmp_path, monkeypatch):
        import onemancompany.agents.onboarding as ob_mod
        import onemancompany.core.config as config_mod
        f = tmp_path / "nicknames.txt"
        f.write_text("风云\n雷电\n\n")
        monkeypatch.setattr(ob_mod, "_NICKNAMES_FILE", f)
        monkeypatch.setattr(config_mod, "DATA_ROOT", tmp_path / "no_data")
        pool = ob_mod._load_nickname_pool()
        assert "风云" in pool
        assert "雷电" in pool

    def test_pick_nickname_exhausted(self, monkeypatch):
        """Cover line 131: all pool candidates collide."""
        import onemancompany.agents.onboarding as ob_mod
        monkeypatch.setattr(ob_mod, "_load_nickname_pool", lambda: ["AB"])
        result = ob_mod._pick_nickname(2, {"AB"})
        # Should generate from random wuxia chars or return empty
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# generate_nickname — no nickname found (line 150)
# ---------------------------------------------------------------------------

class TestGenerateNickname:
    @pytest.mark.asyncio
    async def test_no_unique_nickname(self, monkeypatch):
        import onemancompany.agents.onboarding as ob_mod
        monkeypatch.setattr(ob_mod, "_pick_nickname", lambda *a, **kw: "")
        monkeypatch.setattr(ob_mod, "_get_existing_nicknames", lambda: set())
        result = await ob_mod.generate_nickname("Test", "Engineer")
        assert result == ""


# ---------------------------------------------------------------------------
# install_talent_vessel_config (lines 505, 511-532)
# ---------------------------------------------------------------------------

class TestInstallVesselConfig:
    def test_already_installed(self, tmp_path):
        from onemancompany.agents.onboarding import install_talent_vessel_config
        emp_dir = tmp_path / "00010"
        vessel_dir = emp_dir / "vessel"
        vessel_dir.mkdir(parents=True)
        (vessel_dir / "vessel.yaml").write_text("already: true")
        install_talent_vessel_config(tmp_path / "talent", str(emp_dir), "00010")
        # Should be a no-op
        assert yaml.safe_load((vessel_dir / "vessel.yaml").read_text()) == {"already": True}

    def test_install_from_talent_vessel(self, tmp_path):
        """Cover lines 511-532: copy from talent vessel dir."""
        from onemancompany.agents.onboarding import install_talent_vessel_config
        emp_dir = tmp_path / "00010"
        emp_dir.mkdir(parents=True)
        talent_dir = tmp_path / "talent"
        talent_vessel = talent_dir / "vessel"
        talent_vessel.mkdir(parents=True)
        (talent_vessel / "vessel.yaml").write_text("runner:\n  module: my_runner\nhooks:\n  module: my_hooks\n")
        # Create prompt_sections
        ps = talent_vessel / "prompt_sections"
        ps.mkdir()
        (ps / "custom.md").write_text("content")
        # Create runner module
        (talent_vessel / "my_runner.py").write_text("# runner")
        # Create hooks module
        (talent_vessel / "my_hooks.py").write_text("# hooks")
        install_talent_vessel_config(talent_dir, str(emp_dir), "00010")
        assert (emp_dir / "vessel" / "vessel.yaml").exists()
        assert (emp_dir / "vessel" / "prompt_sections" / "custom.md").exists()
        assert (emp_dir / "vessel" / "my_runner.py").exists()
        assert (emp_dir / "vessel" / "my_hooks.py").exists()

    def test_install_default(self, tmp_path):
        """Cover line 536: fallback to default config."""
        from onemancompany.agents.onboarding import install_talent_vessel_config
        emp_dir = tmp_path / "00010"
        emp_dir.mkdir(parents=True)
        talent_dir = tmp_path / "talent"
        talent_dir.mkdir()
        # No vessel dir in talent — falls back to default
        with patch("onemancompany.core.vessel_config._load_default_vessel_config") as mock_default, \
             patch("onemancompany.core.vessel_config.save_vessel_config"):
            mock_default.return_value = MagicMock()
            install_talent_vessel_config(talent_dir, str(emp_dir), "00010")


# ---------------------------------------------------------------------------
# resolve_talent_dir (lines 558, 591, 597)
# ---------------------------------------------------------------------------

class TestResolveTalentDir:
    def test_empty_talent_id(self):
        from onemancompany.agents.onboarding import resolve_talent_dir
        assert resolve_talent_dir("") is None

    def test_resolve_builtin(self, tmp_path, monkeypatch):
        import onemancompany.agents.onboarding as ob_mod
        builtin = tmp_path / "builtin"
        (builtin / "my_talent").mkdir(parents=True)
        monkeypatch.setattr(ob_mod, "_TALENTS_CLONE_DIR", tmp_path / "clone")
        monkeypatch.setattr(ob_mod, "_BUILTIN_TALENTS_DIR", builtin)
        result = ob_mod.resolve_talent_dir("my_talent")
        assert result == builtin / "my_talent"


# ---------------------------------------------------------------------------
# clone_talent_repo (lines 591-619)
# ---------------------------------------------------------------------------

class TestCloneTalentRepo:
    @pytest.mark.asyncio
    async def test_clone_single_talent(self, tmp_path, monkeypatch):
        import onemancompany.agents.onboarding as ob_mod
        clone_dir = tmp_path / "clone"
        monkeypatch.setattr(ob_mod, "_TALENTS_CLONE_DIR", clone_dir)

        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.communicate = AsyncMock(return_value=(b"", b""))

        # Create the temp clone dir with a profile.yaml
        created_dirs = []
        real_mkdtemp = __import__("tempfile").mkdtemp

        def patched_mkdtemp(**kwargs):
            d = real_mkdtemp(**kwargs)
            created_dirs.append(d)
            Path(d).mkdir(parents=True, exist_ok=True)
            (Path(d) / "profile.yaml").write_text("id: my_talent\n")
            return d

        with patch("asyncio.create_subprocess_exec", new_callable=AsyncMock, return_value=mock_proc), \
             patch("tempfile.mkdtemp", side_effect=patched_mkdtemp):
            result = await ob_mod.clone_talent_repo("https://github.com/test/repo", "my_talent")
        assert result == clone_dir / "my_talent"

    @pytest.mark.asyncio
    async def test_clone_multi_talent(self, tmp_path, monkeypatch):
        import onemancompany.agents.onboarding as ob_mod
        clone_dir = tmp_path / "clone"
        monkeypatch.setattr(ob_mod, "_TALENTS_CLONE_DIR", clone_dir)

        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.communicate = AsyncMock(return_value=(b"", b""))

        created_dirs = []
        real_mkdtemp = __import__("tempfile").mkdtemp

        def patched_mkdtemp(**kwargs):
            d = real_mkdtemp(**kwargs)
            created_dirs.append(d)
            p = Path(d)
            p.mkdir(parents=True, exist_ok=True)
            # Multi-talent: no root profile.yaml, but sub dirs have it
            sub = p / "sub_talent"
            sub.mkdir()
            (sub / "profile.yaml").write_text("id: sub_talent\n")
            return d

        with patch("asyncio.create_subprocess_exec", new_callable=AsyncMock, return_value=mock_proc), \
             patch("tempfile.mkdtemp", side_effect=patched_mkdtemp):
            result = await ob_mod.clone_talent_repo("https://github.com/test/repo", "sub_talent")
        assert (clone_dir / "sub_talent").exists()

    @pytest.mark.asyncio
    async def test_clone_git_failure(self, tmp_path, monkeypatch):
        import subprocess
        import onemancompany.agents.onboarding as ob_mod
        clone_dir = tmp_path / "clone"
        monkeypatch.setattr(ob_mod, "_TALENTS_CLONE_DIR", clone_dir)

        mock_proc = MagicMock()
        mock_proc.returncode = 1
        mock_proc.communicate = AsyncMock(return_value=(b"", b"error"))

        with patch("asyncio.create_subprocess_exec", new_callable=AsyncMock, return_value=mock_proc):
            with pytest.raises(subprocess.CalledProcessError):
                await ob_mod.clone_talent_repo("https://bad/repo", "talent")


# ---------------------------------------------------------------------------
# _inject_default_skills — EA skills + sync logic (lines 637-665)
# ---------------------------------------------------------------------------

class TestInjectDefaultSkills:
    def test_inject_with_ea_skills(self, tmp_path, monkeypatch):
        import onemancompany.agents.onboarding as ob_mod
        monkeypatch.setattr(ob_mod, "_DEFAULT_SKILLS_DIR", tmp_path / "default_skills")

        # Create task_lifecycle (default) and project-brainstorming (EA-only)
        for skill_name in ("task_lifecycle", "project-brainstorming"):
            src_dir = tmp_path / "default_skills" / skill_name
            src_dir.mkdir(parents=True)
            (src_dir / "SKILL.md").write_text(f"---\nname: {skill_name}\n---\nContent")

        skills_dir = tmp_path / "00010" / "skills"
        skills_dir.mkdir(parents=True)

        with patch("onemancompany.core.config.EA_ID", "00010"):
            ob_mod._inject_default_skills(skills_dir, employee_id="00010")
        assert (skills_dir / "task_lifecycle" / "SKILL.md").exists()
        assert (skills_dir / "project-brainstorming" / "SKILL.md").exists()

    def test_inject_sync_existing(self, tmp_path, monkeypatch):
        """Cover lines 647-665: sync SKILL.md for existing skill + subdir sync."""
        import onemancompany.agents.onboarding as ob_mod
        monkeypatch.setattr(ob_mod, "_DEFAULT_SKILLS_DIR", tmp_path / "default_skills")

        src_dir = tmp_path / "default_skills" / "task_lifecycle"
        src_dir.mkdir(parents=True)
        (src_dir / "SKILL.md").write_text("UPDATED CONTENT")
        hooks = src_dir / "hooks"
        hooks.mkdir()
        (hooks / "hook.py").write_text("# hook")

        skills_dir = tmp_path / "skills"
        dst_dir = skills_dir / "task_lifecycle"
        dst_dir.mkdir(parents=True)
        (dst_dir / "SKILL.md").write_text("OLD CONTENT")

        ob_mod._inject_default_skills(skills_dir, employee_id="00020")
        assert (dst_dir / "SKILL.md").read_text() == "UPDATED CONTENT"
        assert (dst_dir / "hooks" / "hook.py").exists()

    def test_inject_hot_updates_subdir_and_toplevel_files(self, tmp_path, monkeypatch):
        """P0: a change to default_skills/ must propagate to an existing
        employee copy — not only NEW files. Pre-fix, manifest.yaml (a
        top-level non-SKILL.md file) and edits to static/ never synced."""
        import onemancompany.agents.onboarding as ob_mod
        monkeypatch.setattr(ob_mod, "_DEFAULT_SKILLS_DIR", tmp_path / "default_skills")

        src_dir = tmp_path / "default_skills" / "task_lifecycle"
        (src_dir / "static").mkdir(parents=True)
        (src_dir / "static" / "deep").mkdir()
        (src_dir / "SKILL.md").write_text("NEW SKILL")
        (src_dir / "manifest.yaml").write_text("version: 2")          # top-level file
        (src_dir / "static" / "core.md").write_text("NEW VOCAB")       # existing subdir file
        (src_dir / "static" / "deep" / "x.md").write_text("NESTED")    # nested subdir file

        skills_dir = tmp_path / "skills"
        dst_dir = skills_dir / "task_lifecycle"
        (dst_dir / "static").mkdir(parents=True)
        (dst_dir / "SKILL.md").write_text("OLD SKILL")
        (dst_dir / "manifest.yaml").write_text("version: 1")
        (dst_dir / "static" / "core.md").write_text("OLD VOCAB")
        # employee-only file that the source does NOT have — must survive
        (dst_dir / "static" / "employee_notes.md").write_text("keep me")

        ob_mod._inject_default_skills(skills_dir, employee_id="00021")

        # source-present files overwrite, including top-level + nested
        assert (dst_dir / "manifest.yaml").read_text() == "version: 2"
        assert (dst_dir / "static" / "core.md").read_text() == "NEW VOCAB"
        assert (dst_dir / "static" / "deep" / "x.md").read_text() == "NESTED"
        # employee-only file is preserved (one-way overlay, never delete-sync)
        assert (dst_dir / "static" / "employee_notes.md").read_text() == "keep me"

    def test_sync_skill_tree_skips_pycache(self, tmp_path):
        import onemancompany.agents.onboarding as ob_mod
        src = tmp_path / "src"
        (src / "__pycache__").mkdir(parents=True)
        (src / "__pycache__" / "junk.pyc").write_text("x")
        (src / "SKILL.md").write_text("ok")
        dst = tmp_path / "dst"
        ob_mod._sync_skill_tree(src, dst)
        assert (dst / "SKILL.md").exists()
        assert not (dst / "__pycache__").exists()


# ---------------------------------------------------------------------------
# _assign_default_avatar (lines 673-694)
# ---------------------------------------------------------------------------

class TestAssignDefaultAvatar:
    def test_already_has_avatar(self, tmp_path):
        from onemancompany.agents.onboarding import _assign_default_avatar
        emp_dir = tmp_path / "00010"
        emp_dir.mkdir()
        (emp_dir / "avatar.png").write_text("img")
        _assign_default_avatar(emp_dir, "00010")

    def test_no_avatars_dir(self, tmp_path, monkeypatch):
        from onemancompany.agents.onboarding import _assign_default_avatar
        import onemancompany.core.config as config_mod
        monkeypatch.setattr(config_mod, "COMPANY_DIR", tmp_path / "company")
        emp_dir = tmp_path / "00010"
        emp_dir.mkdir()
        _assign_default_avatar(emp_dir, "00010")

    def test_no_avatar_files(self, tmp_path, monkeypatch):
        from onemancompany.agents.onboarding import _assign_default_avatar
        import onemancompany.core.config as config_mod
        monkeypatch.setattr(config_mod, "COMPANY_DIR", tmp_path / "company")
        avatars_dir = tmp_path / "company" / "human_resource" / "avatars"
        avatars_dir.mkdir(parents=True)
        emp_dir = tmp_path / "00010"
        emp_dir.mkdir()
        _assign_default_avatar(emp_dir, "00010")

    def test_assign_avatar(self, tmp_path, monkeypatch):
        from onemancompany.agents.onboarding import _assign_default_avatar
        import onemancompany.core.config as config_mod
        monkeypatch.setattr(config_mod, "COMPANY_DIR", tmp_path / "company")
        avatars_dir = tmp_path / "company" / "human_resource" / "avatars"
        avatars_dir.mkdir(parents=True)
        (avatars_dir / "avatar1.png").write_text("img1")
        emp_dir = tmp_path / "00010"
        emp_dir.mkdir()
        _assign_default_avatar(emp_dir, "00010")
        assert (emp_dir / "avatar.png").exists()


# ---------------------------------------------------------------------------
# copy_talent_assets — skills (lines 733-737)
# ---------------------------------------------------------------------------

class TestCopyTalentAssets:
    def test_copy_legacy_md_skills(self, tmp_path):
        from onemancompany.agents.onboarding import copy_talent_assets
        talent_dir = tmp_path / "talent"
        talent_dir.mkdir()
        skills_dir = talent_dir / "skills"
        skills_dir.mkdir()
        (skills_dir / "legacy_skill.md").write_text("---\nname: legacy\n---\nContent")

        emp_dir = tmp_path / "employee"
        emp_dir.mkdir()

        with patch("onemancompany.agents.onboarding.register_tool_user"):
            copy_talent_assets(talent_dir, emp_dir)

        assert (emp_dir / "skills" / "legacy_skill" / "SKILL.md").exists()

    def test_copy_nonexistent_talent(self, tmp_path):
        from onemancompany.agents.onboarding import copy_talent_assets
        emp_dir = tmp_path / "employee"
        emp_dir.mkdir()
        copy_talent_assets(tmp_path / "nonexistent", emp_dir)  # should be no-op


# ---------------------------------------------------------------------------
# copy_talent_assets — tools with manifest (lines 755-770)
# ---------------------------------------------------------------------------

class TestCopyTalentTools:
    def test_copy_tools_with_manifest(self, tmp_path, monkeypatch):
        from onemancompany.agents.onboarding import copy_talent_assets
        import onemancompany.agents.onboarding as ob_mod
        tools_dir = tmp_path / "assets_tools"
        tools_dir.mkdir(parents=True)
        monkeypatch.setattr(ob_mod, "TOOLS_DIR", tools_dir)

        talent_dir = tmp_path / "talent"
        talent_dir.mkdir()
        talent_tools = talent_dir / "tools"
        talent_tools.mkdir()
        (talent_tools / "manifest.yaml").write_text("custom_tools:\n  - my_custom_tool\n")
        tool_subdir = talent_tools / "my_tool"
        tool_subdir.mkdir()
        (tool_subdir / "tool.yaml").write_text("name: my_tool\n")
        # Also add a loose config file
        (talent_tools / "config.yaml").write_text("key: val")

        emp_dir = tmp_path / "employee"
        emp_dir.mkdir()

        with patch("onemancompany.agents.onboarding.register_tool_user"):
            copy_talent_assets(talent_dir, emp_dir)

        assert (tools_dir / "my_tool").exists()
        assert (emp_dir / "tools" / "config.yaml").exists()


# ---------------------------------------------------------------------------
# copy_talent_assets — persona from profile.yaml (lines 777-791)
# ---------------------------------------------------------------------------

class TestCopyTalentPersona:
    def test_persona_from_profile(self, tmp_path):
        from onemancompany.agents.onboarding import copy_talent_assets  # noqa: F811
        talent_dir = tmp_path / "talent"
        talent_dir.mkdir()
        (talent_dir / "profile.yaml").write_text("system_prompt_template: 'I am a robot'\n")

        emp_dir = tmp_path / "employee"
        emp_dir.mkdir()

        with patch("onemancompany.agents.onboarding.register_tool_user"):
            copy_talent_assets(talent_dir, emp_dir)

        assert (emp_dir / "prompts" / "talent_persona.md").exists()

    def test_persona_from_prompts_dir(self, tmp_path):
        from onemancompany.agents.onboarding import copy_talent_assets  # noqa: F811
        talent_dir = tmp_path / "talent"
        talent_dir.mkdir()
        prompts = talent_dir / "prompts"
        prompts.mkdir()
        (prompts / "talent_persona.md").write_text("Custom persona")

        emp_dir = tmp_path / "employee"
        emp_dir.mkdir()

        with patch("onemancompany.agents.onboarding.register_tool_user"):
            copy_talent_assets(talent_dir, emp_dir)

        assert (emp_dir / "prompts" / "talent_persona.md").read_text() == "Custom persona"


# ---------------------------------------------------------------------------
# copy_talent_assets — CLAUDE.md (line 798)
# ---------------------------------------------------------------------------

class TestCopyClaudeMd:
    def test_claude_md_from_talent(self, tmp_path):
        from onemancompany.agents.onboarding import copy_talent_assets  # noqa: F811
        talent_dir = tmp_path / "talent"
        talent_dir.mkdir()
        (talent_dir / "CLAUDE.md").write_text("# Custom Claude MD")

        emp_dir = tmp_path / "employee"
        emp_dir.mkdir()

        with patch("onemancompany.agents.onboarding.register_tool_user"):
            copy_talent_assets(talent_dir, emp_dir)

        assert (emp_dir / "CLAUDE.md").read_text() == "# Custom Claude MD"


# ---------------------------------------------------------------------------
# execute_hire — launch/heartbeat script copy (lines 1021-1031)
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# _SKILL_REQUIRED_RUNBOOKS — skill-conditional default runbook injection
# (regression for bug found in PR #15 smoke test)
# ---------------------------------------------------------------------------

class TestMethodologyDesignerTalentMarketSourced:
    """methodology-debate-convener no longer lives in this repo. It ships
    with the methodology-designer talent at
    https://github.com/YihangChen9/methodology-designer and is fetched at
    hire time via hire_list.json `source_repo` → `clone_talent_repo`.

    These tests pin the post-migration invariants:
      1. The runbook is NOT injected via `_inject_default_skills` (the
         default_skills/ entry was removed).
      2. `_SKILL_REQUIRED_RUNBOOKS` no longer maps methodology_designer.
      3. hire_list.json carries the source_repo URL for the talent.
    """

    def _setup_default_skills(self, tmp_path, monkeypatch):
        import onemancompany.agents.onboarding as ob_mod
        monkeypatch.setattr(ob_mod, "_DEFAULT_SKILLS_DIR", tmp_path / "default_skills")
        # Only task_lifecycle here — methodology-debate-convener is
        # intentionally absent from default_skills now.
        for skill_name in ("task_lifecycle",):
            src_dir = tmp_path / "default_skills" / skill_name
            src_dir.mkdir(parents=True)
            (src_dir / "SKILL.md").write_text(f"---\nname: {skill_name}\n---\nContent for {skill_name}")
        return ob_mod

    def test_inject_does_not_pull_methodology_runbook(self, tmp_path, monkeypatch):
        """Even if employee carries methodology_designer skill, the runbook
        must NOT come from _inject_default_skills — it comes from the
        talent clone instead."""
        ob_mod = self._setup_default_skills(tmp_path, monkeypatch)
        emp_dir = tmp_path / "00006"
        skills_dir = emp_dir / "skills"
        skills_dir.mkdir(parents=True)
        (emp_dir / "profile.yaml").write_text(
            "skills:\n- methodology_designer\nname: Methodology Designer\n"
        )

        ob_mod._inject_default_skills(skills_dir, employee_id="00006")

        # Universal default is still injected
        assert (skills_dir / "task_lifecycle" / "SKILL.md").exists()
        # But the methodology runbook is NOT — it's the talent's job to
        # ship it via clone_talent_repo + copy_talent_assets.
        assert not (skills_dir / "methodology-debate-convener").exists(), (
            "methodology-debate-convener must NOT be injected from default_skills/ "
            "after the talent-market migration"
        )

    def test_methodology_designer_does_not_inject_convener(self):
        """The convener runbook (methodology-debate-convener) ships with
        the talent — it must NOT appear in `_SKILL_REQUIRED_RUNBOOKS`.

        Other shared utility skills (e.g. `paper-framework-figure` for
        nano-banana figure rendering) MAY still be injected onto
        methodology_designer hires from `default_skills/`. So this
        assertion is scoped to the convener specifically, not to the
        whole `methodology_designer` key."""
        from onemancompany.agents.onboarding import _SKILL_REQUIRED_RUNBOOKS
        injected = _SKILL_REQUIRED_RUNBOOKS.get("methodology_designer", [])
        assert "methodology-debate-convener" not in injected, (
            "methodology-debate-convener must come from the talent clone, "
            "not from `_inject_default_skills` (the talent's bundled "
            "skills/ directory is the SSOT)"
        )

    def test_hire_list_uses_talent_market_source(self):
        """hire_list must register the talent as source_type=talent_market
        so `talent_market.onboard()` is called to resolve the repo URL.
        The URL itself lives on the talent market service — we do NOT
        pin it locally (verified against the live service that
        onboard("methodology-designer") returns the right URL)."""
        import json
        repo_root = Path(__file__).resolve().parents[3]
        with open(repo_root / "company" / "hire_list.json") as f:
            entries = json.load(f)
        md_entry = next((e for e in entries if e.get("talent_id") == "methodology-designer"), None)
        assert md_entry is not None, "methodology-designer missing from hire_list.json"
        assert md_entry.get("source_type") == "talent_market", (
            "methodology-designer must have source_type=talent_market so "
            "talent_market.onboard() resolves its repo URL"
        )

    def test_missing_profile_yaml_is_graceful(self, tmp_path, monkeypatch):
        """If profile.yaml is missing and no employee_skills passed, function
        must not crash — fall back to universal-only injection."""
        ob_mod = self._setup_default_skills(tmp_path, monkeypatch)
        skills_dir = tmp_path / "ghost" / "skills"
        skills_dir.mkdir(parents=True)
        # No profile.yaml, no employee_skills arg

        ob_mod._inject_default_skills(skills_dir, employee_id="ghost")
        # task_lifecycle still injected; no crash
        assert (skills_dir / "task_lifecycle" / "SKILL.md").exists()
        assert not (skills_dir / "methodology-debate-convener").exists()


class TestAdversarialReviewerGetsQualityCritic:
    """Whoever has adversarial_review must get the methodology-quality-critic
    runbook auto-injected, so that the Stage 4 critic-side trigger resolves."""

    def test_adversarial_review_employee_gets_quality_critic(self, tmp_path, monkeypatch):
        import onemancompany.agents.onboarding as ob_mod
        monkeypatch.setattr(ob_mod, "_DEFAULT_SKILLS_DIR", tmp_path / "default_skills")

        for skill_name in ("task_lifecycle", "methodology-quality-critic"):
            src_dir = tmp_path / "default_skills" / skill_name
            src_dir.mkdir(parents=True)
            (src_dir / "SKILL.md").write_text(f"---\nname: {skill_name}\n---\nContent")

        emp_dir = tmp_path / "00099"
        skills_dir = emp_dir / "skills"
        skills_dir.mkdir(parents=True)
        (emp_dir / "profile.yaml").write_text(
            "skills:\n- adversarial_review\n- peer_reviewer\nname: Critic\n"
        )

        ob_mod._inject_default_skills(skills_dir, employee_id="00099")

        assert (skills_dir / "methodology-quality-critic" / "SKILL.md").exists()

    def test_skill_required_runbooks_includes_adversarial_review(self):
        from onemancompany.agents.onboarding import _SKILL_REQUIRED_RUNBOOKS
        assert "adversarial_review" in _SKILL_REQUIRED_RUNBOOKS
        assert "methodology-quality-critic" in _SKILL_REQUIRED_RUNBOOKS["adversarial_review"]


class TestExperimentDesignerTalentMarketSourced:
    """experiment-debate-convener moved to the talent repo at
    https://github.com/YihangChen9/experiment-designer. It is fetched at
    hire time via talent_market.onboard() → clone_talent_repo.

    Pins post-migration invariants:
      1. NOT injected via `_inject_default_skills` from default_skills/.
      2. NOT in `_SKILL_REQUIRED_RUNBOOKS`.
      3. hire_list entry has source_type=talent_market (so onboard() is called).
    """

    def _setup_default_skills_minus_convener(self, tmp_path, monkeypatch):
        import onemancompany.agents.onboarding as ob_mod
        monkeypatch.setattr(ob_mod, "_DEFAULT_SKILLS_DIR", tmp_path / "default_skills")
        # Only universals + critics; convener intentionally absent.
        for skill_name in (
            "task_lifecycle",
            "methodology-quality-critic",
            "experiment-quality-critic",
        ):
            src_dir = tmp_path / "default_skills" / skill_name
            src_dir.mkdir(parents=True)
            (src_dir / "SKILL.md").write_text(f"---\nname: {skill_name}\n---\nContent")
        return ob_mod

    def test_experiment_designer_no_convener_from_default_skills(self, tmp_path, monkeypatch):
        ob_mod = self._setup_default_skills_minus_convener(tmp_path, monkeypatch)
        emp_dir = tmp_path / "00200"
        skills_dir = emp_dir / "skills"
        skills_dir.mkdir(parents=True)
        (emp_dir / "profile.yaml").write_text("skills:\n- experiment_designer\nname: ExpDesigner\n")

        ob_mod._inject_default_skills(skills_dir, employee_id="00200")

        assert (skills_dir / "task_lifecycle" / "SKILL.md").exists()
        assert not (skills_dir / "experiment-debate-convener").exists(), (
            "experiment-debate-convener must come from the talent clone, "
            "not from _inject_default_skills"
        )

    def test_adversarial_review_still_gets_experiment_quality_critic(self, tmp_path, monkeypatch):
        """The critic side is unchanged — adversarial_review still gets
        the *-quality-critic runbooks from default_skills/."""
        ob_mod = self._setup_default_skills_minus_convener(tmp_path, monkeypatch)
        emp_dir = tmp_path / "00201"
        skills_dir = emp_dir / "skills"
        skills_dir.mkdir(parents=True)
        (emp_dir / "profile.yaml").write_text("skills:\n- adversarial_review\nname: Critic\n")

        ob_mod._inject_default_skills(skills_dir, employee_id="00201")

        assert (skills_dir / "methodology-quality-critic" / "SKILL.md").exists()
        assert (skills_dir / "experiment-quality-critic" / "SKILL.md").exists()

    def test_experiment_designer_not_in_required_runbooks(self):
        from onemancompany.agents.onboarding import _SKILL_REQUIRED_RUNBOOKS
        assert "experiment_designer" not in _SKILL_REQUIRED_RUNBOOKS

    def test_adversarial_review_includes_experiment_quality_critic(self):
        from onemancompany.agents.onboarding import _SKILL_REQUIRED_RUNBOOKS
        assert "experiment-quality-critic" in _SKILL_REQUIRED_RUNBOOKS["adversarial_review"]

    def test_hire_list_uses_talent_market_source(self):
        """hire_list must register the talent as source_type=talent_market
        so `talent_market.onboard()` is called to resolve the repo URL.
        The URL itself is the talent market service's responsibility,
        not OMC's — we do NOT pin it locally."""
        import json
        repo_root = Path(__file__).resolve().parents[3]
        with open(repo_root / "company" / "hire_list.json") as f:
            entries = json.load(f)
        entry = next((e for e in entries if e.get("talent_id") == "experiment-designer"), None)
        assert entry is not None, "experiment-designer missing from hire_list.json"
        assert entry.get("source_type") == "talent_market", (
            "experiment-designer must have source_type=talent_market so "
            "talent_market.onboard() resolves its repo URL"
        )


class TestExperimentRunnerTalentMarketSourced:
    """experiment-infra + experiment-execution-runbook moved to the
    multi-talent repo at https://github.com/YihangChen9/experiment-team.
    Pins post-migration invariants for both experiment_runner and
    code_implementer skills (Stage 6 execution + implementation
    sub-phases)."""

    def _setup_default_skills_minus_stage6(self, tmp_path, monkeypatch):
        import onemancompany.agents.onboarding as ob_mod
        monkeypatch.setattr(ob_mod, "_DEFAULT_SKILLS_DIR", tmp_path / "default_skills")
        # Only universals + critics. Stage 6 runbooks deliberately absent.
        for skill_name in ("task_lifecycle",):
            src_dir = tmp_path / "default_skills" / skill_name
            src_dir.mkdir(parents=True)
            (src_dir / "SKILL.md").write_text(f"---\nname: {skill_name}\n---\nContent")
        return ob_mod

    def test_experiment_runner_no_runbooks_from_default_skills(self, tmp_path, monkeypatch):
        ob_mod = self._setup_default_skills_minus_stage6(tmp_path, monkeypatch)
        emp_dir = tmp_path / "00300"
        skills_dir = emp_dir / "skills"
        skills_dir.mkdir(parents=True)
        (emp_dir / "profile.yaml").write_text(
            "skills:\n- experiment_runner\nname: ExpRunner\n"
        )

        ob_mod._inject_default_skills(skills_dir, employee_id="00300")

        assert (skills_dir / "task_lifecycle" / "SKILL.md").exists()
        assert not (skills_dir / "experiment-infra").exists(), (
            "experiment-infra must come from the talent clone, not "
            "_inject_default_skills"
        )
        assert not (skills_dir / "experiment-execution-runbook").exists()

    def test_experiment_runner_not_in_required_runbooks(self):
        from onemancompany.agents.onboarding import _SKILL_REQUIRED_RUNBOOKS
        assert "experiment_runner" not in _SKILL_REQUIRED_RUNBOOKS

    def test_code_implementer_not_in_required_runbooks(self):
        """Stage 6 implementation sub-phase routes to code_implementer.
        Like experiment_runner, the runbook ships with the talent."""
        from onemancompany.agents.onboarding import _SKILL_REQUIRED_RUNBOOKS
        assert "code_implementer" not in _SKILL_REQUIRED_RUNBOOKS

    def test_hire_list_runner_uses_talent_market_source(self):
        import json
        repo_root = Path(__file__).resolve().parents[3]
        with open(repo_root / "company" / "hire_list.json") as f:
            entries = json.load(f)
        entry = next((e for e in entries if e.get("talent_id") == "experiment-runner"), None)
        assert entry is not None, "experiment-runner missing from hire_list.json"
        assert entry.get("source_type") == "talent_market"

    def test_hire_list_code_writer_uses_talent_market_source(self):
        import json
        repo_root = Path(__file__).resolve().parents[3]
        with open(repo_root / "company" / "hire_list.json") as f:
            entries = json.load(f)
        entry = next((e for e in entries if e.get("talent_id") == "experiment-code-writer"), None)
        assert entry is not None, "experiment-code-writer missing from hire_list.json"
        assert entry.get("source_type") == "talent_market"


class TestResultAnalystTalentMarketSourced:
    """result-analysis-runbook moved to the talent repo at
    https://github.com/YihangChen9/result-analyst. Same pattern as
    TestExperimentDesignerTalentMarketSourced."""

    def _setup_default_skills_minus_runbook(self, tmp_path, monkeypatch):
        import onemancompany.agents.onboarding as ob_mod
        monkeypatch.setattr(ob_mod, "_DEFAULT_SKILLS_DIR", tmp_path / "default_skills")
        for skill_name in ("task_lifecycle", "result-quality-critic"):
            src_dir = tmp_path / "default_skills" / skill_name
            src_dir.mkdir(parents=True)
            (src_dir / "SKILL.md").write_text(f"---\nname: {skill_name}\n---\nContent")
        return ob_mod

    def test_result_analyst_no_runbook_from_default_skills(self, tmp_path, monkeypatch):
        ob_mod = self._setup_default_skills_minus_runbook(tmp_path, monkeypatch)
        emp_dir = tmp_path / "00210"
        skills_dir = emp_dir / "skills"
        skills_dir.mkdir(parents=True)
        (emp_dir / "profile.yaml").write_text("skills:\n- result_analyst\nname: ResultAnalyst\n")

        ob_mod._inject_default_skills(skills_dir, employee_id="00210")

        assert (skills_dir / "task_lifecycle" / "SKILL.md").exists()
        assert not (skills_dir / "result-analysis-runbook").exists(), (
            "result-analysis-runbook must come from the talent clone, "
            "not from _inject_default_skills"
        )

    def test_result_analyst_not_in_required_runbooks(self):
        from onemancompany.agents.onboarding import _SKILL_REQUIRED_RUNBOOKS
        assert "result_analyst" not in _SKILL_REQUIRED_RUNBOOKS

    def test_adversarial_review_still_gets_result_quality_critic(self):
        from onemancompany.agents.onboarding import _SKILL_REQUIRED_RUNBOOKS
        assert "result-quality-critic" in _SKILL_REQUIRED_RUNBOOKS["adversarial_review"]

    def test_hire_list_uses_talent_market_source(self):
        import json
        repo_root = Path(__file__).resolve().parents[3]
        with open(repo_root / "company" / "hire_list.json") as f:
            entries = json.load(f)
        entry = next((e for e in entries if e.get("talent_id") == "result-analyst"), None)
        assert entry is not None, "result-analyst missing from hire_list.json"
        assert entry.get("source_type") == "talent_market"

    # Keep the Stage 7 critic-side check — adversarial_review still picks
    # up result-quality-critic from default_skills/ (the *-quality-critic
    # runbooks have not been migrated to a talent yet).
    def test_adversarial_review_gets_result_quality_critic(self, tmp_path, monkeypatch):
        """The Stage 7 critic-side runbook (result-quality-critic) must
        be injected alongside the existing methodology + experiment
        quality critics whenever a hire has adversarial_review."""
        import onemancompany.agents.onboarding as ob_mod
        monkeypatch.setattr(ob_mod, "_DEFAULT_SKILLS_DIR", tmp_path / "default_skills")
        for skill_name in (
            "task_lifecycle",
            "methodology-quality-critic",
            "experiment-quality-critic",
            "result-quality-critic",
        ):
            src_dir = tmp_path / "default_skills" / skill_name
            src_dir.mkdir(parents=True)
            (src_dir / "SKILL.md").write_text(f"---\nname: {skill_name}\n---\nContent")

        emp_dir = tmp_path / "00502"
        skills_dir = emp_dir / "skills"
        skills_dir.mkdir(parents=True)
        (emp_dir / "profile.yaml").write_text(
            "skills:\n- adversarial_review\nname: Critic\n"
        )

        ob_mod._inject_default_skills(skills_dir, employee_id="00502")

        # Existing critics must still be present (no regression)
        assert (skills_dir / "methodology-quality-critic" / "SKILL.md").exists()
        assert (skills_dir / "experiment-quality-critic" / "SKILL.md").exists()
        # NEW: Stage 7 critic must also land
        assert (skills_dir / "result-quality-critic" / "SKILL.md").exists()


class TestPaperFrameworkFigureSkill:
    """paper-framework-figure must auto-inject for both methodology_designer
    (Stage 4) and paper_writer (Stage 8), so `load_skill(...)` resolves the
    nano-banana figure runbook in the agent's skills/ dir at runtime."""

    def _setup(self, tmp_path, monkeypatch):
        import onemancompany.agents.onboarding as ob_mod
        monkeypatch.setattr(ob_mod, "_DEFAULT_SKILLS_DIR", tmp_path / "default_skills")
        for skill_name in ("task_lifecycle", "paper-framework-figure"):
            src_dir = tmp_path / "default_skills" / skill_name
            src_dir.mkdir(parents=True)
            (src_dir / "SKILL.md").write_text(f"---\nname: {skill_name}\n---\nContent")
        return ob_mod

    def test_methodology_designer_gets_figure_skill(self, tmp_path, monkeypatch):
        ob_mod = self._setup(tmp_path, monkeypatch)
        emp_dir = tmp_path / "00400"
        skills_dir = emp_dir / "skills"
        skills_dir.mkdir(parents=True)
        (emp_dir / "profile.yaml").write_text(
            "skills:\n- methodology_designer\nname: MethDesigner\n"
        )
        ob_mod._inject_default_skills(skills_dir, employee_id="00400")
        assert (skills_dir / "paper-framework-figure" / "SKILL.md").exists()

    def test_paper_writer_no_figure_skill_injection(self, tmp_path, monkeypatch):
        """Stage 8 (paper_writer) does NOT need paper-framework-figure
        injected anymore — it reuses stage4_framework_figure.png by path.
        Injecting would burn API budget regenerating a duplicate figure."""
        ob_mod = self._setup(tmp_path, monkeypatch)
        emp_dir = tmp_path / "00401"
        skills_dir = emp_dir / "skills"
        skills_dir.mkdir(parents=True)
        (emp_dir / "profile.yaml").write_text(
            "skills:\n- paper_writer\nname: PaperWriter\n"
        )
        ob_mod._inject_default_skills(skills_dir, employee_id="00401")
        assert not (skills_dir / "paper-framework-figure").exists()

    def test_other_employees_do_not_get_figure_skill(self, tmp_path, monkeypatch):
        ob_mod = self._setup(tmp_path, monkeypatch)
        emp_dir = tmp_path / "00402"
        skills_dir = emp_dir / "skills"
        skills_dir.mkdir(parents=True)
        (emp_dir / "profile.yaml").write_text("skills:\n- topic_refiner\n")
        ob_mod._inject_default_skills(skills_dir, employee_id="00402")
        assert not (skills_dir / "paper-framework-figure").exists()

    def test_mapping_includes_methodology_only(self):
        """methodology_designer renders the figure; paper_writer only reuses it.
        Map must list the former and NOT the latter."""
        from onemancompany.agents.onboarding import _SKILL_REQUIRED_RUNBOOKS
        assert "paper-framework-figure" in _SKILL_REQUIRED_RUNBOOKS.get("methodology_designer", [])
        assert "paper-framework-figure" not in _SKILL_REQUIRED_RUNBOOKS.get("paper_writer", [])


class TestPaperFrameworkFigureSkillFile:
    """Sanity-check the paper-framework-figure SKILL.md ships with the
    required hard-gate, wrapper directive, and the 4-section summary
    schema. If any drifts, the skill won't produce the right image."""

    SKILL_PATH = (
        Path(__file__).resolve().parents[3]
        / "src" / "onemancompany" / "default_skills"
        / "paper-framework-figure" / "SKILL.md"
    )

    def test_file_exists(self):
        assert self.SKILL_PATH.exists()

    def test_has_frontmatter(self):
        head = self.SKILL_PATH.read_text(encoding="utf-8")[:600]
        assert head.startswith("---\n")
        assert "name: paper-framework-figure" in head
        assert "google/gemini-2.5-flash-image" in head

    def test_warns_about_chat_mode_bug(self):
        """Without the 'Generate ONE image' wrapper, the model replies in
        chat mode (text, 0 image tokens). The SKILL.md MUST warn the
        agent so it doesn't waste calls."""
        text = self.SKILL_PATH.read_text(encoding="utf-8")
        assert "Generate ONE image" in text
        assert "image_tokens: 0" in text or "no image" in text.lower()

    def test_specifies_4_section_summary_schema(self):
        text = self.SKILL_PATH.read_text(encoding="utf-8")
        for heading in ("背景", "问题和难点", "创新点", "具体的技术路线"):
            assert heading in text, f"missing 4-section schema heading {heading!r}"

    def test_warns_about_api_key_secrecy(self):
        text = self.SKILL_PATH.read_text(encoding="utf-8")
        assert "OPENROUTER_API_KEY" in text
        assert "Do NOT echo" in text or "Do NOT" in text
