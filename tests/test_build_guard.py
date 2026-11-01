import pytest

from ttsdata.build import build, foreign_entries
from ttsdata.config import Config


def test_foreign_entries_lists_only_files_later_stages_added(tmp_path):
    assert foreign_entries(tmp_path / "missing") == []
    for name in ["wavs", "wavs_clean"]:
        (tmp_path / name).mkdir()
    for name in ["metadata.jsonl", "stats.json", "AUDIT.md", "emotions.jsonl"]:
        (tmp_path / name).write_text("x")
    assert foreign_entries(tmp_path) == ["emotions.jsonl", "wavs_clean"]


def test_build_refuses_to_wipe_later_stage_files(tmp_path):
    cfg = Config(root=tmp_path)
    (cfg.dataset_dir / "wavs_sidon").mkdir(parents=True)
    with pytest.raises(SystemExit, match="wavs_sidon"):
        build(cfg)
    assert (cfg.dataset_dir / "wavs_sidon").is_dir()


def test_build_overwrite_and_plain_rebuild_still_work(tmp_path):
    cfg = Config(root=tmp_path)
    (cfg.dataset_dir / "wavs").mkdir(parents=True)
    (cfg.dataset_dir / "stats.json").write_text("{}")
    build(cfg)  # only build outputs there: rebuild allowed (no segments, empty dataset)
    (cfg.dataset_dir / "emotions.jsonl").write_text("x")
    build(cfg, overwrite=True)
    assert not (cfg.dataset_dir / "emotions.jsonl").exists()
