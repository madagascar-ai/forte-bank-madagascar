import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("merge_env", ROOT / "deploy" / "gpu" / "merge_env.py")
merge_env = importlib.util.module_from_spec(spec)
spec.loader.exec_module(merge_env)


def test_github_settings_override_matching_lines_and_keep_server_specific_ones():
    server = "# comment\nFORTE_HOST=100.118.149.106\nFORTE_API_KEY=old\nFORTE_DETECTOR=world\n"
    merged, changed = merge_env.merge(server, {"FORTE_API_KEY": "new", "FORTE_DETECTOR": "world", "FORTE_X": "1"})
    assert "FORTE_HOST=100.118.149.106" in merged
    assert "FORTE_API_KEY=new" in merged and "FORTE_API_KEY=old" not in merged
    assert merged.rstrip().endswith("FORTE_X=1")
    assert sorted(changed) == ["FORTE_API_KEY", "FORTE_X"]  # unchanged values are not reported


@pytest.mark.parametrize("name,value", [("PATH", "/tmp"), ("FORTE_X", "a\nFORTE_API_KEY=evil"), ("forte_x", "1")])
def test_only_single_line_forte_settings_are_accepted(name, value):
    with pytest.raises(ValueError):
        merge_env.merge("FORTE_HOST=x\n", {name: value})


def test_merged_file_is_private(tmp_path, monkeypatch):
    target, source = tmp_path / ".env", tmp_path / ".env.github"
    target.write_text("FORTE_API_KEY=old\n")
    source.write_text("FORTE_API_KEY=new\n")
    monkeypatch.setattr("sys.argv", ["merge_env.py", str(target), str(source)])
    merge_env.main()
    assert target.read_text() == "FORTE_API_KEY=new\n"
    assert target.stat().st_mode & 0o777 == 0o600
