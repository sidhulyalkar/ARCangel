import json
from pathlib import Path

from arc3lab.model.server import discover_model_path, launch_vllm


def _write_model(root: Path, name: str) -> Path:
    path = root / name
    path.mkdir(parents=True)
    (path / "config.json").write_text(json.dumps({"model_type": name, "hidden_size": 1}), encoding="utf-8")
    (path / "tokenizer.json").write_text("{}", encoding="utf-8")
    return path


def test_model_discovery_prefers_qwen38_27b_fp8_over_qualified_qwen36(tmp_path):
    old = _write_model(tmp_path, "vrfai-Qwen3.6-27B-FP8")
    new = _write_model(tmp_path, "Qwen3.8-27B-FP8-Repacked")
    chosen = discover_model_path([str(tmp_path)])
    assert chosen == str(new)
    assert chosen != str(old)


def test_vllm_multimodal_argument_is_canonical_single_json_argv(monkeypatch, tmp_path):
    captured = {}

    class Proc:
        returncode = None
        def poll(self):
            return None
        def terminate(self):
            pass

    class Response:
        ok = True

    def fake_popen(cmd, stdout=None, stderr=None, env=None):
        captured["cmd"] = list(cmd)
        return Proc()

    monkeypatch.setattr("subprocess.Popen", fake_popen)
    monkeypatch.setattr("requests.get", lambda *args, **kwargs: Response())

    log = tmp_path / "vllm.log"
    launch_vllm(
        "/fake/Qwen3.8-27B-FP8",
        limit_mm_per_prompt={"image": 2, "video": 0},
        max_num_seqs=28,
        log_path=log,
        timeout=1,
    )
    cmd = captured["cmd"]
    i = cmd.index("--limit-mm-per-prompt")
    arg = cmd[i + 1]
    assert json.loads(arg) == {"image": 2, "video": 0}
    assert arg == '{"image":2,"video":0}'
    assert "{{" not in arg and "}}" not in arg
    j = cmd.index("--max-num-seqs")
    assert cmd[j + 1] == "28"
