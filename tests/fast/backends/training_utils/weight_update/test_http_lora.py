"""The load path must REPLACE, not skip.

The updater hands a transport a stable adapter name every sync
(``LORA_ADAPTER_NAME``, or ``slot_lora_name(slot)`` under multi-LoRA), so every
sync after the first hits a name the engine already holds. SGLang's registry
raises ``LoRA with name X already exists``. If that is treated as success, v1
loads and every later version is written to disk, refused by the engine, and
logged as a successful sync -- rollouts keep sampling the step-1 adapter while
the trainer moves on, and nothing looks wrong.
"""

from __future__ import annotations

import os
from argparse import Namespace

import pytest
import torch

from miles.backends.training_utils.weight_update.protocols.http_lora import UpdateWeightHttpLora


class _Resp:
    def __init__(self, status_code: int, text: str = ""):
        self.status_code = status_code
        self.text = text


class _FakeClient:
    """Records every call and replays a scripted response per route."""

    def __init__(self, script):
        self._script = script
        self.calls: list[tuple[str, dict]] = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def post(self, url, json):  # noqa: A002
        route = url.rsplit("/", 1)[-1]
        self.calls.append((route, json))
        seq = self._script[route]
        return seq.pop(0) if isinstance(seq, list) else seq


def _protocol(tmp_path) -> UpdateWeightHttpLora:
    p = UpdateWeightHttpLora.__new__(UpdateWeightHttpLora)  # bypass __init__/engine wiring
    p.args = Namespace(lora_rank=32, lora_alpha=64.0)
    p._stage_root = str(tmp_path)
    p._keep_versions = 3
    p._load_timeout_s = 5.0
    return p


def _patch_httpx(monkeypatch, client):
    import httpx

    monkeypatch.setattr(httpx, "Client", lambda **kw: client)


def test_a_clean_load_does_not_unload(tmp_path, monkeypatch):
    c = _FakeClient({"load_lora_adapter": _Resp(200)})
    _patch_httpx(monkeypatch, c)
    _protocol(tmp_path)._load_one("http://e:1", "miles_lora", "/d/miles_lora_v1")
    assert [r for r, _ in c.calls] == ["load_lora_adapter"]


def test_an_already_registered_name_is_replaced_not_skipped(tmp_path, monkeypatch):
    """THE regression: sync 2+ must actually change the served adapter."""
    c = _FakeClient({
        "load_lora_adapter": [
            _Resp(400, "LoRA with name miles_lora already exists. Loaded LoRAs: ..."),
            _Resp(200),
        ],
        "unload_lora_adapter": _Resp(200),
    })
    _patch_httpx(monkeypatch, c)
    _protocol(tmp_path)._load_one("http://e:1", "miles_lora", "/d/miles_lora_v2")

    assert [r for r, _ in c.calls] == [
        "load_lora_adapter", "unload_lora_adapter", "load_lora_adapter"
    ], "a name collision must unload and reload, not be swallowed as success"
    assert c.calls[-1][1]["lora_path"] == "/d/miles_lora_v2", "the NEW version must be the one loaded"


def test_a_failed_unload_is_an_error(tmp_path, monkeypatch):
    c = _FakeClient({
        "load_lora_adapter": [_Resp(400, "already exists")],
        "unload_lora_adapter": _Resp(500, "boom"),
    })
    _patch_httpx(monkeypatch, c)
    with pytest.raises(RuntimeError, match="unload failed"):
        _protocol(tmp_path)._load_one("http://e:1", "miles_lora", "/d/v2")


def test_an_unrelated_failure_never_triggers_an_unload(tmp_path, monkeypatch):
    c = _FakeClient({"load_lora_adapter": _Resp(500, "rank 64 exceeds --max-lora-rank")})
    _patch_httpx(monkeypatch, c)
    with pytest.raises(RuntimeError, match="load failed"):
        _protocol(tmp_path)._load_one("http://e:1", "miles_lora", "/d/v1")
    assert all(r != "unload_lora_adapter" for r, _ in c.calls)


@pytest.mark.parametrize("body", [
    "LoRA with name miles_lora already exists. Loaded LoRAs: dict_keys(['miles_lora'])",
    "adapter already loaded",
])
def test_the_engine_conflict_message_is_recognised(body):
    assert UpdateWeightHttpLora._is_name_conflict(_Resp(400, body))


def test_an_unrelated_message_is_not_mistaken_for_a_conflict():
    # "already" alone is too loose: it also appears in unrelated engine errors.
    assert not UpdateWeightHttpLora._is_name_conflict(_Resp(500, "request already aborted by client"))


def test_rank_comes_from_the_tensors_not_the_flags(tmp_path):
    p = _protocol(tmp_path)
    p.args.lora_rank = 999  # a wrong flag must not reach the published config
    cfg = p._adapter_config({
        "m.layers.0.self_attn.q_proj.lora_A.weight": torch.zeros(16, 2048),
        "m.layers.0.self_attn.q_proj.lora_B.weight": torch.zeros(2048, 16),
    })
    assert cfg["r"] == 16
    assert isinstance(cfg["lora_alpha"], float), "a float alpha must not be truncated"


def test_old_versions_are_pruned_but_the_current_one_survives(tmp_path):
    p = _protocol(tmp_path)
    for v in range(1, 7):
        os.makedirs(tmp_path / f"miles_lora_v{v}")
    p._prune_old_versions("miles_lora", current=6)
    left = sorted(int(d.split("_v")[-1]) for d in os.listdir(tmp_path))
    assert left == [4, 5, 6], f"expected the 3 newest, got {left}"
