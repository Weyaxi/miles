"""HTTP-level tests for how the session server decides outbound chat-request arguments.

Body fields: ``request_rules.chat_request_rules`` (what the server owns, what it
rejects, what it forwards).  ``chat_template_kwargs``: ``TITOTokenizer.for_request``
against the ``session_args`` the session records on its first committed turn.
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

import pytest
import requests
from fastapi.responses import JSONResponse
from tests.fast.router.test_sessions import _create_session, _post_chat, _serve_router
from tests.fast.router.test_sessions_v2 import _serve_router as _serve_router_v2

from miles.utils.lora import LORA_ADAPTER_NAME
from miles.utils.test_utils.mock_sglang_server import MockSGLangServer

USER = {"role": "user", "content": "hi"}
LAUNCH_KWARGS = {"enable_thinking": False}  # both ``_serve_router`` helpers launch with this
THINKING_ON = {"enable_thinking": True}


def _serve(version: str, extra_args: dict | None = None):
    serve = _serve_router_v2 if version == "v2" else _serve_router
    return serve(extra_args)


def _records(url: str, session_id: str) -> list[dict]:
    return requests.get(f"{url}/sessions/{session_id}", timeout=5.0).json()["records"]


def _metadata(url: str, session_id: str) -> dict:
    return requests.get(f"{url}/sessions/{session_id}", timeout=5.0).json()["metadata"]


class TestForbiddenClientFields:
    @pytest.mark.parametrize(
        ("field", "value"),
        [("input_ids", [1, 2, 3]), ("routed_experts_start_len", 0), ("logprob_start_len", 0), ("lora_path", "x")],
    )
    def test_tito_control_fields_return_400_and_record_nothing(self, field, value):
        with _serve_router() as env:
            session_id = _create_session(env.url)
            resp = _post_chat(env.url, session_id, {"messages": [USER], field: value})
            assert resp.status_code == 400
            assert f"{field}={value!r} is not accepted" in resp.json()["error"]
            assert _records(env.url, session_id) == []

    def test_model_adapter_suffix_rejected_only_when_lora_rollout_is_enabled(self):
        with _serve_router({"lora_rank": 8}) as env:
            session_id = _create_session(env.url)
            resp = _post_chat(env.url, session_id, {"messages": [USER], "model": "base:adapter"})
            assert resp.status_code == 400
            assert "LoRA adapter" in resp.json()["error"]
        with _serve_router() as env:
            session_id = _create_session(env.url)
            assert _post_chat(env.url, session_id, {"messages": [USER], "model": "base:adapter"}).status_code == 200
            assert env.backend.request_log[-1]["model"] == "base:adapter"


class TestServerOwnedFields:
    def test_client_values_are_replaced_and_replay_flags_are_always_present(self):
        with _serve_router() as env:
            session_id = _create_session(env.url)
            resp = _post_chat(
                env.url,
                session_id,
                {
                    "messages": [USER],
                    "temperature": 0.7,
                    "logprobs": False,
                    "return_meta_info": False,
                    "no_stop_trim": True,
                    "return_routed_experts": True,
                    "return_indexer_topk": True,
                },
            )
            assert resp.status_code == 200
            wire = env.backend.request_log[-1]
            assert wire["logprobs"] is True
            assert wire["return_meta_info"] is True
            assert wire["no_stop_trim"] is False
            assert wire["return_routed_experts"] is False
            assert wire["return_indexer_topk"] is False
            assert "lora_path" not in wire
            assert wire["temperature"] == 0.7  # not in the table: the client's

    def test_lora_path_follows_lora_rollout_enabled(self):
        with _serve_router({"lora_rank": 8}) as env:
            session_id = _create_session(env.url)
            assert _post_chat(env.url, session_id, {"messages": [USER]}).status_code == 200
            assert env.backend.request_log[-1]["lora_path"] == LORA_ADAPTER_NAME
        with _serve_router({"lora_rank": 8, "lora_train_only": True}) as env:
            session_id = _create_session(env.url)
            assert _post_chat(env.url, session_id, {"messages": [USER]}).status_code == 200
            assert "lora_path" not in env.backend.request_log[-1]


class TestChatTemplateKwargs:
    def test_request_kwargs_override_the_launch_for_renderer_and_wire(self):
        with _serve_router() as env:
            session_id = _create_session(env.url)
            assert _post_chat(env.url, session_id, {"messages": [USER]}).status_code == 200
            launch_wire = env.backend.request_log[-1]
            assert launch_wire["chat_template_kwargs"] == LAUNCH_KWARGS

            session_id = _create_session(env.url)
            resp = _post_chat(env.url, session_id, {"messages": [USER], "chat_template_kwargs": THINKING_ON})
            assert resp.status_code == 200
            wire = env.backend.request_log[-1]
            assert wire["chat_template_kwargs"] == THINKING_ON
            assert wire["input_ids"] != launch_wire["input_ids"]  # the local render followed the request

    def test_non_object_chat_template_kwargs_is_400(self):
        with _serve_router() as env:
            session_id = _create_session(env.url)
            resp = _post_chat(env.url, session_id, {"messages": [USER], "chat_template_kwargs": "oops"})
            assert resp.status_code == 400
            assert resp.json()["error"] == "chat_template_kwargs must be an object"


@pytest.mark.parametrize("version", ["v1", "v2"])
class TestSessionArgs:
    """The first committed turn records ``session_args``; later turns render alike."""

    def _turn(self, env, session_id: str, messages: list, **extra) -> requests.Response:
        return _post_chat(env.url, session_id, {"messages": messages, **extra})

    def test_first_commit_records_omitted_kwargs_inherit_and_changes_are_400(self, version):
        with _serve(version) as env:
            session_id = _create_session(env.url)
            assert _metadata(env.url, session_id)["session_args"] == {}

            first = self._turn(env, session_id, [USER], chat_template_kwargs=THINKING_ON)
            assert first.status_code == 200
            assert _metadata(env.url, session_id)["session_args"] == {"chat_template_kwargs": THINKING_ON}
            assistant = first.json()["choices"][0]["message"]
            history = [USER, assistant, {"role": "user", "content": "more"}]

            second = self._turn(env, session_id, history)
            assert second.status_code == 200
            assert env.backend.request_log[-1]["chat_template_kwargs"] == THINKING_ON

            third = self._turn(env, session_id, history, chat_template_kwargs=LAUNCH_KWARGS)
            assert third.status_code == 400
            assert "on its first turn" in third.json()["error"]

            fourth = self._turn(env, session_id, history, chat_template_kwargs=THINKING_ON)
            assert fourth.status_code == 200
            # Retrying the same history: v1 rolls back one assistant step and re-appends
            # (2 linear records); v2 commits the retry as a sibling node (3 nodes).
            if version == "v1":
                assert len(_records(env.url, session_id)) == 2
            else:
                assert len(_metadata(env.url, session_id)["tree"]["nodes"]) == 3

    def test_failed_first_turn_records_nothing(self, version):
        original = MockSGLangServer._handle_generate_like_request
        calls = {"n": 0}

        async def fail_first(self, request, compute_fn):
            calls["n"] += 1
            if calls["n"] == 1:
                return JSONResponse(status_code=500, content={"error": "backend down"})
            return await original(self, request, compute_fn)

        with _serve(version) as env:
            session_id = _create_session(env.url)
            with patch.object(MockSGLangServer, "_handle_generate_like_request", new=fail_first):
                assert self._turn(env, session_id, [USER], chat_template_kwargs=THINKING_ON).status_code == 500
                assert _metadata(env.url, session_id)["session_args"] == {}
                assert self._turn(env, session_id, [USER]).status_code == 200
            assert env.backend.request_log[-1]["chat_template_kwargs"] == LAUNCH_KWARGS
            assert _metadata(env.url, session_id)["session_args"] == {"chat_template_kwargs": LAUNCH_KWARGS}
            assert self._turn(env, session_id, [USER], chat_template_kwargs=THINKING_ON).status_code == 400


def test_v2_concurrent_first_turns_with_different_kwargs_record_exactly_once():
    """Both replies are served, only the first commit is recorded, and its kwargs become the session's."""
    with _serve("v2") as env:
        session_id = _create_session(env.url)
        arrivals = 0
        release = None

        async def wait_for_pair(self, request, compute_fn):
            nonlocal arrivals, release
            payload = await request.json()
            self.request_log.append(payload)
            if release is None:
                release = asyncio.Event()
            arrivals += 1
            if arrivals == 2:
                release.set()
            await asyncio.wait_for(release.wait(), timeout=5.0)
            return JSONResponse(content=compute_fn(payload))

        with patch.object(MockSGLangServer, "_handle_generate_like_request", new=wait_for_pair):
            with ThreadPoolExecutor(max_workers=2) as pool:
                futures = [
                    pool.submit(_post_chat, env.url, session_id, {"messages": [USER], "chat_template_kwargs": kwargs})
                    for kwargs in (THINKING_ON, LAUNCH_KWARGS)
                ]
                responses = [future.result(timeout=10.0) for future in futures]

        assert all(response.status_code == 200 for response in responses)
        records = _records(env.url, session_id)
        assert len(records) == 1
        recorded = records[0]["request"]["chat_template_kwargs"]
        assert _metadata(env.url, session_id)["session_args"] == {"chat_template_kwargs": recorded}
        other = LAUNCH_KWARGS if recorded == THINKING_ON else THINKING_ON
        assistant = records[0]["response"]["choices"][0]["message"]
        history = [USER, assistant, {"role": "user", "content": "more"}]
        assert _post_chat(env.url, session_id, {"messages": history, "chat_template_kwargs": other}).status_code == 400
        assert (
            _post_chat(env.url, session_id, {"messages": history, "chat_template_kwargs": recorded}).status_code == 200
        )
