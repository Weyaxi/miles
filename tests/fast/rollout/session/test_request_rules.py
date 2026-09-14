"""Unit tests for ``miles.rollout.session.request_rules``: the two rule functions,
``apply_rules``, and the table ``chat_request_rules`` builds from the config."""

import logging

import pytest
from tests.fast.fixtures.session_fixtures import make_session_server_config

from miles.rollout.session.errors import MessageValidationError
from miles.rollout.session.request_rules import apply_rules, chat_request_rules, server_first, server_strict
from miles.utils.lora import LORA_ADAPTER_NAME

RULES_LOGGER = "miles.rollout.session.request_rules"


class TestApplyRules:
    def test_unruled_fields_are_forwarded_as_sent_and_keep_their_order(self):
        rules = {"logprobs": server_first(True, why="tito")}
        client = {"model": "m", "temperature": 0.7, "unknown": {"x": 1}, "explicit_none": None}

        out = apply_rules(client, rules)

        assert out == {**client, "logprobs": True}
        assert list(out) == ["model", "temperature", "unknown", "explicit_none", "logprobs"]

    def test_server_first_replaces_a_different_value_and_logs_why(self, caplog):
        rules = {"logprobs": server_first(True, why="TITO reads logprobs")}

        with caplog.at_level(logging.WARNING, logger=RULES_LOGGER):
            out = apply_rules({"logprobs": False}, rules)

        assert out == {"logprobs": True}
        assert "logprobs=False from the client replaced by True: TITO reads logprobs" in caplog.text

    def test_server_first_is_silent_when_the_client_agrees_or_says_nothing(self, caplog):
        rules = {"logprobs": server_first(True, why="tito")}

        with caplog.at_level(logging.WARNING, logger=RULES_LOGGER):
            assert apply_rules({"logprobs": True}, rules) == {"logprobs": True}
            assert apply_rules({}, rules) == {"logprobs": True}
            assert apply_rules({"logprobs": None}, rules) == {"logprobs": True}

        assert caplog.text == ""

    def test_server_strict_rejects_a_different_value_with_why(self):
        rules = {"input_ids": server_strict(None, why="rendered by the session server")}

        with pytest.raises(MessageValidationError) as excinfo:
            apply_rules({"input_ids": [1, 2]}, rules)

        assert str(excinfo.value) == "input_ids=[1, 2] is not accepted: rendered by the session server"
        assert excinfo.value.status_code == 400

    def test_server_strict_accepts_the_same_value(self):
        rules = {"lora_path": server_strict("adapter", why="training picks it")}
        assert apply_rules({"lora_path": "adapter"}, rules) == {"lora_path": "adapter"}

    def test_a_none_server_value_keeps_the_field_off_the_wire(self):
        rules = {"lora_path": server_strict(None, why="no lora")}
        assert apply_rules({"model": "m"}, rules) == {"model": "m"}
        assert apply_rules({"lora_path": None}, rules) == {}


class TestChatRequestRules:
    def test_default_config_values_and_strictness(self):
        rules = chat_request_rules(make_session_server_config())

        assert {name: rule.value for name, rule in rules.items()} == {
            "logprobs": True,
            "return_meta_info": True,
            "no_stop_trim": False,
            "return_routed_experts": False,
            "return_indexer_topk": False,
            "lora_path": None,
            "input_ids": None,
            "routed_experts_start_len": None,
            "logprob_start_len": None,
        }
        assert {name for name, rule in rules.items() if rule.strict} == {
            "lora_path",
            "input_ids",
            "routed_experts_start_len",
            "logprob_start_len",
        }
        assert all(rule.why for rule in rules.values())

    def test_replay_flags_follow_the_launch_flags(self):
        rules = chat_request_rules(
            make_session_server_config(use_rollout_routing_replay=True, use_rollout_indexer_replay=True)
        )
        assert rules["return_routed_experts"].value is True
        assert rules["return_indexer_topk"].value is True

    def test_lora_path_follows_lora_rollout_enabled(self):
        assert chat_request_rules(make_session_server_config(lora_rank=8))["lora_path"].value == LORA_ADAPTER_NAME
        assert chat_request_rules(make_session_server_config(lora_adapter_path="/a"))["lora_path"].value == (
            LORA_ADAPTER_NAME
        )
        assert (
            chat_request_rules(make_session_server_config(lora_rank=8, lora_train_only=True))["lora_path"].value
            is None
        )
