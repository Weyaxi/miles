"""Which fields of the outbound chat request the session server decides, and how.

``chat_request_rules(config)`` is the one table: every field named there is
the server's, under one of two rules.  ``server_first`` replaces a differing
client value and logs it; ``server_strict`` rejects it with HTTP 400.  A field
named nowhere is the client's and is forwarded as sent (sampling parameters,
``model``, ``messages``, ``tools``, unknown keys).  ``chat_template_kwargs``
is not in this table: the TITO tokenizer decides it per request
(``TITOTokenizer.for_request``).
"""

import logging
from dataclasses import dataclass
from typing import Any

from miles.rollout.session.config import SessionServerConfig
from miles.rollout.session.errors import MessageValidationError
from miles.utils.lora import LORA_ADAPTER_NAME, lora_rollout_enabled

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Rule:
    """The server's ``value`` for one field and what a differing client value
    does: a warning (``strict=False``) or a 400 (``strict=True``).  ``value=None``
    means the server sets nothing here, so the field stays off the wire.
    ``why`` is the one-line reason that goes into the warning or error text."""

    value: Any
    strict: bool
    why: str


def server_first(value: Any, *, why: str) -> Rule:
    """The server's value wins; a differing client value is replaced and logged."""
    return Rule(value, strict=False, why=why)


def server_strict(value: Any, *, why: str) -> Rule:
    """The server's value wins; a differing client value is rejected with 400."""
    return Rule(value, strict=True, why=why)


def chat_request_rules(config: SessionServerConfig) -> dict[str, Rule]:
    """The fields of an outbound ``/v1/chat/completions`` body the session server decides."""
    return {
        # TITO needs these on every request: agent-side overrides would break token accumulation.
        "logprobs": server_first(True, why="TITO reads meta_info.output_token_logprobs"),
        "return_meta_info": server_first(True, why="wraps output_token_logprobs in choice.meta_info"),
        "no_stop_trim": server_first(False, why="stop-token text is trimmed; token ids come from logprobs"),
        # R3 replay follows the launch flags, on or off.
        "return_routed_experts": server_first(
            bool(config.use_rollout_routing_replay), why="follows --use-rollout-routing-replay"
        ),
        "return_indexer_topk": server_first(
            bool(config.use_rollout_indexer_replay), why="follows --use-rollout-indexer-replay"
        ),
        # A client setting these has passed out-of-scope information: fail loud.
        "lora_path": server_strict(
            LORA_ADAPTER_NAME if lora_rollout_enabled(config) else None,
            why="the served adapter is selected by training",
        ),
        "input_ids": server_strict(None, why="TITO token ids are rendered by the session server"),
        "routed_experts_start_len": server_strict(None, why="R3 offsets are computed by the session server"),
        "logprob_start_len": server_strict(None, why="not supported on the session chat path"),
    }


def apply_rules(client: dict[str, Any], rules: dict[str, Rule]) -> dict[str, Any]:
    """The outbound body: client fields as sent, ruled fields at the server's value.

    Client key order is kept and ruled fields the client did not send follow.
    A client value of ``None`` counts as not sent.
    """
    out: dict[str, Any] = {}
    for name in dict.fromkeys([*client, *rules]):
        rule = rules.get(name)
        if rule is None:
            out[name] = client[name]
            continue
        sent = client.get(name)
        if sent is not None and sent != rule.value:
            if rule.strict:
                raise MessageValidationError(f"{name}={sent!r} is not accepted: {rule.why}")
            logger.warning("%s=%r from the client replaced by %r: %s", name, sent, rule.value, rule.why)
        if rule.value is not None:
            out[name] = rule.value
    return out
