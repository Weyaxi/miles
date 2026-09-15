"""Portable LoRA weight transfer: stage the adapter, then tell engines to load it.

Miles has two transports that declare ``supports_lora``: ``broadcast`` (an NCCL
collective spanning trainer ranks and engines) and ``cuda_ipc`` (same host only).
Both are bound to one vendor, and cuda_ipc to one host, so neither can carry an
adapter from a trainer on NVIDIA GPUs to engines on AMD GPUs -- NCCL and RCCL do
not interoperate. The transports that DO work across hosts, ``p2p`` and
``disk-delta``, both ``assert lora_rank <= 0``.

``load_lora_adapter_from_tensors`` looks like the portable escape hatch but is
not: SGLang's ``MultiprocessingSerializer`` emits a fixed-size shared-memory
handle rather than bytes (a 64 B tensor serialises to 464 chars, a 2 MB tensor to
472), so the data never travels. It also has no caller in the repo.

What is portable is a file plus a path-based load, which the engine already
exposes: ``POST /load_lora_adapter {"lora_name", "lora_path", "pinned"}``.

So rank 0 gathers the adapter, writes safetensors to --update-weight-disk-dir,
and POSTs the path. As with disk-delta, that directory is expected to be visible
to the engines; --custom-update-weight-post-write-path covers the case where it
is not. Nothing but HTTP and a filesystem crosses
the boundary, so the trainer and the engines need not share a vendor, an
interconnect, or a host. Only sane for LoRA: a full-weight sync this way would
move the whole model every step.

``use_weight_update_session = False`` because this protocol owns registration and
reload itself; it does not need the session frame's pause/resume around a
collective that is not happening.
"""
from __future__ import annotations

import contextlib
import json
import logging
import os
import time
from argparse import Namespace
from collections.abc import Callable, Sequence

import torch
import torch.distributed as dist

from miles.backends.sglang_utils.sglang_api_client import SGLangApiClient
from miles.backends.training_utils.parallel import ParallelState
from miles.backends.training_utils.weight_update.hf_weight_iterator import WeightUpdatePlacement
from miles.backends.training_utils.weight_update.protocol import WeightTransferProtocol

logger = logging.getLogger(__name__)


class UpdateWeightHttpLora(WeightTransferProtocol):
    """Stage the adapter to a directory, then POST its path to every engine."""

    # gather_pp so one rank holds the whole adapter; tp/ep are always gathered.
    required_placement = WeightUpdatePlacement(gather_pp=True)
    supports_lora = True
    # Engines already hold the base; a resync request here is a misconfiguration,
    # not something to silently do at 70 GB per step.
    needs_base_resync_for_lora = False
    use_weight_update_session = False

    def __init__(self, args: Namespace) -> None:
        super().__init__(args)
        self._buf: dict[str, torch.Tensor] = {}
        self._stage_root: str = getattr(args, "update_weight_disk_dir", None) or "/tmp/miles_lora_sync"
        self._post_write_hook: Callable | None = None
        if getattr(args, "custom_update_weight_post_write_path", None):
            from miles.utils.function_registry import load_function

            self._post_write_hook = load_function(args.custom_update_weight_post_write_path)

    # ------------------------------------------------------------------ setup

    def connect(
        self,
        rollout_engines: Sequence[SGLangApiClient],
        engine_gpu_counts: Sequence[int] | None,
        engine_gpu_offsets: Sequence[int] | None,
        parallel_state: ParallelState,
        placement: WeightUpdatePlacement,
        selector: str,
    ) -> None:
        self.rollout_engines = rollout_engines
        assert placement.gather_pp, "http-lora needs the full adapter on one rank"
        # One sender: the adapter is fully gathered, so fanning the upload across
        # ranks would write the same bytes N times.
        self.is_sender = dist.get_rank() == 0 if dist.is_initialized() else True
        self.group_name = "miles-http-lora"
        self._buf.clear()

    # ------------------------------------------------------------------ stream

    def send_bucket(self, bucket: list[tuple[str, torch.Tensor]]) -> None:
        """Buffer adapter tensors. Names are ``{lora_name}:{hf_key}``; anything
        else is a base weight and is dropped loudly rather than written into an
        adapter file where it would be nonsense."""
        for name, tensor in bucket:
            if ":" not in name:
                logger.warning("http-lora: dropping non-adapter tensor %r", name)
                continue
            self._buf[name] = tensor.detach().to("cpu", copy=True)

    # --------------------------------------------------------------- finalize

    def finalize(self, weight_version: int) -> None:
        if not self.is_sender:
            return
        if not self._buf:
            logger.warning("http-lora: nothing buffered at finalize; skipping publish")
            return

        by_adapter: dict[str, dict[str, torch.Tensor]] = {}
        for name, tensor in self._buf.items():
            lora_name, hf_key = name.split(":", 1)
            by_adapter.setdefault(lora_name, {})[hf_key] = tensor
        self._buf.clear()

        for lora_name, tensors in by_adapter.items():
            t0 = time.time()
            local_dir = self._write_adapter(lora_name, weight_version, tensors)
            nbytes = sum(t.numel() * t.element_size() for t in tensors.values())
            t_write = time.time() - t0

            t0 = time.time()
            # Same escape hatch disk-delta uses: when the staging dir is not
            # genuinely shared with the engine hosts, the operator supplies
            # --custom-update-weight-post-write-path to replicate it (rsync, an
            # object store, whatever their cluster uses). Inventing a second
            # copy mechanism here would duplicate that contract.
            if self._post_write_hook is not None:
                self._post_write_hook(self.args, local_dir, list(self.rollout_engines or []))
            t_ship = time.time() - t0

            t0 = time.time()
            self._load_everywhere(lora_name, local_dir)
            t_load = time.time() - t0

            self.update_weight_metrics.update({
                "http_lora/bytes": float(nbytes), "http_lora/write_s": t_write,
                "http_lora/ship_s": t_ship, "http_lora/load_s": t_load,
            })
            logger.info(
                "http-lora: %s v%d  %.0f MB  write %.1fs  ship %.1fs  load %.1fs  -> %d engine(s)",
                lora_name, weight_version, nbytes / 1e6, t_write, t_ship, t_load,
                len(self.rollout_engines or []),
            )

    # ---------------------------------------------------------------- helpers

    def _write_adapter(self, lora_name: str, version: int, tensors: dict[str, torch.Tensor]) -> str:
        from safetensors.torch import save_file

        # Versioned dir: an engine may still be reading the previous one, and
        # overwriting in place is how a half-written adapter gets loaded.
        d = os.path.join(self._stage_root, f"{lora_name}_v{version}")
        os.makedirs(d, exist_ok=True)
        tmp = os.path.join(d, ".adapter_model.safetensors.tmp")
        save_file(tensors, tmp, metadata={"format": "pt"})
        os.replace(tmp, os.path.join(d, "adapter_model.safetensors"))  # atomic publish

        cfg = self._adapter_config(tensors)
        with open(os.path.join(d, "adapter_config.json"), "w") as fh:
            json.dump(cfg, fh, indent=2)

        # Consumed by a different process and often a different user (trainer as
        # root in a container, shipper not). safetensors writes 0600, so without
        # this the reader gets "Permission denied" on a file that is plainly there.
        os.chmod(d, 0o755)
        for f in os.listdir(d):
            with contextlib.suppress(OSError):
                os.chmod(os.path.join(d, f), 0o644)
        return d

    def _adapter_config(self, tensors: dict[str, torch.Tensor]) -> dict:
        """PEFT-shaped config. Target modules come from the tensors we actually
        hold, not the training args: claiming a module whose weights are absent is
        how engines apply uninitialised slots."""
        args = self.args
        targets = sorted({
            k.split(".lora_")[0].rsplit(".", 1)[-1]
            for k in tensors
            if ".lora_" in k
        })
        # Short names match by name, so "down_proj"/"up_proj" also select the
        # fused MoE experts. Without this exclusion an adapter carrying no expert
        # weights is rejected for the missing tensors, reporting only expert
        # paths -- which reads as stale data and is not.
        has_expert_weights = any(".mlp.experts." in k for k in tensors)
        exclude = None if has_expert_weights else r".*mlp\.experts.*"
        return {
            "peft_type": "LORA",
            "task_type": "CAUSAL_LM",
            "r": int(getattr(args, "lora_rank", 0)),
            "lora_alpha": int(getattr(args, "lora_alpha", 0)),
            "lora_dropout": float(getattr(args, "lora_dropout", 0.0) or 0.0),
            "bias": "none",
            "inference_mode": True,
            "target_modules": targets,
            "exclude_modules": exclude,
        }

    def _load_everywhere(self, lora_name: str, remote_dir: str) -> None:
        """Register the staged adapter on every engine."""
        import httpx

        for eng in self.rollout_engines or []:
            base = eng.server_url.rstrip("/")
            try:
                r = httpx.post(
                    base + "/load_lora_adapter",
                    json={"lora_name": lora_name, "lora_path": remote_dir, "pinned": False},
                    timeout=1800.0,
                )
            except Exception as ex:  # noqa: BLE001
                raise RuntimeError(f"http-lora: load failed on {base}: {type(ex).__name__}: {ex}") from ex
            # A re-registered name is not an error: the engine already has it.
            if r.status_code != 200 and "already" not in r.text.lower():
                raise RuntimeError(f"http-lora: load failed on {base}: HTTP {r.status_code} {r.text[:160]}")
