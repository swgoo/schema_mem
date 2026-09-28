"""Train the DB task from scratch using the root-only source release.

SchemaMem: full-history initialization -> alternating transactions -> maintenance.
Mamba-3/Gated DeltaNet: recurrent transactions -> recurrent maintenance.

Both curricula run in one process; no pretrained weights or external datasets
are supplied or required. Task, baseline, training and evaluation code is local
to this file. Project dependencies are limited to the root model/config files.
Output directories are created at runtime. See train_db.md for usage.
"""

import argparse
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from functools import partial
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parent
FAMILIES = ("schemamem", "mamba3", "gdn")
BIT_WIDTHS = (3, 4, 5)
SEEDS = (42, 43, 44)
PROTOCOL = "db-paper-continuous-v1"


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-family", choices=FAMILIES, default="schemamem")
    parser.add_argument("--address-bits", type=int, choices=BIT_WIDTHS, default=3)
    parser.add_argument("--value-bits", type=int, choices=BIT_WIDTHS, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--eval-samples", type=int, default=512)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--transaction-step-limit", type=int, default=1_800_000)
    parser.add_argument("--transaction-stage-limit", type=int, default=300_000)
    parser.add_argument("--maintenance-step-limit", type=int, default=360_000)
    parser.add_argument("--maintenance-stage-limit", type=int, default=120_000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs/db_paper")
    parser.add_argument("--project", default="SchemaMem-DB-Paper")
    parser.add_argument("--entity", default=None, help="Use the configured W&B account by default.")
    parser.add_argument("--mode", choices=("online", "offline", "disabled"), default="online")
    parser.add_argument("--wandb-sweep", action="store_true", help="Read model/bit-width/seed from a W&B agent.")
    parser.add_argument("--dry-run", action="store_true", help="Print the complete protocol without training or W&B access.")
    parser.add_argument("--print-sweep", action="store_true", help="Print a W&B grid config for all 81 runs; creates no sweep.")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Run a small CPU SchemaMem check; no W&B, saved weights, or external test files.")
    return parser


def training_protocol(args):
    """One final target per family, fixed before either curriculum starts."""
    if args.model_family not in FAMILIES:
        raise ValueError("Only the paper's SchemaMem, Mamba-3, and Gated DeltaNet are supported")
    if args.address_bits not in BIT_WIDTHS or args.value_bits not in BIT_WIDTHS:
        raise ValueError("The paper grid uses A3/A4/A5 and V3/V4/V5")
    for name in ("batch_size", "eval_batch_size", "eval_samples", "eval_every",
                 "transaction_step_limit", "transaction_stage_limit",
                 "maintenance_step_limit", "maintenance_stage_limit"):
        if getattr(args, name) < 1:
            raise ValueError(f"{name} must be positive")
    schema = args.model_family == "schemamem"
    target = .95 if schema else .99
    addresses = 2 ** args.address_bits
    common = dict(model_family=args.model_family, seed=args.seed, device=args.device,
                  batch_size=args.batch_size, eval_samples=args.eval_samples,
                  eval_every=args.eval_every, final_accuracy=target)
    transactions = dict(
        **common, addresses=addresses, values=2 ** args.value_bits,
        address_bits=args.address_bits, value_bits=args.value_bits,
        bitwise_output=True, hidden_size=32, layers=3, slots=176,
        intermediate_size=16, kv_heads=1, mamba_state_size=80,
        all_trace_layers=True, input_bias=True, commit_length=96,
        stages=[addresses // 4, addresses // 2, addresses], accuracy=.80,
        pretrain_attention=schema, pretrain_accuracy=.80,
        alternate_commit_batches=schema, lr=3e-4,
        default_queries=None, no_commit=False, no_chunk=False,
        steps=args.transaction_step_limit, max_stage_steps=args.transaction_stage_limit,
    )
    maintenance = dict(
        **common, eval_batch_size=args.eval_batch_size, eval_seed=20261001,
        stage_delays=[96, 192, 384], accuracy=.90, passes_required=2,
        lr=1e-4, warmup_steps=500, weight_decay=.01, max_grad_norm=1.,
        legacy_fraction=.20, full_history_every=4 if schema else 0,
        filler_weight=.10, heldout_delays=[768, 1536],
        steps=args.maintenance_step_limit, max_stage_steps=args.maintenance_stage_limit,
    )
    return dict(protocol=PROTOCOL, transactions=transactions, maintenance=maintenance)


def sweep_config(args):
    """JSON is also valid YAML; write this output to a W&B sweep file."""
    # Agent assignments are read from run.config, not ambiguous underscore CLI flags.
    command = ["${env}", "${interpreter}", "${program}", "--wandb-sweep"]
    for name in ("batch_size", "eval_batch_size", "eval_samples", "eval_every",
                 "transaction_step_limit", "transaction_stage_limit",
                 "maintenance_step_limit", "maintenance_stage_limit", "device", "output_root"):
        command.extend(["--" + name.replace("_", "-"), str(getattr(args, name))])
    config = dict(
        name=PROTOCOL, method="grid", program="train_db.py", command=command,
        project=args.project,
        metric=dict(name="maintenance/val/min_accuracy", goal="maximize"),
        parameters=dict(model_family=dict(values=list(FAMILIES)),
                        address_bits=dict(values=list(BIT_WIDTHS)),
                        value_bits=dict(values=list(BIT_WIDTHS)),
                        seed=dict(values=list(SEEDS))),
    )
    if args.entity:
        config["entity"] = args.entity
    return config


def file_digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def provenance():
    packages = {}
    for package in ("torch", "transformers", "numpy", "wandb", "mamba-ssm",
                    "flash-linear-attention", "fla-core", "causal-conv1d"):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    files = ("train_db.py", "modeling_schema_mem_db.py", "modeling_schema_mem.py",
             "configuration_schema_mem.py")
    return dict(packages=packages, source_sha256={p: file_digest(ROOT / p) for p in files})


# Task: defaults live only in the label oracle, never in model inputs.
GROUPS = ("retained", "updated", "default", "filler")


@dataclass
class Episode:
    keys: torch.Tensor
    values: torch.Tensor
    labels: torch.Tensor
    groups: torch.Tensor
    final: torch.Tensor
    segments: list


def transaction_batch(batch_size, active, addresses, values, rng, device="cpu"):
    count, half = active, active // 2
    keys = np.stack([rng.choice(addresses, size=count, replace=False) for _ in range(batch_size)])
    contents = keys % values
    chosen = np.stack([rng.permutation(count)[:half] for _ in range(batch_size)])
    rows = np.arange(batch_size)[:, None]
    replacement = (contents[rows, chosen] + rng.integers(1, values, size=(batch_size, half))) % values
    order = np.stack([rng.permutation(count) for _ in range(batch_size)])
    address_ids = np.concatenate([keys[rows, chosen], keys[rows, order]], axis=1)
    value_ids = np.concatenate([replacement, np.full_like(keys, -1)], axis=1)
    contents[rows, chosen] = replacement
    labels = np.full_like(value_ids, -100)
    labels[:, half:] = contents[rows, order]
    changed = np.zeros_like(contents, dtype=bool)
    changed[rows, chosen] = True
    groups = np.full_like(labels, -1)
    groups[:, half:] = np.where(changed[rows, order], 1, 2)
    final = labels != -100
    tensors = [torch.from_numpy(x).to(device) for x in (address_ids, value_ids, labels, groups, final)]
    return Episode(*tensors, [(0, half), (half, half + count)])


def maintenance_batch(batch_size, scenario, delay, rng, device="cpu", addresses=8, values=8):
    if scenario not in ("legacy", "retention", "interference", "overwrite") or delay < 0:
        raise ValueError("Unknown scenario or negative delay")
    if addresses < 4 or addresses % 4 or values < 3:
        raise ValueError("Addresses must be divisible by four; at least three values are required")
    quarter, half = addresses // 4, addresses // 2
    order = np.stack([rng.permutation(addresses) for _ in range(batch_size)])
    rows = np.arange(batch_size)[:, None]
    defaults = np.arange(addresses) % values
    db = np.broadcast_to(defaults, (batch_size, addresses)).copy()
    seen, latest = np.zeros_like(db, dtype=bool), np.zeros_like(db, dtype=bool)
    parts, segments, offset, previous_read = [], [], 0, False

    def append(keys, vals, group=None, final=False):
        nonlocal offset, previous_read
        if not keys.shape[1]:
            return
        labels = np.full_like(vals, -100) if group is None else db[rows, keys].copy()
        groups = (np.full_like(vals, -1) if group is None else
                  np.full_like(vals, group) if np.isscalar(group) else group)
        parts.append((keys.copy(), vals.copy(), labels, groups, np.full_like(vals, final, dtype=bool)))
        is_read = bool(np.all(vals == -1))
        # Filler and final queries are one read stream: no extra read/read commit.
        if is_read and previous_read:
            segments[-1] = (segments[-1][0], offset + keys.shape[1])
        else:
            segments.append((offset, offset + keys.shape[1]))
        previous_read = is_read
        offset += keys.shape[1]

    def write(keys):
        vals = rng.integers(values, size=keys.shape)
        invalid = (vals == db[rows, keys]) | (vals == defaults[keys])
        while invalid.any():
            vals[invalid] = rng.integers(values, size=int(invalid.sum()))
            invalid = (vals == db[rows, keys]) | (vals == defaults[keys])
        db[rows, keys], seen[rows, keys] = vals, True
        latest[:] = False
        latest[rows, keys] = True
        append(keys, vals)

    def query_all(final=False, all_written_retained=False):
        keys = np.stack([rng.permutation(addresses) for _ in range(batch_size)])
        groups = np.where(~seen[rows, keys], 2,
                          np.where(latest[rows, keys] & (not all_written_retained), 1, 0))
        append(keys, np.full_like(keys, -1), groups, final)

    write(order[:, :quarter if scenario == "interference" else half])
    if scenario in ("interference", "overwrite"):
        query_all()
        write(order[:, quarter:half] if scenario == "interference" else order[:, :quarter])
    if delay and scenario != "legacy":
        keys = np.take_along_axis(order[:, half:], rng.integers(half, size=(batch_size, delay)), axis=1)
        append(keys, np.full_like(keys, -1), 3)
    query_all(final=True, all_written_retained=scenario == "retention")
    tensors = [torch.from_numpy(np.concatenate([p[i] for p in parts], axis=1)).to(device) for i in range(5)]
    return Episode(*tensors, segments)


class Mamba3Block(nn.Module):
    def __init__(self, hidden_size, state_size, layer_idx):
        super().__init__()
        from transformers.models.mamba2.modeling_mamba2 import Mamba2RMSNorm
        from mamba_ssm.modules.mamba3 import Mamba3
        self.norm = Mamba2RMSNorm(hidden_size, eps=1e-5)
        self.mixer = Mamba3(d_model=hidden_size, d_state=state_size, expand=2,
                            headdim=16, ngroups=2, is_mimo=False, chunk_size=64,
                            layer_idx=layer_idx)

    def forward(self, hidden):
        return hidden.float() + self.mixer(self.norm(hidden.to(self.norm.weight.dtype)))


class Mamba3Backbone(nn.Module):
    def __init__(self, hidden_size, layers, state_size):
        super().__init__()
        from transformers.models.mamba2.modeling_mamba2 import Mamba2RMSNorm
        from mamba_ssm.models.mixer_seq_simple import _init_weights
        self.layers = nn.ModuleList([Mamba3Block(hidden_size, state_size, i) for i in range(layers)])
        self.norm_f = Mamba2RMSNorm(hidden_size, eps=1e-5)
        self.apply(partial(_init_weights, n_layer=layers))

    def forward(self, hidden):
        for layer in self.layers:
            hidden = layer(hidden)
        return self.norm_f(hidden)


class DBModel(nn.Module):
    """Binary input/output adapters around the three paper architectures."""

    def __init__(self, config, pretrain=False):
        super().__init__()
        self.config = dict(config)
        self.model_family = config["model_family"]
        self.addresses, self.values = config["addresses"], config["values"]
        self.address_bits, self.value_bits = config["address_bits"], config["value_bits"]
        h, layers = config["hidden_size"], config["layers"]
        self.input_proj = nn.Linear(self.address_bits + self.value_bits + 1, h, bias=True)
        if self.model_family == "schemamem":
            from modeling_schema_mem_db import SchemaMemDBConfig, SchemaMemDBModel
            window = self.addresses * 3 // 2 + 1 if pretrain else config["commit_length"]
            backbone_config = SchemaMemDBConfig(
                vocab_size=1, hidden_size=h, num_hidden_layers=layers,
                intermediate_size=config["intermediate_size"], num_attention_heads=4,
                num_key_value_heads=config["kv_heads"], head_dim=h // 4,
                schema_size=config["slots"], schema_layer_indices=list(range(layers)),
                sliding_window=window, max_position_embeddings=4096, attention_dropout=0.,
                use_cache=False, bos_token_id=None, eos_token_id=None, pad_token_id=None)
            backbone_config._attn_implementation = "sdpa"
            self.backbone = SchemaMemDBModel(backbone_config)
            self.backbone.embed_tokens = nn.Identity()
        elif self.model_family == "mamba3":
            self.backbone = Mamba3Backbone(h, layers, config["mamba_state_size"])
        elif self.model_family == "gdn":
            from fla.models.gated_deltanet import GatedDeltaNetConfig, GatedDeltaNetModel
            backbone_config = GatedDeltaNetConfig(
                vocab_size=1, hidden_size=h, num_hidden_layers=layers,
                head_dim=48, num_heads=2, expand_v=1.,
                intermediate_size=config["intermediate_size"], attn=None, attn_mode="chunk",
                use_gate=True, use_short_conv=True, conv_size=4, allow_neg_eigval=False,
                use_cache=False, pad_token_id=None, bos_token_id=None, eos_token_id=None,
                fuse_norm=False, fuse_swiglu=False)
            self.backbone = GatedDeltaNetModel(backbone_config)
            self.backbone.set_input_embeddings(nn.Identity())
        else:
            raise ValueError(self.model_family)
        self.classifier = nn.Linear(h, self.value_bits, bias=False)

    def encode(self, keys, values):
        valid = (values >= 0).unsqueeze(-1)
        abits = (keys.unsqueeze(-1) >> torch.arange(self.address_bits, device=keys.device)) & 1
        vbits = ((values.clamp_min(0).unsqueeze(-1) >> torch.arange(self.value_bits, device=keys.device)) & 1) * valid
        return torch.cat([abits, vbits, valid], dim=-1).to(self.input_proj.weight.dtype)

    def forward(self, episode, full_history=False, zero_s=False):
        inputs = self.input_proj(self.encode(episode.keys, episode.values))
        if self.model_family == "schemamem":
            from modeling_schema_mem_db import SchemaCache

            class DBState(SchemaCache):
                def commit(self, layer_idx=None):
                    changed = super().commit(layer_idx)
                    if zero_s:
                        for idx in changed:
                            self.states[idx] = torch.zeros_like(self.states[idx])
                    return changed

            cache = DBState()
            old_window = self.backbone.config.sliding_window
            try:
                if full_history:
                    self.backbone.config.sliding_window = max(old_window, inputs.shape[1] + 1)
                    hidden = self.backbone(inputs_embeds=inputs, schema_cache=cache).last_hidden_state
                else:
                    parts = []
                    for i, (start, stop) in enumerate(episode.segments):
                        parts.append(self.backbone(inputs_embeds=inputs[:, start:stop], schema_cache=cache).last_hidden_state)
                        if i < len(episode.segments) - 1:
                            cache.commit()  # Differentiable; no same-layer state enters the writer.
                    hidden = torch.cat(parts, dim=1)
            finally:
                self.backbone.config.sliding_window = old_window
        else:
            if full_history or zero_s:
                raise ValueError("Full-history mode and zero-S apply only to SchemaMem")
            # A continuous scan, without resets at SchemaMem commit boundaries.
            hidden = (self.backbone(inputs) if self.model_family == "mamba3" else
                      self.backbone(inputs_embeds=inputs, use_cache=False).last_hidden_state)
        return self.classifier(hidden)


def resource_counts(model):
    layout = {}
    if model.model_family == "schemamem":
        for i in range(model.config["layers"]):
            layout[f"{i}/schema"] = dict(elements=model.config["slots"] * model.config["hidden_size"])
    elif model.model_family == "mamba3":
        for i, layer in enumerate(model.backbone.layers):
            tensors = layer.mixer.allocate_inference_cache(1, 1, device="cpu")
            for name, tensor in zip(("angle", "ssm", "key", "value"), tensors):
                layout[f"{i}/{name}"] = dict(elements=tensor.numel(), shape=list(tensor.shape))
    else:
        for i, layer in enumerate(model.backbone.layers):
            attn = layer.attn
            layout[f"{i}/recurrent"] = dict(elements=attn.num_v_heads * attn.head_k_dim * attn.head_v_dim)
            for name, width in (("q", attn.key_dim), ("k", attn.key_dim), ("v", attn.value_dim)):
                layout[f"{i}/conv_{name}"] = dict(elements=width * attn.conv_size)
    return dict(parameters=sum(p.numel() for p in model.parameters()),
                recurrent_state_elements=sum(x["elements"] for x in layout.values()),
                recurrent_state_layout=layout)


def amp(device):
    return torch.autocast("cuda", dtype=torch.bfloat16) if str(device).startswith("cuda") else nullcontext()


def bit_scores(logits, episode):
    bits = ((episode.labels.clamp_min(0)[..., None] >> torch.arange(logits.shape[-1], device=logits.device)) & 1).bool()
    matches = (logits.float() >= 0) == bits
    losses = F.binary_cross_entropy_with_logits(logits.float(), bits.float(), reduction="none").mean(-1)
    return losses, matches.all(-1), matches.float().mean(-1)


def query_loss(logits, episode, balanced=False, filler_weight=.1):
    losses, _, _ = bit_scores(logits, episode)
    if not balanced:
        return losses[episode.labels != -100].mean()
    terms, weights = [], []
    for group in range(4):
        mask = episode.groups == group
        weight = filler_weight if group == 3 else 1.
        if mask.any() and weight:
            terms.append(losses[mask].mean() * weight)
            weights.append(weight)
    return sum(terms) / sum(weights)


def accumulate_stats(totals, name, mask, scores):
    if mask.any():
        loss, exact, bits = scores
        row = np.array([loss[mask].sum().item(), exact[mask].sum().item(),
                        bits[mask].sum().item(), mask.sum().item()])
        totals[name] = totals.get(name, np.zeros(4)) + row


def mean_stats(totals, prefix):
    return {f"{prefix}/{name}/{metric}": float(row[i] / row[3])
            for name, row in totals.items()
            for i, metric in enumerate(("ce", "accuracy", "bit_accuracy"))}


@torch.no_grad()
def evaluate_transactions(model, settings, active, full_history=False, zero_s=False):
    was_training = model.training
    model.eval()
    totals = {}
    rng = np.random.default_rng(settings["seed"] + 100000 + active)
    try:
        for start in range(0, settings["eval_samples"], settings["batch_size"]):
            episode = transaction_batch(min(settings["batch_size"], settings["eval_samples"] - start),
                                        active, model.addresses, model.values, rng, settings["device"])
            with amp(settings["device"]):
                logits = model(episode, full_history, zero_s)
            if not torch.isfinite(logits).all():
                raise FloatingPointError("Nonfinite validation output")
            scores = bit_scores(logits, episode)
            for name, mask in (("all", episode.final), ("preserved", episode.groups == 2),
                               ("rewritten", episode.groups == 1)):
                accumulate_stats(totals, name, mask, scores)
    finally:
        model.train(was_training)
    return mean_stats(totals, "val_zero_S" if zero_s else "val")


def stage_cases(stage, delay):
    cases = [("legacy", 0), ("retention", delay)]
    if stage >= 1:
        cases += [("interference", delay), ("overwrite", delay)]
    return cases


@torch.no_grad()
def evaluate_maintenance(model, settings, stage, max_delay, prefix="val"):
    was_training = model.training
    model.eval()
    result, aggregate = {}, {}
    try:
        for scenario, delay in stage_cases(stage, max_delay):
            index = ("legacy", "retention", "interference", "overwrite").index(scenario)
            rng = np.random.default_rng(settings["eval_seed"] + 1009 * index + delay)
            totals = {}
            for start in range(0, settings["eval_samples"], settings["eval_batch_size"]):
                episode = maintenance_batch(min(settings["eval_batch_size"], settings["eval_samples"] - start),
                                            scenario, delay, rng, settings["device"], model.addresses, model.values)
                with amp(settings["device"]):
                    logits = model(episode)
                if not torch.isfinite(logits).all():
                    raise FloatingPointError("Nonfinite validation output")
                scores = bit_scores(logits, episode)
                for group, name in enumerate(GROUPS):
                    accumulate_stats(totals, name, (episode.groups == group) & episode.final, scores)
            result.update(mean_stats(totals, f"{prefix}/{scenario}"))
            for name, row in totals.items():
                aggregate[name] = aggregate.get(name, np.zeros(4)) + row
        result.update(mean_stats(aggregate, prefix))
        result[f"{prefix}/min_accuracy"] = min(v for k, v in result.items() if k.endswith("/accuracy"))
        return result
    finally:
        model.train(was_training)


class PhaseLog:
    """Local JSONL plus one monotonically indexed W&B run for both curricula."""

    def __init__(self, run, root, phase, offset=0):
        self.run, self.phase, self.offset = run, phase, offset
        self.directory = root / phase
        self.directory.mkdir()

    def log(self, row):
        with (self.directory / "metrics.jsonl").open("a") as handle:
            handle.write(json.dumps(row) + "\n")
        record = {f"{self.phase}/{key}": value for key, value in row.items()}
        record.update(step=self.offset + row["step"], pipeline_stage=self.phase)
        self.run.log(record)
        print(json.dumps(record), flush=True)

    def finish(self, report):
        (self.directory / "results.json").write_text(json.dumps(report, indent=2) + "\n")
        self.run.summary.update({f"{self.phase}/{key}": value for key, value in report.items()})


def save_checkpoint(path, model, optimizer, rng, **state):
    payload = dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                   rng=rng.bit_generator.state, torch_rng=torch.get_rng_state(),
                   cuda_rng=torch.cuda.get_rng_state_all() if next(model.parameters()).is_cuda else [],
                   **state)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


@contextmanager
def stop_signals():
    stopped = {"signal": None}
    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}

    def request_stop(sig, frame):
        stopped["signal"] = sig

    try:
        for sig in previous:
            signal.signal(sig, request_stop)
        yield stopped
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def optimizer_step(model, optimizer, episode, settings, full_history, balanced=False):
    optimizer.zero_grad(set_to_none=True)
    with amp(settings["device"]):
        logits = model(episode, full_history=full_history)
        loss = query_loss(logits, episode, balanced, settings.get("filler_weight", .1))
    if not torch.isfinite(loss):
        raise FloatingPointError("Nonfinite training loss")
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
    optimizer.step()
    return float(loss.detach()), float(grad_norm)


def train_transactions(model, settings, logger, stopped):
    """Full-history initialization, if applicable, then the recurrent curriculum."""
    rng = np.random.default_rng(settings["seed"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["lr"], weight_decay=.01)
    phase = "attention_pretrain" if settings["pretrain_attention"] else "recurrent"
    phase_start = stage = stage_start = passes = step = 0
    best, counters = {}, dict(full_history_steps=0, recurrent_steps=0, tokens=0)
    reason, started = "step_limit", time.monotonic()

    def save(name="last.pt"):
        save_checkpoint(logger.directory / name, model, optimizer, rng,
                        config=settings, run_id=logger.run.id, phase=phase,
                        next_step=step + 1, stage_index=stage, stage_start=stage_start,
                        phase_start=phase_start, passes=passes, best=best, counters=counters)

    for step in range(1, settings["steps"] + 1):
        active = settings["stages"][stage]
        episode = transaction_batch(settings["batch_size"], active, model.addresses,
                                    model.values, rng, settings["device"])
        full = phase == "attention_pretrain" or (
            settings["alternate_commit_batches"] and (step - phase_start) % 2 == 1)
        loss, grad = optimizer_step(model, optimizer, episode, settings, full)
        counters["full_history_steps" if full else "recurrent_steps"] += 1
        counters["tokens"] += episode.keys.numel()
        if step == 1 or step % 50 in (0, 1):
            logger.log(dict(step=step, **{"phase/name": phase, "stage/active": active,
                "train/loss": loss, "train/grad_norm": grad, "train/full_history": full,
                "train/lr": settings["lr"], "train/elapsed_seconds": time.monotonic() - started}, **counters))
        if step % settings["eval_every"] == 0:
            metrics = evaluate_transactions(model, settings, active, phase == "attention_pretrain")
            if model.model_family == "schemamem" and phase == "recurrent":
                metrics.update(evaluate_transactions(model, settings, active, zero_s=True))
            threshold = (settings["pretrain_accuracy"] if phase == "attention_pretrain" else
                         settings["final_accuracy"] if stage == len(settings["stages"]) - 1 else
                         settings["accuracy"])
            score = min(metrics["val/preserved/accuracy"], metrics["val/rewritten/accuracy"])
            passes = passes + 1 if score >= threshold else 0
            logger.log(dict(step=step, **{"phase/name": phase, "stage/active": active,
                "stage/threshold": threshold, "stage/passes": passes}, **metrics, **counters))
            key = f"{phase}/{active}"
            if metrics["val/all/ce"] < best.get(key, float("inf")):
                best[key] = metrics["val/all/ce"]
                save(f"{phase}_best_{active}.pt")
            if passes >= 2:
                save(f"{phase}_completed_{active}.pt")
                stage += 1
                stage_start, passes = step, 0
                if stage == len(settings["stages"]):
                    if phase == "recurrent":
                        reason = "all_stages_passed"
                        break
                    save("attention_pretrain_final.pt")
                    # Rebuild masks with the recurrent window; preserve every weight.
                    torch.manual_seed(settings["seed"])
                    replacement = DBModel(settings).to(settings["device"])
                    replacement.load_state_dict(model.state_dict(), strict=True)
                    model = replacement
                    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["lr"], weight_decay=.01)
                    rng = np.random.default_rng(settings["seed"])
                    phase, phase_start, stage = "recurrent", step, 0
                    logger.log(dict(step=step, event="start_recurrent", optimizer_reset=True))
                else:
                    logger.log(dict(step=step, event="promoted", phase=phase, active=settings["stages"][stage]))
            save()
        if stopped["signal"]:
            reason = "interrupted"
            break
        if step - stage_start >= settings["max_stage_steps"]:
            reason = "stage_limit"
            break
    save()
    report = dict(step=step, stop_reason=reason, phase=phase, stage=stage,
                  curriculum_complete=reason == "all_stages_passed", **counters)
    logger.finish(report)
    return model, report


def train_maintenance(model, architecture, settings, source, logger, stopped):
    rng = np.random.default_rng(settings["seed"] + 300000)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["lr"], weight_decay=settings["weight_decay"])
    stage = stage_start = passes = step = 0
    best, counters = {}, dict(full_history_steps=0, recurrent_steps=0, tokens=0)
    reason, started = "step_limit", time.monotonic()

    def save(name="last.pt"):
        save_checkpoint(logger.directory / name, model, optimizer, rng,
                        config=dict(architecture, **settings), settings=settings, source=source,
                        run_id=logger.run.id, phase="recurrent", next_step=step + 1,
                        stage_index=stage, stage_start=stage_start, passes=passes, best=best, counters=counters)

    logger.log(dict(step=0, event="initial", stage=stage,
                    **evaluate_maintenance(model, settings, stage, settings["stage_delays"][stage])))
    save()
    for step in range(1, settings["steps"] + 1):
        max_delay = settings["stage_delays"][stage]
        if rng.random() < settings["legacy_fraction"]:
            scenario = "legacy"
        else:
            scenario = "retention" if stage == 0 else rng.choice(["retention", "interference", "overwrite"]).item()
        delay = 0 if scenario == "legacy" else int(rng.choice([0, 32] + settings["stage_delays"][:stage + 1]))
        episode = maintenance_batch(settings["batch_size"], scenario, delay, rng,
                                    settings["device"], model.addresses, model.values)
        full = settings["full_history_every"] > 0 and step % settings["full_history_every"] == 0
        lr = settings["lr"] * min(1., step / max(1, settings["warmup_steps"]))
        for group in optimizer.param_groups:
            group["lr"] = lr
        loss, grad = optimizer_step(model, optimizer, episode, settings, full, balanced=True)
        counters["full_history_steps" if full else "recurrent_steps"] += 1
        counters["tokens"] += episode.keys.numel()
        if step == 1 or step % 50 in (0, 1):
            logger.log(dict(step=step, stage=stage, max_delay=max_delay, scenario=scenario, delay=delay,
                **{"train/loss": loss, "train/grad_norm": grad, "train/full_history": full,
                   "train/lr": lr, "train/sequence_length": episode.keys.shape[1],
                   "train/elapsed_seconds": time.monotonic() - started}, **counters))
        if step % settings["eval_every"] == 0:
            metrics = evaluate_maintenance(model, settings, stage, max_delay)
            threshold = settings["final_accuracy"] if stage == len(settings["stage_delays"]) - 1 else settings["accuracy"]
            passes = passes + 1 if metrics["val/min_accuracy"] >= threshold else 0
            logger.log(dict(step=step, stage=stage, max_delay=max_delay,
                            threshold=threshold, passes=passes, **metrics, **counters))
            if metrics["val/min_accuracy"] > best.get(str(stage), -1):
                best[str(stage)] = metrics["val/min_accuracy"]
                save(f"best_stage_{stage}.pt")
            if passes >= settings["passes_required"]:
                save(f"completed_stage_{stage}.pt")
                stage += 1
                stage_start, passes = step, 0
                if stage == len(settings["stage_delays"]):
                    reason = "all_stages_passed"
                    break
                logger.log(dict(step=step, event="promoted", stage=stage, max_delay=settings["stage_delays"][stage]))
            save()
        if stopped["signal"]:
            reason = "interrupted"
            break
        if step - stage_start >= settings["max_stage_steps"]:
            reason = "stage_limit"
            break
    save()
    if reason != "interrupted":
        for delay in settings["heldout_delays"]:
            # Reporting only, never used to promote or select a checkpoint.
            logger.log(dict(step=step, **evaluate_maintenance(model, settings, 2, delay,
                                                            prefix=f"heldout_delay_{delay}")))
    report = dict(step=step, stop_reason=reason, stage=stage,
                  curriculum_complete=reason == "all_stages_passed", **counters)
    logger.finish(report)
    return report


def run_training(args, run, protocol):
    name = f"{args.model_family}-A{args.address_bits}V{args.value_bits}-seed{args.seed}"
    root = args.output_root.resolve() / name / run.id
    root.mkdir(parents=True, exist_ok=False)
    run.name = name
    run.define_metric("step")
    run.define_metric("*", step_metric="step")
    config = dict(protocol, seed=args.seed, model_family=args.model_family,
                  address_bits=args.address_bits, value_bits=args.value_bits,
                  uninterrupted_fresh_run=True, **provenance())
    run.config.update(config)
    (root / "protocol.json").write_text(json.dumps(config, indent=2) + "\n")
    first = protocol["transactions"]
    torch.manual_seed(args.seed)
    model = DBModel(first, pretrain=first["pretrain_attention"]).to(args.device).train()
    resources = resource_counts(model)
    run.config.update(resources)
    (root / "resources.json").write_text(json.dumps(resources, indent=2) + "\n")

    def finish(phase, report, offset=0):
        complete = phase == "maintenance" and report["curriculum_complete"]
        result = dict(protocol=PROTOCOL, run_id=run.id, phase=phase, pipeline_complete=complete,
                      pipeline_stop_reason=report["stop_reason"], total_training_steps=offset + report["step"],
                      checkpoint=str(root / phase / "last.pt"))
        (root / "results.json").write_text(json.dumps(result, indent=2) + "\n")
        run.summary.update(result)
        return complete

    with stop_signals() as stopped:
        model, first_report = train_transactions(model, first, PhaseLog(run, root, "transactions"), stopped)
        if not first_report["curriculum_complete"]:
            return finish("transactions", first_report)
        offset = first_report["step"]
        checkpoint = root / "transactions/last.pt"
        source = dict(model=args.model_family, seed=args.seed, path=str(checkpoint),
                      sha256=file_digest(checkpoint), config=first, run_id=run.id, step=offset)
        (root / "source_manifest.json").write_text(json.dumps(dict(checkpoints=[source]), indent=2) + "\n")
        # Continuous workflow: pass weights in memory; no external checkpoint required.
        torch.manual_seed(args.seed)
        replacement = DBModel(first).to(args.device)
        replacement.load_state_dict(model.state_dict(), strict=True)
        model = replacement.train()
        report = train_maintenance(model, first, protocol["maintenance"], source,
                                   PhaseLog(run, root, "maintenance", offset), stopped)
        return finish("maintenance", report, offset)


def smoke_test():
    """Small fixed CPU check, independent of the experiment CLI configuration."""
    torch.set_num_threads(1)
    torch.manual_seed(42)
    args = build_parser().parse_args(["--device", "cpu", "--address-bits", "3", "--value-bits", "3",
                                      "--batch-size", "2", "--eval-batch-size", "2", "--eval-samples", "2"])
    plan = training_protocol(args)
    model = DBModel(plan["transactions"]).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    rng = np.random.default_rng(42)
    episode = transaction_batch(2, 8, 8, 8, rng)
    for full_history in (True, False):
        optimizer_step(model, optimizer, episode, plan["transactions"], full_history)
    episode = maintenance_batch(2, "overwrite", 96, rng)
    optimizer_step(model, optimizer, episode, plan["maintenance"], False, balanced=True)
    metrics = evaluate_maintenance(model, plan["maintenance"], 2, 96)
    if not all(np.isfinite(value) for value in metrics.values()):
        raise FloatingPointError("Smoke-test evaluation returned nonfinite metrics")
    print(json.dumps(dict(smoke_test="passed", device="cpu", model="schemamem",
                          checks=["full-history backward", "recurrent backward", "maintenance backward/evaluation"],
                          note="Untrained smoke check, not an accuracy or convergence experiment."), indent=2))


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if sum((args.dry_run, args.print_sweep, args.smoke_test)) > 1:
        parser.error("Choose only one of --dry-run, --print-sweep, or --smoke-test")
    if args.smoke_test:
        if args.wandb_sweep:
            parser.error("--smoke-test does not run a W&B sweep")
        smoke_test()
        return 0
    try:
        protocol = training_protocol(args)
    except ValueError as error:
        parser.error(str(error))
    if args.dry_run or args.print_sweep:
        print(json.dumps(sweep_config(args) if args.print_sweep else protocol, indent=2))
        return 0

    import wandb

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error("CUDA is unavailable; use --dry-run to inspect the protocol")
    if args.device == "cpu" and args.model_family != "schemamem":
        parser.error("The paper's Mamba-3/GDN kernel paths require CUDA")
    torch.set_num_threads(1)
    with wandb.init(project=os.environ.get("WANDB_PROJECT", args.project),
                    entity=args.entity, mode=args.mode, resume="never") as run:
        if args.wandb_sweep:
            allowed = {"model_family", "address_bits", "value_bits", "seed"}
            unknown = set(run.config) - allowed
            if unknown:
                raise ValueError(f"Unsupported sweep parameters: {sorted(unknown)}")
            for key, value in run.config.items():
                setattr(args, key, value)
            protocol = training_protocol(args)
            if args.device == "cpu" and args.model_family != "schemamem":
                raise ValueError("The paper's Mamba-3/GDN kernel paths require CUDA")
        complete = run_training(args, run, protocol)
    return 0 if complete else 2


if __name__ == "__main__":
    raise SystemExit(main())
