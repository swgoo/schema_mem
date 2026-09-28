"""Standalone DB evaluation using the root-only source release.

Only root model/config files and installed third-party libraries are required;
this script does not import the training entry point. Evaluate your own
train_db.py outputs: trained weights and evaluation artifacts are not supplied.
Load only trusted checkpoints: training checkpoints contain Python objects.
See eval_db.md for usage. Output directories are created at runtime.
"""

import argparse
from collections import defaultdict
from contextlib import nullcontext
import csv
from dataclasses import dataclass
from functools import partial
import hashlib
import importlib.metadata
import io
import json
from pathlib import Path
import tempfile

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parent
FAMILIES = ("schemamem", "mamba3", "gdn")
SEEDS = (42, 43, 44)
BIT_WIDTHS = (3, 4, 5)
DELAYS = (96, 192, 384, 768, 1536)
SCENARIOS = ("legacy", "retention", "interference", "overwrite")
GROUPS = ("retained", "updated", "default", "filler")
SCENARIO_GROUPS = {
    "legacy": ("updated", "default"),
    "retention": ("retained", "default"),
    "interference": ("retained", "updated", "default"),
    "overwrite": ("retained", "updated", "default"),
}
PANELS = (
    ("retention/retained", "Retention"),
    ("interference/retained", "Retained after disjoint writes"),
    ("overwrite/updated", "Overwritten values"),
    ("overwrite/default", "Never-written defaults"),
)
LABELS = dict(schemamem="SchemaMem", mamba3="Mamba-3", gdn="Gated DeltaNet")
COLORS = dict(schemamem="#0072B2", mamba3="#D55E00", gdn="#009E73")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, nargs="+", default=[],
                        help="Explicit terminal maintenance checkpoints; never selects best weights.")
    parser.add_argument("--checkpoint-root", type=Path, nargs="+", default=[],
                        help="Discover maintenance/last.pt from your training runs; duplicates are errors.")
    parser.add_argument("--manifest", type=Path,
                        help='JSON {"checkpoints": [{"path": "...", "sha256": "optional"}]}')
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/db_evaluation")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--samples", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=128,
                        help="Part of the sampling protocol; changing it changes the generated episodes.")
    parser.add_argument("--eval-seed", type=int, default=20261001)
    parser.add_argument("--delays", type=int, nargs="+", default=list(DELAYS))
    parser.add_argument("--zero-state", action="store_true",
                        help="SchemaMem only: zero every layer's S after each commit; use a separate output directory.")
    parser.add_argument("--require-full-grid", action="store_true",
                        help="Require all 81 model/address/value/seed combinations before inference.")
    parser.add_argument("--allow-incomplete", action="store_true",
                        help="Explicitly allow nonterminal checkpoints; mark the comparison provisional.")
    parser.add_argument("--dry-run", action="store_true", help="CPU-only checkpoint/protocol audit; no inference.")
    parser.add_argument("--force", action="store_true", help="Recompute reports instead of reusing exact identities.")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--plot-only", action="store_true", help="Redraw --output/comparison.json without weights.")
    parser.add_argument("--smoke-test", action="store_true",
                        help="CPU SchemaMem checkpoint/evaluation check with temporary random weights; no downloads.")
    return parser


def protocol_for(args):
    if args.samples < 1 or args.batch_size < 1 or args.eval_seed < 0:
        raise ValueError("Samples/batch size must be positive; evaluation seed must be nonnegative")
    if not args.delays or any(d <= 0 for d in args.delays) or args.delays != sorted(set(args.delays)):
        raise ValueError("Delays must be distinct positive integers in increasing order")
    return dict(eval_seed=args.eval_seed, eval_samples=args.samples,
                eval_batch_size=args.batch_size, delays=args.delays, stage=2, mode="recurrent",
                state_mode="zero_S" if args.zero_state else "normal")


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def load_checkpoint(path, expected_hash=None):
    # Hash exactly the bytes deserialized, even if a trainer replaces the path.
    data = Path(path).read_bytes()
    sha = hashlib.sha256(data).hexdigest()
    if expected_hash is not None and expected_hash != sha:
        raise ValueError(f"Checkpoint changed or manifest hash mismatch: {path}")
    return torch.load(io.BytesIO(data), map_location="cpu", weights_only=False), sha


def checkpoint_paths(args):
    paths = {p.resolve(): None for p in args.checkpoint}
    for root in args.checkpoint_root:
        if not root.is_dir():
            raise ValueError(f"Checkpoint root does not exist: {root}")
        for path in root.rglob("last.pt"):
            # Discover only the layout written by the public training entry point.
            if path.parent.name == "maintenance":
                paths.setdefault(path.resolve(), None)
    if args.manifest:
        manifest = json.loads(args.manifest.read_text())
        for entry in manifest["checkpoints"]:
            entry = dict(path=entry) if isinstance(entry, str) else entry
            path = Path(entry["path"])
            path = (path if path.is_absolute() else args.manifest.parent / path).resolve()
            expected = entry.get("sha256")
            if paths.get(path) and expected and paths[path] != expected:
                raise ValueError(f"Conflicting checkpoint hashes: {path}")
            paths[path] = expected or paths.get(path)
    if not paths:
        raise ValueError("Provide --checkpoint, --checkpoint-root, or --manifest")
    return paths


def validate_architecture(config, settings):
    family = config["model_family"]
    a, v = config["address_bits"], config["value_bits"]
    if family not in FAMILIES or a not in BIT_WIDTHS or v not in BIT_WIDTHS:
        raise ValueError("Paper scope is SchemaMem/Mamba-3/GDN, A3–5 and V3–5 only")
    expected = dict(addresses=2**a, values=2**v, hidden_size=32, layers=3,
                    intermediate_size=16, input_bias=True, bitwise_output=True,
                    commit_length=96)
    if family == "schemamem":
        expected.update(slots=176, kv_heads=1, all_trace_layers=True)
    elif family == "mamba3":
        expected.update(mamba_state_size=80)
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"Non-paper architecture: {key}={config.get(key)!r}, expected {value!r}")
    if config.get("no_commit") or config.get("no_chunk"):
        raise ValueError("Evaluation requires recurrent checkpoints")
    if settings["model_family"] != family or settings["seed"] != config["seed"]:
        raise ValueError("Inconsistent checkpoint family/seed")
    target = .95 if family == "schemamem" else .99
    if settings["final_accuracy"] != target or settings["stage_delays"] != [96, 192, 384]:
        raise ValueError("Expected SchemaMem 95% / baselines 99% targets and delays 96/192/384")


def describe_checkpoint(path, expected_hash=None, allow_incomplete=False):
    ck, sha = load_checkpoint(path, expected_hash)
    if not {"source", "settings", "model", "stage_index", "next_step"} <= ck.keys():
        raise ValueError(f"Expected a terminal maintenance checkpoint, not transaction weights: {path}")
    config, settings = ck["source"]["config"], ck["settings"]
    validate_architecture(config, settings)
    stage, step = int(ck["stage_index"]), int(ck["next_step"]) - 1
    if not 0 <= stage <= 3 or step < 0:
        raise ValueError(f"Invalid checkpoint stage/step: {path}")
    passed = stage == 3
    results_path = path.with_name("results.json")
    results = json.loads(results_path.read_text()) if results_path.exists() else {}
    reason = results.get("stop_reason", "unknown") if results.get("step") == step else "unknown"
    if passed:
        reason = "all_stages_passed"
    terminal = passed or reason in ("step_limit", "stage_limit")
    if not terminal and not allow_incomplete:
        raise ValueError(f"Nonterminal or unverifiable stop at {path}; use --allow-incomplete explicitly")
    return dict(path=str(path), sha256=sha, family=config["model_family"], seed=int(settings["seed"]),
                address_bits=config["address_bits"], value_bits=config["value_bits"],
                run=str(ck.get("run_id", path.parent.name)), config=config,
                step=step, source_step=int(ck["source"]["step"]), stage_index=stage,
                target_accuracy=settings["final_accuracy"], threshold_passed=passed,
                stop_reason=reason, terminal=terminal,
                source_pretraining=bool(config.get("pretrain_attention", False)))


def entry_key(entry):
    return tuple(entry[k] for k in ("address_bits", "value_bits", "family", "seed"))


def validate_selection(entries, require_full_grid=False):
    keys = [entry_key(e) for e in entries]
    if len(set(keys)) != len(keys):
        raise ValueError("Duplicate model/cell/seed checkpoints; select one terminal checkpoint explicitly")
    expected = {(a, v, f, s) for a in BIT_WIDTHS for v in BIT_WIDTHS for f in FAMILIES for s in SEEDS}
    if require_full_grid and set(keys) != expected:
        raise ValueError(f"Incomplete paper grid: missing={sorted(expected-set(keys))}, extra={sorted(set(keys)-expected)}")
    return set(keys) == expected


@dataclass
class Episode:
    keys: torch.Tensor
    values: torch.Tensor
    labels: torch.Tensor
    groups: torch.Tensor
    final: torch.Tensor
    segments: list


def make_episode(batch_size, scenario, delay, rng, device="cpu", addresses=8, values=8):
    """Same RNG order as the paper; labels never enter the model input."""
    if scenario not in SCENARIOS or delay < 0:
        raise ValueError("Unknown scenario or negative delay")
    if addresses < 4 or addresses % 4 or values < 3 or batch_size < 1:
        raise ValueError("Invalid episode dimensions")
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
                            headdim=16, ngroups=2, is_mimo=False, chunk_size=64, layer_idx=layer_idx)

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
    """Exact paper architecture/key layout, with recurrent-only inference."""

    def __init__(self, config):
        super().__init__()
        self.config = dict(config)
        self.model_family = config["model_family"]
        self.addresses, self.values = config["addresses"], config["values"]
        self.address_bits, self.value_bits = config["address_bits"], config["value_bits"]
        h, layers = config["hidden_size"], config["layers"]
        self.input_proj = nn.Linear(self.address_bits + self.value_bits + 1, h, bias=True)
        if self.model_family == "schemamem":
            from modeling_schema_mem_db import SchemaMemDBConfig, SchemaMemDBModel
            backbone_config = SchemaMemDBConfig(
                vocab_size=1, hidden_size=h, num_hidden_layers=layers,
                intermediate_size=config["intermediate_size"], num_attention_heads=4,
                num_key_value_heads=config["kv_heads"], head_dim=h // 4,
                schema_size=config["slots"], schema_layer_indices=list(range(layers)),
                sliding_window=config["commit_length"], max_position_embeddings=4096,
                attention_dropout=0., use_cache=False, bos_token_id=None,
                eos_token_id=None, pad_token_id=None)
            backbone_config._attn_implementation = "sdpa"
            self.backbone = SchemaMemDBModel(backbone_config)
            self.backbone.embed_tokens = nn.Identity()
        elif self.model_family == "mamba3":
            self.backbone = Mamba3Backbone(h, layers, config["mamba_state_size"])
        elif self.model_family == "gdn":
            from fla.models.gated_deltanet import GatedDeltaNetConfig, GatedDeltaNetModel
            backbone_config = GatedDeltaNetConfig(
                vocab_size=1, hidden_size=h, num_hidden_layers=layers, head_dim=48,
                num_heads=2, expand_v=1., intermediate_size=config["intermediate_size"],
                attn=None, attn_mode="chunk", use_gate=True, use_short_conv=True,
                conv_size=4, allow_neg_eigval=False, use_cache=False, pad_token_id=None,
                bos_token_id=None, eos_token_id=None, fuse_norm=False, fuse_swiglu=False)
            self.backbone = GatedDeltaNetModel(backbone_config)
            self.backbone.set_input_embeddings(nn.Identity())
        else:
            raise ValueError(self.model_family)
        self.classifier = nn.Linear(h, self.value_bits, bias=False)

    def forward(self, episode, *, zero_state=False):
        if zero_state and self.model_family != "schemamem":
            raise ValueError("S=0 is defined only for SchemaMem, not baseline state resets")
        valid = (episode.values >= 0).unsqueeze(-1)
        abits = (episode.keys.unsqueeze(-1) >> torch.arange(self.address_bits, device=episode.keys.device)) & 1
        vbits = ((episode.values.clamp_min(0).unsqueeze(-1) >>
                  torch.arange(self.value_bits, device=episode.keys.device)) & 1) * valid
        inputs = self.input_proj(torch.cat([abits, vbits, valid], dim=-1).to(self.input_proj.weight.dtype))
        if self.model_family == "schemamem":
            from modeling_schema_mem_db import SchemaCache

            class EvaluationCache(SchemaCache):
                def commit(self, layer_idx=None):
                    changed = super().commit(layer_idx)
                    if zero_state:
                        for index in changed:
                            self.states[index] = torch.zeros_like(self.states[index])
                    return changed

            # Keep writes, local attention, static E and position counters intact.
            cache, parts = EvaluationCache(), []  # Fresh state for every episode/batch.
            for i, (start, stop) in enumerate(episode.segments):
                parts.append(self.backbone(inputs_embeds=inputs[:, start:stop], schema_cache=cache).last_hidden_state)
                if i < len(episode.segments) - 1:
                    cache.commit()
            hidden = torch.cat(parts, dim=1)
        else:
            # Baselines scan continuously, with no reset at SchemaMem boundaries.
            hidden = (self.backbone(inputs) if self.model_family == "mamba3" else
                      self.backbone(inputs_embeds=inputs, use_cache=False).last_hidden_state)
        return self.classifier(hidden)


def resource_counts(model):
    layout = {}
    if model.model_family == "schemamem":
        layout["schema"] = model.config["layers"] * model.config["slots"] * model.config["hidden_size"]
    elif model.model_family == "mamba3":
        for i, layer in enumerate(model.backbone.layers):
            for name, tensor in zip(("angle", "ssm", "key", "value"),
                                    layer.mixer.allocate_inference_cache(1, 1, device="cpu")):
                layout[f"{i}/{name}"] = tensor.numel()
    else:
        for i, layer in enumerate(model.backbone.layers):
            a = layer.attn
            layout[f"{i}/recurrent"] = a.num_v_heads * a.head_k_dim * a.head_v_dim
            for name, width in (("q", a.key_dim), ("k", a.key_dim), ("v", a.value_dim)):
                layout[f"{i}/conv_{name}"] = width * a.conv_size
    return dict(parameters=sum(p.numel() for p in model.parameters()),
                recurrent_state_elements=sum(layout.values()), recurrent_state_layout=layout)


def query_stats(logits, episode):
    bits = ((episode.labels.clamp_min(0)[..., None] >> torch.arange(logits.shape[-1], device=logits.device)) & 1).bool()
    matches = (logits.float() >= 0) == bits
    bce = F.binary_cross_entropy_with_logits(logits.float(), bits.float(), reduction="none").mean(-1)
    result = {}
    for group, name in enumerate(GROUPS):
        mask = (episode.groups == group) & episode.final
        if mask.any():
            result[name] = np.array([matches.all(-1)[mask].sum().item(), mask.sum().item(),
                                     matches[mask].sum().item(), bce[mask].sum().item()], dtype=float)
    return result


@torch.no_grad()
def evaluate_delay(model, protocol, delay, device="cpu"):
    was_training = model.training
    model.eval()
    result, counts, aggregate = {}, {}, {}

    def record(totals, prefix):
        for group, (correct, count, bit_correct, ce) in totals.items():
            key = f"{prefix}/{group}"
            result.update({f"{key}/accuracy": correct/count,
                           f"{key}/bit_accuracy": bit_correct/(model.value_bits*count),
                           f"{key}/ce": ce/count})
            counts[key] = int(count)

    try:
        for index, scenario in enumerate(SCENARIOS):
            actual_delay = 0 if scenario == "legacy" else delay
            rng = np.random.default_rng(protocol["eval_seed"] + 1009*index + actual_delay)
            totals = {}
            for start in range(0, protocol["eval_samples"], protocol["eval_batch_size"]):
                episode = make_episode(min(protocol["eval_batch_size"], protocol["eval_samples"]-start),
                                       scenario, actual_delay, rng, device, model.addresses, model.values)
                context = torch.autocast("cuda", dtype=torch.bfloat16) if str(device).startswith("cuda") else nullcontext()
                with context:
                    logits = model(episode, zero_state=protocol.get("state_mode", "normal") == "zero_S")
                if not torch.isfinite(logits).all():
                    raise FloatingPointError("Nonfinite evaluation output")
                for name, values in query_stats(logits, episode).items():
                    totals[name] = totals.get(name, np.zeros(4)) + values
            record(totals, f"delay_{delay}/{scenario}")
            for name, values in totals.items():
                aggregate[name] = aggregate.get(name, np.zeros(4)) + values
        record(aggregate, f"delay_{delay}")
        result[f"delay_{delay}/min_accuracy"] = min(v for k, v in result.items() if k.endswith("/accuracy"))
        return result, counts
    finally:
        model.train(was_training)


def runtime_identity(device):
    packages = {}
    for name in ("torch", "numpy", "transformers", "mamba-ssm", "flash-linear-attention", "fla-core", "causal-conv1d"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    code = {name: digest(ROOT/name) for name in ("eval_db.py", "modeling_schema_mem_db.py",
                                                "modeling_schema_mem.py", "configuration_schema_mem.py")}
    return dict(packages=packages, code=code, device=device,
                precision="bf16_autocast" if device.startswith("cuda") else "fp32",
                gpu=torch.cuda.get_device_name(torch.device(device)) if device.startswith("cuda") else None)


def validate_reports(payload):
    protocol, reports = payload["protocol"], payload["reports"]
    if not reports:
        raise ValueError("No evaluation reports")
    validate_selection(reports)
    reference_runtime = reports[0]["identity"]["runtime"]
    state_mode = protocol.get("state_mode", "normal")
    if state_mode not in ("normal", "zero_S"):
        raise ValueError("Unknown state intervention")
    for r in reports:
        if state_mode == "zero_S" and r["family"] != "schemamem":
            raise ValueError("S=0 reports must contain SchemaMem only")
        if r["identity"]["protocol"] != protocol or r["identity"]["runtime"] != reference_runtime:
            raise ValueError("Mixed evaluation protocol or runtime identities")
        for delay in protocol["delays"]:
            for scenario, groups in SCENARIO_GROUPS.items():
                for group in groups:
                    for metric in ("accuracy", "bit_accuracy", "ce"):
                        key = f"delay_{delay}/{scenario}/{group}/{metric}"
                        x = r["metrics"].get(key)
                        if x is None or not np.isfinite(x) or x < 0 or (metric != "ce" and x > 1):
                            raise ValueError(f"Missing or invalid metric: {key}")


def summarize(payload):
    """Means/sample SD across training seeds, never across individual queries."""
    validate_reports(payload)
    grouped = defaultdict(list)
    for r in payload["reports"]:
        grouped[(r["address_bits"], r["value_bits"], r["family"])].append(r)
    summary, drops = [], []
    for (a, v, family), rows in sorted(grouped.items()):
        for key in sorted(rows[0]["metrics"]):
            values = np.array([r["metrics"][key] for r in rows])
            summary.append(dict(address_bits=a, value_bits=v, family=family, metric=key,
                                seeds=sorted(r["seed"] for r in rows), n=len(rows), mean=float(values.mean()),
                                sd=float(values.std(ddof=1)) if len(rows) > 1 else None))
        if 384 in payload["protocol"]["delays"] and 1536 in payload["protocol"]["delays"]:
            for key, _ in PANELS:
                # Paired within each seed BEFORE aggregating; never subtract SDs.
                values = [100*(r["metrics"][f"delay_384/{key}/accuracy"] -
                               r["metrics"][f"delay_1536/{key}/accuracy"]) for r in rows]
                drops.append(dict(address_bits=a, value_bits=v, family=family, condition=key,
                                  n=len(rows), mean_pp=float(np.mean(values)),
                                  sd_pp=float(np.std(values, ddof=1)) if len(rows) > 1 else None,
                                  per_seed={str(r["seed"]): value for r, value in zip(rows, values)}))
    return summary, drops


def save_outputs(payload, output):
    summary, drops = summarize(payload)
    write_json(output/"comparison.json", payload)
    write_json(output/"summary.json", dict(metrics=summary, paired_accuracy_drop=drops))
    with (output/"per_seed.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["address_bits", "value_bits", "family", "seed", "run", "threshold_passed",
                         "stop_reason", "total_steps", "metric", "value"])
        for r in payload["reports"]:
            for key, value in sorted(r["metrics"].items()):
                writer.writerow([r["address_bits"], r["value_bits"], r["family"], r["seed"], r["run"],
                                 r["threshold_passed"], r["stop_reason"], r["source_step"]+r["step"], key, value])
    lines = ["# DB maintenance evaluation", "",
             f"State condition: {payload['protocol'].get('state_mode', 'normal')}.", "",
             f"Checkpoints: {len(payload['reports'])}; complete 81-run grid: {payload['full_grid']}; "
             f"provisional: {payload['provisional']}.", "",
             "Recurrent inference only. Means and sample SD are across training seeds, not confidence intervals.",
             "The evaluation generator seed was also used for validation; this is not an untouched test set.",
             "A delay consists of NULL queries to never-written addresses; only final queries are scored.",
             "Capped endpoints are retained and marked. Training budgets are unequal.",
             "Persistent-state counts exclude local KV caches, pending updates, and runtime buffers.", "",
             "| Cell | Model | Seed | Target | Passed | Stop reason | Total steps | Parameters | State elements |",
             "|---|---|---:|---:|---|---|---:|---:|---:|"]
    for r in payload["reports"]:
        lines.append(f"| A{r['address_bits']}V{r['value_bits']} | {LABELS[r['family']]} | {r['seed']} | "
                     f"{r['target_accuracy']:.0%} | {r['threshold_passed']} | {r['stop_reason']} | "
                     f"{r['source_step']+r['step']:,} | {r['parameters']:,} | {r['recurrent_state_elements']:,} |")
    lines += ["", "## Exact accuracy: mean ± sample SD (%)", "",
              "| Cell | Model | Delay | Condition | Seeds | Accuracy |",
              "|---|---|---:|---|---|---:|"]
    panels = {p for p, _ in PANELS}
    for row in summary:
        parts = row["metric"].split("/")
        if len(parts) == 4 and "/".join(parts[1:3]) in panels and parts[-1] == "accuracy":
            value = f"{100*row['mean']:.2f}"
            if row["sd"] is not None:
                value += f" ± {100*row['sd']:.2f}"
            lines.append(f"| A{row['address_bits']}V{row['value_bits']} | {LABELS[row['family']]} | "
                         f"{parts[0][6:]} | {'/'.join(parts[1:3])} | {row['seeds']} | {value} |")
    (output/"report.md").write_text("\n".join(lines) + "\n")


def render(payload, output):
    """Grid delay profiles plus individual seed endpoints, using saved JSON only."""
    validate_reports(payload)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows, delays = payload["reports"], payload["protocol"]["delays"]
    aa, vv = sorted({r["address_bits"] for r in rows}), sorted({r["value_bits"] for r in rows})
    grouped = defaultdict(list)
    for r in sorted(rows, key=entry_key):
        grouped[(r["address_bits"], r["value_bits"], r["family"])].append(r)

    def finish(fig, axes, name, title):
        handles = {}
        for ax in axes.flat:
            hs, labels = ax.get_legend_handles_labels()
            handles.update(zip(labels, hs))
        if handles:
            fig.legend(handles.values(), handles.keys(), loc="lower center", ncol=3, bbox_to_anchor=(.5, .025))
        fig.suptitle(title + (" — S=0 after every commit"
                            if payload["protocol"].get("state_mode") == "zero_S" else ""))
        fig.text(.5, .01, "Sample SD across available training seeds (not CI). Capped endpoints included.",
                 ha="center", fontsize=8)
        fig.tight_layout(rect=(0, .08, 1, .94))
        for ext in ("png", "pdf"):
            fig.savefig(output/f"{name}.{ext}", dpi=160)
        plt.close(fig)

    for metric in ("accuracy", "bit_accuracy"):
        for key, title in PANELS:
            fig, axes = plt.subplots(len(aa), len(vv), squeeze=False,
                                     figsize=(4.5*len(vv), 3.5*len(aa)), sharex=True, sharey=True)
            for i, a in enumerate(aa):
                for j, v in enumerate(vv):
                    ax = axes[i, j]
                    for family in FAMILIES:
                        subset = grouped.get((a, v, family), [])
                        if not subset:
                            continue
                        y = np.array([[100*r["metrics"][f"delay_{d}/{key}/{metric}"] for d in delays] for r in subset])
                        mean = y.mean(0)
                        ax.plot(delays, mean, "o-", color=COLORS[family], label=LABELS[family])
                        if len(subset) > 1:
                            sd = y.std(0, ddof=1)
                            ax.fill_between(delays, mean-sd, mean+sd, color=COLORS[family], alpha=.14)
                    ax.set_title(f"A{a}V{v}")
                    ax.set_xscale("log", base=2)
                    ax.set_xticks(delays, labels=delays)
                    ax.set_ylim(0, 102)
                    ax.axvline(384, color="gray", ls=":")
                    ax.grid(alpha=.2)
                    if i == len(aa)-1:
                        ax.set_xlabel("Delay (NULL queries)")
                    if j == 0:
                        ax.set_ylabel("Accuracy (%)")
            finish(fig, axes, f"{key.replace('/', '_')}_{metric}", f"{title}: {metric}")
    fig, axes = plt.subplots(len(aa), len(vv), squeeze=False,
                             figsize=(4.5*len(vv), 3.5*len(aa)), sharey=True)
    for i, a in enumerate(aa):
        for j, v in enumerate(vv):
            ax = axes[i, j]
            for k, family in enumerate(FAMILIES):
                subset = grouped.get((a, v, family), [])
                for n, r in enumerate(subset):
                    y = [100*r["metrics"][f"delay_{delays[-1]}/{key}/accuracy"] for key, _ in PANELS]
                    x = np.arange(4)+(k-1)*.22+(n-(len(subset)-1)/2)*.04
                    ax.scatter(x, y, color=COLORS[family], marker=("o", "s", "^")[n % 3],
                               label=LABELS[family] if n == 0 else None)
            ax.set_title(f"A{a}V{v}")
            ax.set_xticks(range(4), labels=["Retain", "Interf.", "Update", "Default"])
            ax.set_ylim(0, 102)
            ax.grid(axis="y", alpha=.2)
    finish(fig, axes, "individual_seeds", f"Individual seeds at delay {delays[-1]}")


def run_evaluation(args):
    protocol = protocol_for(args)
    entries = [describe_checkpoint(path, sha, args.allow_incomplete)
               for path, sha in checkpoint_paths(args).items()]
    entries.sort(key=entry_key)
    if args.zero_state and any(e["family"] != "schemamem" for e in entries):
        raise ValueError("--zero-state requires SchemaMem-only checkpoints")
    full_grid = validate_selection(entries, args.require_full_grid)
    for e in entries:
        print(f"A{e['address_bits']}V{e['value_bits']} {e['family']} seed={e['seed']} "
              f"step={e['step']:,} target={e['target_accuracy']:.0%} {e['stop_reason']}", flush=True)
    if args.dry_run:
        print(json.dumps(dict(protocol=protocol, full_grid=full_grid, checkpoints=len(entries)), indent=2))
        return
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; no automatic CPU fallback")
    if args.device != "cpu" and not args.device.startswith("cuda"):
        raise ValueError("Use cpu or cuda[:index]")
    if args.device == "cpu" and any(e["family"] != "schemamem" for e in entries):
        raise ValueError("Mamba-3/GDN use CUDA kernels; CPU evaluation supports SchemaMem only")
    torch.set_num_threads(1)
    runtime = runtime_identity(args.device)
    args.output.mkdir(parents=True, exist_ok=True)
    existing_manifest = args.output/"manifest.json"
    if existing_manifest.exists():
        previous = json.loads(existing_manifest.read_text())["protocol"].get("state_mode", "normal")
        if previous != protocol["state_mode"]:
            raise ValueError("Use separate output directories for normal and S=0 evaluation")
    write_json(args.output/"manifest.json", dict(protocol=protocol, checkpoints=entries, runtime=runtime))
    reports = []
    for entry in entries:
        identity = dict(sha256=entry["sha256"], protocol=protocol, runtime=runtime)
        path = args.output/f"A{entry['address_bits']}V{entry['value_bits']}"/f"{entry['family']}_{entry['seed']}.json"
        report = None
        if path.exists() and not args.force:
            cached = json.loads(path.read_text())
            if cached.get("identity") == identity:
                validate_reports(dict(protocol=protocol, reports=[cached]))
                report = dict(cached, **entry)
                print(f"Reusing verified report: {path}", flush=True)
        if report is None:
            ck, _ = load_checkpoint(entry["path"], entry["sha256"])
            model = DBModel(entry["config"])
            model.load_state_dict(ck["model"], strict=True)
            resources = resource_counts(model)
            model.to(args.device).eval()
            report = dict(entry, identity=identity, **resources, metrics={}, query_counts={})
            for delay in protocol["delays"]:
                metrics, counts = evaluate_delay(model, protocol, delay, args.device)
                report["metrics"].update(metrics)
                report["query_counts"].update(counts)
                print(f"{path.stem} A{entry['address_bits']}V{entry['value_bits']} delay={delay} done", flush=True)
            del model, ck
            if args.device.startswith("cuda"):
                torch.cuda.empty_cache()
        write_json(path, report)
        reports.append(report)
    payload = dict(protocol=protocol, full_grid=full_grid,
                   provisional=any(not e["terminal"] for e in entries), reports=reports)
    save_outputs(payload, args.output)
    if not args.no_plots:
        render(payload, args.output)
    print(f"Saved {len(reports)} checkpoint evaluations to {args.output}", flush=True)


def smoke_test():
    """Verify checkpoint loading and evaluation without distributed weights/tests."""
    torch.set_num_threads(1)
    torch.manual_seed(42)
    config = dict(model_family="schemamem", seed=42, addresses=8, values=8, address_bits=3,
                  value_bits=3, hidden_size=32, layers=3, intermediate_size=16, input_bias=True,
                  bitwise_output=True, commit_length=96, slots=176, kv_heads=1,
                  all_trace_layers=True, pretrain_attention=True)
    model = DBModel(config).eval()
    protocol = protocol_for(build_parser().parse_args(["--samples", "2", "--batch-size", "2", "--delays", "96"]))
    with tempfile.TemporaryDirectory(prefix="schemamem-eval-smoke-") as directory:
        path = Path(directory)/"last.pt"
        # Deliberately untrained, not a fabricated completed experiment.
        torch.save(dict(model=model.state_dict(), source=dict(config=config, step=0),
                        settings=dict(model_family="schemamem", seed=42, final_accuracy=.95,
                                      stage_delays=[96, 192, 384]), stage_index=0, next_step=1), path)
        entry = describe_checkpoint(path, allow_incomplete=True)
        checkpoint, _ = load_checkpoint(path, entry["sha256"])
        restored = DBModel(entry["config"]).eval()
        restored.load_state_dict(checkpoint["model"], strict=True)
        expected, _ = evaluate_delay(model, protocol, 96)
        actual, counts = evaluate_delay(restored, protocol, 96)
        if actual != expected or counts["delay_96/overwrite/updated"] != 4:
            raise RuntimeError("Checkpoint/evaluation smoke-test mismatch")
        if any(p.grad is not None for p in restored.parameters()):
            raise RuntimeError("Evaluation unexpectedly created parameter gradients")
        # A random nonzero writer makes the intervention observable. Check state
        # reads directly rather than expecting random predictions to get worse.
        with torch.no_grad():
            for layer in restored.backbone.layers:
                layer.schema_write_delta_proj.weight.normal_(std=.1)
            episode = make_episode(2, "retention", 96, np.random.default_rng(42))
            observed = {False: [], True: []}
            for zero in (False, True):
                hooks = []
                def inspect_state(layer, args, kwargs):
                    state = kwargs["schema_cache"].get_state(layer.layer_idx)
                    observed[zero].append(state is not None and bool(state.count_nonzero()))
                try:
                    for layer in restored.backbone.layers:
                        hooks.append(layer.register_forward_pre_hook(inspect_state, with_kwargs=True))
                    logits = restored(episode, zero_state=zero)
                    if zero:
                        if not torch.equal(normal_logits[:, :episode.segments[0][1]],
                                           logits[:, :episode.segments[0][1]]):
                            raise RuntimeError("S=0 changed pre-commit outputs")
                    else:
                        normal_logits = logits
                finally:
                    for hook in hooks:
                        hook.remove()
            if not any(observed[False]) or any(observed[True]):
                raise RuntimeError("S=0 state-read intervention failed")
    print(json.dumps(dict(smoke_test="passed", device="cpu", model="schemamem",
                          checks=["temporary checkpoint round-trip", "recurrent evaluation", "final-query counts",
                                  "S=0 at all post-commit reads", "unchanged pre-commit outputs"],
                          note="Random weights; no trained accuracy claim or persistent output."), indent=2))


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.smoke_test:
        if args.plot_only or args.dry_run or args.checkpoint or args.checkpoint_root or args.manifest or args.zero_state:
            parser.error("--smoke-test uses its own temporary weights; do not combine with checkpoint/evaluation options")
        smoke_test()
    elif args.plot_only:
        if args.dry_run or args.no_plots or args.checkpoint or args.checkpoint_root or args.manifest or args.zero_state:
            parser.error("--plot-only reads comparison.json; do not combine it with checkpoint/evaluation options")
        payload = json.loads((args.output/"comparison.json").read_text())
        save_outputs(payload, args.output)
        render(payload, args.output)
    else:
        run_evaluation(args)


if __name__ == "__main__":
    main()
