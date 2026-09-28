# Standalone DB Evaluation

`eval_db.py` evaluates your own checkpoints produced by [train_db.py](train_db.py)
using the DB maintenance protocol. **Trained checkpoints and precomputed results
are not distributed.** Train your models first; evaluation does not download
weights or access W&B runs.

This entry point works with the root-only release and does not import the training
entry point. Its episode generator, model adapters, checkpoint loader, metrics,
aggregation, and plotting are self-contained. SchemaMem uses root-level model and
configuration files; Mamba-3 and Gated DeltaNet use installed official libraries.
No source subdirectories or external test files are needed. No training is
started or interrupted. See [README.md](README.md) for environment setup.

## Usage

Run from the directory containing the released files, with your Python environment
activated. The paths below refer to outputs you generate, not supplied artifacts.

```bash
# Audit your checkpoints without GPU inference or output-file changes.
python eval_db.py \
  --checkpoint-root outputs/db_paper --require-full-grid --dry-run

# Evaluate all 81 endpoints and generate tables and figures.
python eval_db.py \
  --checkpoint-root outputs/db_paper --require-full-grid \
  --output outputs/db_evaluation

# Evaluate a single endpoint or a smaller set while experiments are in progress.
# Replace this placeholder with your own maintenance/last.pt path.
python eval_db.py \
  --checkpoint outputs/db_paper/MODEL-AxVy-seedS/RUN_ID/maintenance/last.pt \
  --output outputs/db_evaluation_subset

# Regenerate plots and tables from saved evaluation JSON; no weights are needed.
python eval_db.py --plot-only --output outputs/db_evaluation

# Check checkpoint loading and evaluation without any trained weights.
python eval_db.py --smoke-test
```

`--checkpoint-root` discovers `maintenance/last.pt` files created by the public
training script. It does not pick the newest or best checkpoint. If multiple
checkpoints represent the same model/cell/seed,
provide an explicit selection using `--checkpoint` or `--manifest` instead.

A manifest can list your own checkpoint paths and optional SHA-256 hashes:

```json
{
  "checkpoints": [
    {"path": "relative/path/to/maintenance/last.pt"}
  ]
}
```

Relative paths are resolved against the manifest's directory. The evaluator also
writes a manifest with checkpoint hashes, which can be reused with `--manifest`.
Checkpoint files are never modified. Only load trusted checkpoints: training
checkpoints contain Python objects and are loaded with `weights_only=False`.

## Protocol

- Models: three-layer SchemaMem, Mamba-3, and Gated DeltaNet, as configured in
  `train_db.py`. No separate plain-attention bottom layer.
- Grid: A3/A4/A5 × V3/V4/V5 × seeds 42/43/44 × three models = 81 checkpoints.
- Delays: 96, 192, 384, 768, and 1,536 NULL queries.
- Samples: 512 episodes per scenario, evaluated in batches of 128.
- Generator seed: 20261001, with the original scenario/delay offsets.
- Inference: recurrent mode only, `eval()` and no gradients. CUDA uses BF16
  autocast. CPU is supported only for SchemaMem and uses FP32; it is not a
  numerically identical replacement for the CUDA paper evaluation.
- Stopping targets recorded in checkpoints: SchemaMem 95%, baselines 99%.
  These are training criteria, not assumed evaluation accuracies.

Every episode starts with fresh state. Retention writes half the addresses;
interference writes a quarter, queries all, then writes a disjoint quarter;
overwrite writes half, queries all, then overwrites a quarter. Delays query only
never-written addresses, and final queries cover every address in random order.
An immediate-query `legacy` condition is also retained to match the original
evaluation. Targets and predictions are never fed back as inputs.

Only final queries are scored. Exact accuracy requires every value bit to match;
bit accuracy and mean per-bit BCE are reported separately. Retained, updated, and
default values remain separate. Filler/intermediate queries are not scored.

SchemaMem commits at the period-96 and write/read segment boundaries. The delay
and final queries remain one read stream with no extra commit between them.
Mamba-3 and Gated DeltaNet scan continuously without resets at these boundaries.

### Dynamic-state ablation

For SchemaMem checkpoints, `--zero-state` sets every layer's dynamic state `S`
to zero immediately after **every** periodic or segment-boundary commit. This is
an inference-only intervention: weights, static schema embeddings, local attention,
write computations, and position counters are unchanged. It is not a single reset
followed by normal accumulation, nor does it switch to full-history attention.

Run the same checkpoint selection and sampling options twice, using separate
output directories. For example, with a manifest containing only your SchemaMem
checkpoints:

```bash
python eval_db.py --manifest schemamem_checkpoints.json \
  --delays 384 1536 --output outputs/state_normal
python eval_db.py --manifest schemamem_checkpoints.json \
  --delays 384 1536 --zero-state --output outputs/state_zero
```

The paper's state ablation uses all nine address/value settings and all three
SchemaMem training seeds, with the default sample count, batch size, and evaluation
seed above. Normal and ablated evaluations therefore use identical generated
episodes. The recorded `state_mode` distinguishes the conditions; the evaluator
rejects mixing them in one output directory. Baseline checkpoints are not accepted
with `--zero-state`.

`--samples`, `--batch-size`, `--eval-seed`, and `--delays` can be changed for smoke
tests or additional experiments, but this changes the recorded evaluation
protocol. In particular, batch size affects episode generation order. Defaults
must be retained to reproduce the paper's sampling conditions. The original
evaluation seed was also used during curriculum validation; this is a development
evaluation, not an untouched test set.

## Checkpoint Selection and Outputs

`--require-full-grid` rejects missing or duplicate combinations before inference.
Without it, subsets are supported and identified as incomplete grids. Budget-capped
terminal runs are included with `threshold_passed=false`; no seed is discarded
for poor performance. Nonterminal or unverifiable checkpoints require explicit
`--allow-incomplete` and mark the comparison provisional.

The script creates the following outputs under `--output`; none are required
to exist before the first evaluation:

- `manifest.json`: checkpoint paths, hashes, configurations, and runtime provenance.
- `A{a}V{v}/{family}_{seed}.json`: per-checkpoint metrics and query counts.
- `comparison.json`: all selected endpoints and the exact evaluation protocol.
- `per_seed.csv`: all metrics without averaging away individual seeds.
- `summary.json`: means, sample SDs, and paired accuracy drops from delay 384 to 1,536.
- `report.md`: endpoint audit and main-condition tables.
- PNG/PDF figures: exact/bit accuracy delay curves and individual-seed endpoints.

Means and sample SDs are calculated over training seeds, not individual queries.
A single-seed result has no estimated SD. Accuracy drops are calculated within
each seed before aggregation. Training-step counts and capped runs remain visible;
the comparison does not claim matched training budgets.

Saved per-checkpoint results are reused only if the checkpoint hash, evaluation
protocol, source hashes, and runtime identity match. Use `--force` to recompute.
`--no-plots` produces numerical outputs without requiring Matplotlib.
The evaluator loads only the final checkpoint itself: a `source` field records
training provenance but does not require the earlier checkpoint file to exist.

## Root-Only Smoke Check

```bash
python eval_db.py --smoke-test
```

This fixed CPU check creates a small randomly initialized SchemaMem model,
round-trips its weights through a temporary checkpoint, and verifies recurrent
evaluation and final-query counts. It also checks that the S=0 intervention leaves
pre-commit outputs unchanged and zeros all post-commit state reads. It needs no
checkpoint download, W&B account,
baseline kernel libraries, or separate test suite. Temporary files are removed
automatically. It does not produce trained accuracy results or persistent outputs.
