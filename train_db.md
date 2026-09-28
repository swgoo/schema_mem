# DB Experiments: Continuous Training

`train_db.py` is the training entry point in the source release.
Both curricula run from scratch in a single process and a single W&B run. No
pretrained weights, external dataset, or previous run is required. Final accuracy
targets are set before training starts; no separate continuation is required.

All data generation, baseline construction, training, evaluation, and logging
are implemented in this file. Its project dependencies are the root-level model
and configuration files. No source subdirectories are included or needed.
For environment setup and the file list, see [README.md](README.md).

## Fixed Settings

| Setting | SchemaMem | Mamba-3 | Gated DeltaNet |
|---|---|---|---|
| Layers / hidden width | 3 / 32 | 3 / 32 | 3 / 32 |
| Memory | 176 slots per layer | SISO, state size 80 | 2 heads, head dim 48 |
| Separate plain-attention bottom layer | None | None | None |
| Input | Little-endian address/value bits + validity bit, projection with bias | Same | Same |
| Output | Per-value-bit logits; all bits must match for exact-value accuracy | Same | Same |
| Final passing threshold | 95% | 99% | 99% |

The full grid is A3/A4/A5 × V3/V4/V5 × seeds 42/43/44 × three models = 81 runs.
The default batch size is 512, with validation on 512 episodes every 250 steps.
Each address has a default value of `address % values`; there is no separate
default-value pretraining stage. Only NULL queries receive loss, and target values
are never fed back as subsequent inputs.

1. **Transactions:** Query `N/4 → N/2 → N` addresses. Half of the queried addresses
   are updated first; the rest retain their defaults. Intermediate stages require
   80% accuracy, while the last stage uses the model-specific final threshold.
   SchemaMem first learns the same curriculum in no-commit mode, with an 80% threshold
   at every stage, then retains those weights to begin the recurrent curriculum.
   It subsequently alternates full-history and commit batches at a 1:1 ratio.
   Baselines remain in recurrent mode throughout.
2. **Maintenance:** Train with maximum delays of `96 → 192 → 384`, starting with
   retention and then adding disjoint-write interference and overwriting.
   Immediate retrieval without delay accounts for 20% of episodes; otherwise,
   delays are sampled from `0, 32, and the delay milestones reached so far`.
   Only SchemaMem uses full-history mode for one in every four batches.
   BCE is averaged within each group of default, retained, and updated values;
   filler queries receive a weight of 0.1. Intermediate stages require 90% accuracy,
   while the last stage uses the model-specific final threshold.

Both curricula require **two consecutive passing validations**. Maintenance uses
the minimum group accuracy across scenarios. SchemaMem commits at intervals of at
most 96 positions and at write/read boundaries; baseline recurrent states are not
reset at those boundaries.

The optimizer is AdamW with weight decay 0.01 and gradient clipping at 1.
The transaction learning rate is 3e-4; maintenance uses 1e-4 with a 500-step warmup.
Resetting the optimizer at the no-commit → recurrent and transaction → maintenance
transitions is part of the curriculum, not a process interruption or restart.

## Usage

Use the directory containing the released files as the working directory and
activate your own Python environment. Install the dependencies for the models
you intend to run. The complete sweep requires all three models' dependencies.

```bash
# Inspect settings without using a GPU or accessing W&B
python train_db.py --model-family schemamem --address-bits 3 --value-bits 4 --dry-run

# CPU forward/backward/evaluation check with a small random SchemaMem model
python train_db.py --smoke-test

# Train one model/configuration/seed from start to finish
python train_db.py --model-family schemamem --address-bits 3 --value-bits 4 --seed 42
python train_db.py --model-family mamba3 --address-bits 3 --value-bits 4 --seed 42
python train_db.py --model-family gdn --address-bits 3 --value-bits 4 --seed 42

# Generate the full 81-run grid configuration (JSON is also valid YAML)
python train_db.py --print-sweep > db-sweep.yaml
python -m wandb sweep db-sweep.yaml
# Use the ENTITY/PROJECT/SWEEP_ID returned by the command above
python -m wandb agent ENTITY/PROJECT/SWEEP_ID
```

The default W&B project is `SchemaMem-DB-Paper`; use `--project` and `--entity`
to configure the destination. Use `--mode disabled` for local logging only, or
`--mode offline` to defer uploads. `--print-sweep` alone neither creates a sweep
nor starts training. The YAML file is generated locally, not supplied as part
of the release. `--smoke-test` uses its own fixed, small CPU configuration,
does not connect to W&B or save weights, and does not measure convergence.

## Outputs and Training Limits

Each run is stored under `outputs/db_paper/MODEL-AxVy-seedS/RUN_ID/`.
These directories and files are created when you run training; no existing
output directory or checkpoint is distributed. Use `--output-root` to choose
another destination.

- `protocol.json`, `resources.json`: Actual settings, source hashes, package versions, and resource counts
- `transactions/`: No-commit/transaction logs and weights
- `maintenance/last.pt`: Final recurrent weights and model configuration
- `source_manifest.json`: Hash and path of the transaction checkpoint produced by this run
- `results.json`: Total steps, final checkpoint path, completion status, and stop reason

Default limits are 1,800,000 total transaction steps / 300,000 per stage and
360,000 total maintenance steps / 120,000 per stage. The generous maintenance
limit is set from the outset, so there is no separate continuation after reaching
a smaller limit.
Reaching a limit is not considered success: it is reported as
`pipeline_complete=false` with exit code 2. Maintenance does not start if the
transaction curriculum is incomplete. An existing run directory is never overwritten.

Post-training evaluations at delays 768/1536 are for reporting only, not stage
promotion or checkpoint selection. Maintenance validation/evaluation uses the
original protocol's generator seed, 20261001.
This script provides a continuous training recipe; it does not guarantee identical
weights or convergence steps across runs or environments.

## Evaluate Your Checkpoints

After training, use the standalone evaluator. A partial collection is supported;
add `--require-full-grid` only when all 81 endpoints are present.

```bash
python eval_db.py --checkpoint-root outputs/db_paper --dry-run
python eval_db.py --checkpoint-root outputs/db_paper --output outputs/db_evaluation
```

See [eval_db.md](eval_db.md) for checkpoint selection and output details.
