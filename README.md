# SchemaMem

SchemaMem combines chunk-local attention with a persistent, schema-indexed state.
This source release includes the model implementation and a controlled
address–value task for studying state retention, interference, and overwriting.

## Anonymous Supplementary Material

This source-only package contains the files listed below and this README; no
trained checkpoints, experiment logs, account credentials, or repository history
are included. W&B examples use placeholders or your own configured account.
Generated logs and checkpoints may contain local paths and account information;
do not include them when redistributing the anonymous source package.

## Files

| File | Purpose |
|---|---|
| [train_db.py](train_db.py) | From-scratch DB training, curricula, and W&B sweep configuration |
| [eval_db.py](eval_db.py) | Checkpoint evaluation, seed aggregation, tables, and figures |
| [modeling_schema_mem_db.py](modeling_schema_mem_db.py) | Mistral-style SchemaMem model used for the DB task |
| [modeling_schema_mem.py](modeling_schema_mem.py) | Schema state/cache and Gemma-compatible model implementation |
| [configuration_schema_mem.py](configuration_schema_mem.py) | SchemaMem configuration |
| [utils.py](utils.py) | Gemma conversion and model utilities; not needed by the DB entry points |
| [requirements-mamba3.txt](requirements-mamba3.txt) | Mamba-3 kernel/environment add-on pins |
| [requirements-gdn.txt](requirements-gdn.txt) | Gated DeltaNet / FLA dependency pins |
| [train_db.md](train_db.md), [eval_db.md](eval_db.md) | Training and evaluation instructions |

## Environment

Use Python 3.11 or newer (development environment: Python 3.12). Install PyTorch
for your CPU or CUDA environment first. The remaining common dependencies are
NumPy, Transformers, W&B for training, and Matplotlib for plotting. Evaluation
does not require W&B; `--no-plots` avoids the Matplotlib dependency.

Reference package versions in the development environment:

| Package | Version |
|---|---|
| PyTorch | 2.10.0+cu130 |
| Transformers | 5.14.1 |
| NumPy | 2.4.1 |
| W&B | 0.28.1 |
| Matplotlib | 3.11.1 |
| Mamba-SSM | 2.3.2.post1 |
| Flash Linear Attention / FLA Core | 0.5.2 / 0.5.2 |

After activating your own environment and installing a suitable PyTorch build:

```bash
python -m pip install numpy==2.4.1 transformers==5.14.1 wandb==0.28.1 matplotlib==3.11.1
```

Transformers must provide both Mistral and Gemma4 APIs: the DB model reuses the
schema cache from the Gemma-compatible implementation even though it uses a
Mistral-style decoder. Installing an older Mistral-only Transformers version is
not sufficient. No Gemma weights are needed for the DB task.

For baseline experiments, install the official Mamba-3 implementation exposing
`mamba_ssm.modules.mamba3.Mamba3` and FLA's Gated DeltaNet implementation. Their
CUDA kernels must match your hardware and PyTorch/CUDA environment.
[requirements-mamba3.txt](requirements-mamba3.txt) records add-ons for the CUDA 13
reference environment, **not** a complete or hardware-independent installation
recipe. [requirements-gdn.txt](requirements-gdn.txt) pins the FLA packages.
These optional baseline libraries are imported only when their model is selected.
CPU smoke checks use SchemaMem and do not require either baseline library.

## Smoke Checks

Run these commands from the directory containing the released files:

```bash
python train_db.py --smoke-test
python eval_db.py --smoke-test
python train_db.py --dry-run
```

The smoke checks use small, randomly initialized SchemaMem models on CPU. They
check forward/backward execution, checkpoint round-tripping, and recurrent
evaluation without W&B login, distributed weights, or external test files.
Temporary evaluation files are removed automatically. These are implementation
checks, not convergence or model-quality experiments.

## Train and Evaluate

Train one configuration using local logging, or omit `--mode disabled` to log
to your W&B account:

```bash
python train_db.py --model-family schemamem --address-bits 3 --value-bits 4 --seed 42 --mode disabled
```

Replace `schemamem` with `mamba3` or `gdn` for a baseline. SchemaMem first learns
with full-history attention, then transitions to recurrent training. All models
proceed through transaction and maintenance curricula in a single process.
The final targets are 95% for SchemaMem and 99% for the baselines. Stage limits
are safeguards, not successful completion criteria.

After your runs finish:

```bash
python eval_db.py --checkpoint-root outputs/db_paper --dry-run
python eval_db.py --checkpoint-root outputs/db_paper --output outputs/db_evaluation
```

The evaluator uses the terminal `maintenance/last.pt` files created by training.
It never downloads weights or chooses the best checkpoint based on evaluation
accuracy. Duplicate model/configuration/seed endpoints must be resolved explicitly.
Only load trusted checkpoint files.

For the complete grid—three models × A3/A4/A5 × V3/V4/V5 × seeds 42/43/44:

```bash
python train_db.py --print-sweep > db-sweep.yaml
python -m wandb sweep db-sweep.yaml
# Use the identifier returned by W&B, from the same working directory.
python -m wandb agent ENTITY/PROJECT/SWEEP_ID

# Once all runs have terminal maintenance checkpoints:
python eval_db.py --checkpoint-root outputs/db_paper --require-full-grid
```

`db-sweep.yaml` is generated by the command above; it is not a required release
file. Full experiments require CUDA for the Mamba-3/GDN kernels and can take
substantial training time. No sweep or training job starts merely by inspecting
the configuration.

## Evaluation Scope

Evaluation uses recurrent mode at delays 96, 192, 384, 768, and 1,536, with 512
episodes per scenario. Retained, overwritten, and never-written default values
are measured separately; exact-value accuracy requires all value bits to match.
Outputs include all selected seeds, sample standard deviations, paired accuracy
drops, and training-step counts. Capped runs remain visible rather than being
silently discarded.

The evaluation seed was also used for curriculum validation. These are controlled
development diagnostics, not an untouched test set or a claim about agent-level
performance. Training budgets differ across models. Re-running this recipe does
not guarantee identical weights, convergence steps, or reported numerical results.

See [train_db.md](train_db.md) and [eval_db.md](eval_db.md) for the full protocol,
checkpoint layout, runtime outputs, and command-line options.
