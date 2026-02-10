# FiLM Disease-Vector Conditioning for nnU-Net v2

This document describes the MLP + FiLM (Feature-wise Linear Modulation) approach
for disease-conditioned segmentation in the CHD nnU-Net fork.

## Overview

A binary disease flag vector (length K=8) is mapped through a small MLP to a dense
embedding.  Per-stage linear heads predict channel-wise **gamma** and **beta**
coefficients, which modulate feature maps via:

```
y = (1 + gamma) * x + beta
```

FiLM is applied at the bottleneck and every decoder stage.  When no disease vector
is provided, the network runs the exact baseline forward pass with zero extra compute.

### Why FiLM?

FiLM is lightweight (only linear heads per stage, no convolutions), preserves spatial
structure, and has a clean identity initialisation: zero-init gamma/beta heads guarantee
``y = x`` at the start of training, so the network begins from the baseline.

## Disease Column Order

The disease vector has **K = 8** binary flags in this fixed order:

| Index | Column |
|-------|--------|
| 0     | HLHS   |
| 1     | ASD    |
| 2     | VSD    |
| 3     | AVSD   |
| 4     | DORV   |
| 5     | PuA    |
| 6     | ToF    |
| 7     | TGA    |

## disease_map.json Format

A JSON object mapping nnU-Net case identifiers to disease vectors:

```json
{
  "SV_001": [0, 1, 0, 0, 1, 0, 0, 0],
  "SV_002": [1, 0, 0, 0, 0, 0, 0, 0],
  ...
}
```

- Keys must match the case identifiers used by nnU-Net (derived from filenames:
  `<case_id>_0000.nii.gz` in raw data, `<case_id>.npz` in preprocessed data).
- Values are lists of K integers (0 or 1).

### Building disease_map.json from the CSV

Given a CSV with columns `[index, SV, HLHS, ASD, VSD, AVSD, DORV, PuA, ToF, TGA, ...]`:

```python
import csv, json

DISEASE_COLS = ["HLHS", "ASD", "VSD", "AVSD", "DORV", "PuA", "ToF", "TGA"]
disease_map = {}
with open("your_metadata.csv") as f:
    reader = csv.DictReader(f)
    for row in reader:
        case_id = row["SV"]
        disease_map[case_id] = [int(row[c]) for c in DISEASE_COLS]

with open("disease_map.json", "w") as f:
    json.dump(disease_map, f, indent=2)
```

## Placement

Place `disease_map.json` in the **preprocessed dataset base folder**:

```
$nnUNet_preprocessed/DatasetXXX_Name/disease_map.json
```

The trainer automatically looks for it there.  If the file is absent, conditioning is
disabled and the network runs as exact baseline.

## Enabling / Disabling Conditioning

| Scenario                        | What happens                                  |
|---------------------------------|-----------------------------------------------|
| `disease_map.json` present      | Conditioning enabled for train + val steps     |
| `disease_map.json` absent       | Exact baseline (no FiLM modules are used)      |
| Case ID missing from JSON       | **Error** during training (data integrity)     |

## Hyperparameters

| Parameter   | Default | Meaning                                       |
|-------------|---------|-----------------------------------------------|
| `disease_K` | 8       | Length of the binary disease flag vector       |
| `disease_H` | 64      | Hidden dimension of the disease MLP            |
| `disease_E` | 32      | Output embedding dim (input to FiLM heads)     |

To change these, subclass the trainer and set them in `__init__`.

## Training

```bash
# 1000-epoch (default DA5 length)
nnUNetv2_train DATASET_ID 3d_fullres FOLD \
    -tr nnUNetTrainerDA5FiLM

# 100-epoch variant
nnUNetv2_train DATASET_ID 3d_fullres FOLD \
    -tr nnUNetTrainerDA5FiLM_100epochs
```

Everything else (planning, preprocessing, fold structure) is unchanged.

## Inference

### Default (baseline, no conditioning)

Standard nnU-Net inference works out of the box:

```bash
nnUNetv2_predict -i INPUT -o OUTPUT -d DATASET_ID \
    -tr nnUNetTrainerDA5FiLM -c 3d_fullres
```

### With disease conditioning (recommended)

During training, the trainer automatically copies `disease_map.json` to the model
output folder.  The dedicated inference script auto-detects this file and sets the
disease vector per case:

```bash
python -m nnunetv2.inference.predict_disease_conditioned \
    -i /path/to/input \
    -o /path/to/output \
    -m /path/to/model_folder \
    -f 0
```

You can also point to a custom disease map:

```bash
python -m nnunetv2.inference.predict_disease_conditioned \
    -i /path/to/input \
    -o /path/to/output \
    -m /path/to/model_folder \
    --disease_json /path/to/disease_map.json
```

The script:
1. Looks for `disease_map.json` in the model folder (auto-copied during training)
2. Falls back to `--disease_json` if provided explicitly
3. If neither is found, runs baseline inference (no conditioning)
4. Groups input files by case ID, sets the disease vector per case, predicts

### Programmatic API (advanced)

For custom workflows, you can use the network's attribute API directly:

```python
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
import json, torch

predictor = nnUNetPredictor(...)
predictor.initialize_from_trained_model_folder(model_folder, ...)

with open("disease_map.json") as f:
    disease_map = json.load(f)

for case_id, input_file, output_file in cases:
    vec = torch.tensor([disease_map[case_id]], dtype=torch.float32, device=predictor.device)
    mod = predictor.network
    if hasattr(mod, '_orig_mod'):
        mod = mod._orig_mod
    mod.set_disease_vec(vec)
    predictor.predict_from_files([[input_file]], [output_file], ...)
    mod.clear_disease_vec()
```

## Checkpoint Compatibility

| Checkpoint source          | Loading into              | Behaviour                                    |
|----------------------------|---------------------------|----------------------------------------------|
| Vanilla ResEncUNet         | FiLM trainer              | OK (`strict=False`); FiLM modules use zero-init, warning logged |
| FiLM trainer               | FiLM trainer              | OK (full match)                              |
| FiLM trainer               | Vanilla nnUNetTrainer     | **Fails** (unexpected keys); use `strict=False` manually |

## Architecture Details

```
                     disease_vec (B, K)
                          │
                    ┌─────▼─────┐
                    │ disease_mlp│  Linear(K→H) + ReLU + Linear(H→E)
                    └─────┬─────┘
                          │ e (B, E)
                          │
   Input ──► Encoder ──► Bottleneck ──► FiLM(bottleneck, e) ──┐
                │              ▲                                │
                │ (skips)      │                                │
                │              │                         ┌──────▼──────┐
                │              │                         │  Decoder s=0│
                │              │                         └──────┬──────┘
                │              │                         FiLM(features, e)
                │              │                                │
                │              └─── ... (repeat all stages) ────┘
                │
                └──────► skip connections ──────────────────────► concat
```

Each **FiLMLayer** does:

```
e  →  gamma_head(e)  →  gamma (B, C)     ─┐
                                            ├──►  y = (1 + gamma) * x + beta
e  →  beta_head(e)   →  beta  (B, C)     ─┘
```

- `gamma_head` and `beta_head` are `Linear(E, C)`, zero-initialised.
- At init: gamma=0, beta=0 → y = x (identity, exact baseline).
- gamma and beta are broadcast to spatial dims automatically (2D and 3D).
- When `disease_vec=None`, FiLM layers are completely bypassed.

### FiLM vs MLP+Injection (Option A)

| Aspect              | FiLM (this branch)                      | MLP+Injection (MLPembeddingBaseline)      |
|---------------------|-----------------------------------------|-------------------------------------------|
| Modulation          | Channel-wise scale + shift              | Concat + Conv1×1 + Norm + NonLin          |
| Extra params/stage  | 2 × Linear(E, C)                       | Conv(C+E, C) + Norm(C) + NonLin           |
| Compute overhead    | Tiny (linear + broadcast + mul/add)     | Moderate (conv + norm + nonlin)            |
| Init behaviour      | Exact identity (zero-init)              | Near-identity (He-init, random)            |
| Spatial interaction | None (channel-wise only)                | 1×1 conv mixes channels                   |

## Running Sanity Tests

```bash
python3 -m nnunetv2.tests.test_film_conditioning
```

Tests verify:
1. Output shapes match between baseline and conditioned paths (3D + 2D, with/without DS).
2. FiLM head parameters receive non-zero gradients at zero-init.
3. Full gradient flow (disease_mlp) with non-zero FiLM heads.
4. The `set_disease_vec()` / `clear_disease_vec()` inference API works.
5. Zero-init identity: output with zero disease_vec matches baseline exactly.
6. Ablation smoke test: different disease vectors produce different outputs.

## Files Added

| File | Purpose |
|------|---------|
| `nnunetv2/architectures/__init__.py` | New package |
| `nnunetv2/architectures/film_conditioned_unet.py` | `FiLMLayer` + `FiLMConditionedResEncUNet` |
| `nnunetv2/training/.../nnUNetTrainerDA5FiLM.py` | `nnUNetTrainerDA5FiLM` + `_100epochs` variant |
| `nnunetv2/inference/predict_disease_conditioned.py` | Standalone disease-conditioned inference CLI |
| `nnunetv2/tests/test_film_conditioning.py` | Sanity tests |
| `docs/film_conditioning.md` | This document |
