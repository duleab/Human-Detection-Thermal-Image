# Thermal UAV search-and-rescue research

This repository begins the PhD project **“Occlusion- and Uncertainty-Aware Real-Time Human
Detection and Tracking in Thermal Imagery from Unmanned Aerial Vehicles for Search and Rescue.”**
Notebooks 01–11 are implemented. Civilian human-presence detection and anonymous tracks are
the intended scope. Architectural improvements, novelty, robustness, and real-time performance
are hypotheses requiring experiments.

Notebook 02 is [Dataset Audit and Annotation Standardization](notebooks/02_Dataset_Audit_and_Annotation_Standardization.ipynb).
Its [run guide](docs/Notebook02.md) covers Kaggle/Colab mounts, the explicit dataset profile,
schema contracts, guarded artifacts, and limitations. Real data are intended to be mounted
on Kaggle/Colab; local checks use clearly labeled synthetic fixtures. Missing inputs produce
a blocked report, never a fabricated dataset manifest.

Notebook 03 is [Exploratory Analysis and Visibility Annotation](notebooks/03_Exploratory_Analysis_and_Visibility_Annotation.ipynb).
Its [run guide](docs/Notebook03.md) covers verified manifest loading, development-only EDA,
stratified reviewer packages, two-rater agreement, explicit adjudication, and masked sidecar
labels. Real inputs/reviewer labels are still pending remote execution.

Notebook 04 is [Leakage Control and Evaluation Splits](notebooks/04_Leakage_Control_and_Evaluation_Splits.ipynb).
Its [run guide](docs/Notebook04.md) covers preserved official provenance, independent research
partitions, verified/duplicate constraints, quarantine, reserved calibration/test data and
immutable fingerprints. Missing inputs produce a blocked report and null fingerprint.
Notebook 05 is [Data Loaders, Preprocessing and Augmentation](notebooks/05_Data_Loaders_Preprocessing_and_Augmentation.ipynb).
Its [run guide](docs/Notebook05.md) covers verified frozen-split loading, thermal decoding,
paired transforms, auxiliary masks, optional causal evaluation windows and bounded CPU checks.
Missing real inputs produce a blocked report; synthetic checks remain explicitly labeled.
Notebook 06 is [Baseline Architecture and Training](notebooks/06_Baseline_Architecture_and_Training.ipynb).
Its [run guide](docs/Notebook06.md) covers offline FCOS initialization, person adaptation,
isolated diagnostics, optional COCO development evaluation and explicitly approved complete
training with immutable epoch checkpoints and resume. Default real execution remains blocked
until successful upstream data and local weights are provided. Only bounded random-weight
fixture checks have run; no complete baseline training or installation was performed.
Notebook 07 is [Proposed Architecture Implementation](notebooks/07_Proposed_Architecture_Implementation.ipynb).
Its [run guide](docs/Notebook07.md) covers the optional high-resolution pathway, image-conditioned
fusion, residual feature gates, metadata fallback, visibility/variance heads and independent
switches. Real comparisons require the missing trained Notebook 06 baseline. Explicit random
fixtures validate software only; supervised auxiliary objectives belong to Notebook 08.
Notebook 08 is [Loss Functions and Theoretical Checks](notebooks/08_Loss_Functions_and_Theoretical_Checks.ipynb).
Its [run guide](docs/Notebook08.md) covers native FCOS assignment/loss reuse, masked visibility,
diagonal localization likelihood, explicit regression weighting and conditional fusion bounds.
Analytic software checks run without real data; real-model readiness remains blocked upstream.
No optimizer or complete training runs in Notebook 08. Stop after Notebook 08.
Notebook 09 is [Proposed Model Training and Validation](notebooks/09_Proposed_Model_Training_and_Validation.ipynb).
Its [run guide](docs/Notebook09.md) covers the implemented weighted objective, metadata-aware
development evaluation, isolated preflight, complete checkpoint resume and explicit full-training
approval. Default Run All performs no updates; missing upstream inputs remain blocked.
Notebook 10 is [Ablations and Comparative Experiments](notebooks/10_Ablations_and_Comparative_Experiments.ipynb).
Its [run guide](docs/Notebook10.md) explains the variant registry, common warm start,
verified development results, seed coverage and final selection. Locally, the registry
reports missing research inputs and exports an empty results table.
Notebook 11 is [Confidence Calibration and Uncertainty Validation](notebooks/11_Confidence_Calibration_and_Uncertainty_Validation.ipynb).
Its [run guide](docs/Notebook11.md) describes grouped held-out calibration, the operating
threshold, review cues and conditional localization checks. The local run reports missing
frozen inputs. The optional synthetic check exercises the calibration arithmetic.

## Notebook 01: purpose, inputs, and outputs

Open [01_Research_Configuration_and_Environment.ipynb](notebooks/01_Research_Configuration_and_Environment.ipynb).
It establishes the protocol, baseline choice, configuration, environment diagnostics, and seeds.
It requires `src/`, `configs/base.yaml`, Python 3.11+, PyYAML, and nbformat. Fresh-kernel automation
also needs nbclient/ipykernel. A dataset, pretrained weights, GPU, and internet are unnecessary.

Every execution creates `artifacts/experiments/<name>_<UTC-time>_<hash>_<random-id>/` locally,
or `/kaggle/working/artifacts/experiments/<unique-id>/` on Kaggle. It writes:

- `config.json`: effective settings; `run.json`: experiment ID, paths, SHA-256, and stage.
- `environment.json`: interpreter, package versions, hardware, errors, CPU NMS diagnostic,
  and the actual installed FCOS factory source and its hash when available.
- `reproducibility.json`: actual Python/NumPy/torch seed status and determinism settings.
- `dependency_readiness.json`: current and future dependency gaps, separate from data readiness.
- `source_hashes.json`: provenance for the configuration and shared code.
- `packages.txt`: project dependency inventory, explicitly not a complete environment lockfile.
- Empty artifact subdirectories for subsequent work; no result tables or model checkpoints.

Files use exclusive creation. Re-running Cell 14 refuses to overwrite its artifacts; use Run All
for a new run directory. Dataset versions and split fingerprints stay null until verified.
Draft manifest schema versions reserve a future contract; row validators belong to Notebook 02.

## Concrete baseline and dependency decisions

The proposed baseline is **torchvision FCOS ResNet-50-FPN**. The installed 0.22.1 source was
inspected: its returned ResNet layers `[2, 3, 4]` supply C3–C5 to the 256-channel FPN;
`LastLevelP6P7` extends it to P3–P7, nominal strides 8–128. It has separate encoder/FPN/head
interfaces and no default P2. The high-resolution pathway is a later controlled modification.
No detector factory or training API is introduced in this stage.

The thermal-only input proposal repeats the thermal channel three times, preserving the
reference encoder. Decoding, scaling, normalization, class mapping, and person-head adaptation
must be verified in later notebooks. Reference resize bounds are provisionally 800/1333 pixels.
COCO_V1 pretraining is proposed for Notebook 06; the current weight source is `none` and all
downloads are disabled. Configure `model.weights_source: local` and `local_weights_path` for
an offline weight file; Notebook 01 verifies file presence without deserializing it.

Primary references: [versioned FCOS source](https://github.com/pytorch/vision/blob/v0.22.1/torchvision/models/detection/fcos.py),
[backbone source](https://github.com/pytorch/vision/blob/v0.22.1/torchvision/models/detection/backbone_utils.py),
[official factory/weights documentation](https://docs.pytorch.org/vision/stable/generated/torchvision.models.detection.fcos_resnet50_fpn.html),
and [reference COCO evaluation code](https://github.com/pytorch/vision/tree/v0.22.1/references/detection).
Torchvision's code uses the [BSD-3-Clause license](https://github.com/pytorch/vision/blob/main/LICENSE).
Review pretrained-weight and dataset terms separately before use or redistribution.

HIT-UAV is the first candidate: inspect the actual mounted copy against its
[upstream documentation](https://github.com/suojiashun/HIT-UAV-Infrared-Thermal-Dataset) and
[license](https://github.com/suojiashun/HIT-UAV-Infrared-Thermal-Dataset/blob/main/LICENSE) in
Notebook 02. Collection-level claims do not establish per-image metadata availability.

The inspected local runtime is Windows, Python 3.13.3, torch **2.7.1+cpu**, torchvision
**0.22.1+cpu**, NumPy 2.4.6, PyYAML 6.0.2, and nbformat 5.10.4. CPU NMS works; CUDA is
unavailable. There is no dataset in this workspace and no `/kaggle/input` mount. No dataset
search outside the project is assumed complete. `pycocotools` is missing, so the later detection
evaluator is not ready. The notebook re-inspects the runtime each execution.

No packages were installed or upgraded. `requirements.txt` describes Notebook 01 execution
dependencies. If these are missing, review the minimal proposed command
`python -m pip install -r requirements.txt` before authorizing environment changes. The later
evaluator needs `python -m pip install pycocotools`. Select a matched torch/torchvision pair for
the target platform using the official PyTorch installer guidance; do not blindly upgrade Kaggle.

## Cell-by-cell implementation

| Cell | Type | Role |
|---|---|---|
| 1 | Markdown | Research scope, purpose, inputs, outputs |
| 2 | Markdown | Research questions and falsifiable hypotheses |
| 3 | Markdown | Proposed contributions, transparent baseline and primary sources |
| 4 | Markdown | Metrics, denominators, grouping, leakage controls |
| 5 | Markdown | Stages, dependencies, deferred manifest contracts |
| 6 | Code | Imports and project/config discovery |
| 7 | Code | Python, packages, hardware, FCOS source, CPU NMS |
| 8 | Code | Seed and deterministic-algorithm configuration |
| 9 | Code | Central stage-validated configuration |
| 10 | Code | Actual input mounts, dataset selection, writable paths |
| 11 | Code | Configuration hash and run metadata |
| 12 | Code | Unique experiment ID and artifact directories |
| 13 | Code | Required/optional dependency readiness |
| 14 | Code | Immutable artifacts and saved-configuration hash verification |
| 15 | Markdown | Reproducibility limits and unsupported claims |
| 16 | Markdown | Readiness checklist, local/Kaggle instructions, stop condition |

Shared modules: `src/config.py` implements validated YAML, canonical hashes, paths, and
exclusive artifacts; `src/reproducibility.py` implements dependency/source/hardware diagnostics
and generator seeding. Tests cover safety and reproducibility contracts, not detector accuracy.

## Running locally

1. Use the intended Python interpreter; inspect dependencies before modifying its environment.
2. Open the notebook from the project root or `notebooks/` in Jupyter, select that kernel,
   restart it, and Run All. Do not execute from a kernel that already initialized CUDA.
3. Edit `configs/base.yaml` before execution. Explicit dataset paths are resolved relative to
   the project root unless absolute; missing configured paths raise an error. Null means no
   selection. The dataset version and licensing status require evidence, not a guessed value.
4. Inspect the unique run directory printed in Cell 12 and the readiness report from Cell 13.

Optional environment variables `THERMAL_PROJECT_ROOT`, `THERMAL_DATASET_ROOT`, and
`THERMAL_CONFIG_PATH` select explicit code, dataset, and configuration locations. The notebook
also exposes the corresponding editable hints in Cell 6. For hash randomization control set
`PYTHONHASHSEED=42` **before launching the kernel**; setting it in a running kernel is ineffective.

Lightweight checks from the project root:

```powershell
python -m unittest discover -s tests -v
python -m ruff check src tests notebooks tmp/jupyter-notebook
python -m mypy src
python -m bandit -r src -ll
```

The authoring/verification helpers under `tmp/jupyter-notebook/` are optional. The execution
helper uses nbclient to start a fresh IPython kernel, executes the delivered notebook, and
saves execution counts and stdout under that run's `logs/` directory. It does not train.

## Running on Kaggle

1. Attach a project code dataset containing `src/`, `configs/`, and `notebooks/`, and import
   this notebook into a Python notebook. It discovers actual mount names; if ambiguous, set
   `PROJECT_ROOT_HINT` in Cell 6 to the real directory containing `src/` and `configs/`.
2. Optionally attach HIT-UAV and set `DATASET_ROOT_HINT` to its real root. Notebook 01 requires
   no dataset. It lists immediate mounted directories, without identifying them from names alone.
3. To edit configuration from a read-only code mount, copy `configs/base.yaml` to
   `/kaggle/working/base.yaml`, edit the copy, and set `CONFIG_PATH_HINT` in Cell 6.
4. Keep `/kaggle/input` read-only. Generated outputs must be inside `/kaggle/working`.
5. Internet and accelerator can remain off. Restart the kernel and Run All. Save/download the
   resulting `/kaggle/working/artifacts/experiments/<unique-id>/` folder.

Expected runtime category: **smoke test**, seconds to a few minutes depending on import and
storage overhead. This is not a full experiment or a GPU performance measurement.

## Notebook 01 completion boundary

Notebook 01 is ready for use when all nine code cells execute, the audit bundle exists, and
the saved configuration hash verifies. Tests and fresh-kernel evidence are recorded in the
delivered validation report under `artifacts/` once executed.

Notebook 01 itself does not audit data, parse annotations, create splits, construct models,
train, calibrate, track, or evaluate performance. Notebook 02 now implements the data audit
and annotation contracts described in its separate guide. Real
Kaggle and GPU execution require those environments; simulated path tests do not establish
hardware compatibility. No training, downloads, installs, or deletion are performed here.

Notebook 01 stops at its own boundary. Notebook 02 was subsequently authorized and implemented;
its data-dependent entry requirement is an accessible HIT-UAV copy with provenance/license
files. Notebook 03 was subsequently authorized and implemented; missing visibility or track
labels remain unavailable. Notebook 04 was subsequently authorized and implemented.
Notebook 05 was subsequently authorized and implemented.
Notebook 06 was subsequently authorized and implemented.
Notebook 07 was subsequently authorized and implemented as architecture/software support.
Notebook 08 was subsequently authorized and implemented as objective/theory support.
Notebook 09 was subsequently authorized and implemented as guarded proposed-training support.
Notebook 10 implements the controlled ablation registry and guarded FCOS execution.
Notebook 11 implements the frozen calibration and localization validation workflow.
**Stop after Notebook 11.**
"# Human-Detection-Thermal-Image" 
