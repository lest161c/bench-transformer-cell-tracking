# Et-Ultra fork bridge (SPEC 0032, ticket T0)

Status: design complete, runtime UNVERIFIED. T3 (cluster adapter smoke) must
confirm the items under "To verify" before any T8 training is launched.

## Problem

- Achard's fork `scripts/train.py` resolves feature backbones from
  `trackastra.data.pretrained_features.AVAILABLE_PRETRAINED_BACKBONES`, a
  self-contained module (2121 lines) that duplicates the feature-extraction API.
- Our adapter `benchmark_ssl/src/etultra_dynaclr/dynaclr_features.py` registers
  `dynaclr-scaledcnn` into the *different* registry shipped by the
  `trackastra_pretrained_feats` package (bentaculum/Trackastra-et-Ultra).
- The fork instantiates a backbone as
  `backbone(image_size=..., save_path=..., device=..., mode=...)` (from
  `FeatureExtractor.from_model_name`) and, via `from_config`, also passes
  `batch_size`, `additional_features`, `normalize_embeddings`, `seed`. Our
  adapter reads its weights from `checkpoint_path`, not `model_folder`.

## Mechanism

`benchmark_ssl/src/etultra_dynaclr/fork_registry_bridge.py` (same directory as
the adapter) defines `ForkScaledCNNDynaCLRFeatures(ScaledCNNDynaCLRFeatures)`,
maps `model_folder` -> `checkpoint_path` and registers it into the fork registry
as `dynaclr-scaledcnn`. It auto-registers on import.

## Loading (must precede argparse construction in `scripts/train.py`)

1. `export PYTHONPATH=$BENCH/benchmark_ssl/src:$PYTHONPATH`
2. Add one line near the top of `$BENCH/cyril_trackastra/scripts/train.py`:
   `import etultra_dynaclr.fork_registry_bridge  # noqa: F401`
   (Alternative, if the fork file must stay untouched: append the same import to
   the tail of `trackastra/data/pretrained_features.py`.)
3. Run with `--features pretrained_feats --pretrained_feats_model
   dynaclr-scaledcnn --pretrained_model_path <checkpoint>`.

## To verify on the cluster (T3)

- After the import, `dynaclr-scaledcnn` appears in
  `AVAILABLE_PRETRAINED_BACKBONES` and in `train.py --help` choices.
- `--pretrained_model_path` -> `model_folder` -> `checkpoint_path`: build the
  extractor through `PretrainedFeatureExtractorConfig`/`from_config` and assert
  the frozen encoder loaded (no ValueError, no state_dict mismatch).
- Synthetic 1024x1024 frame + 2 masks -> `compute_region_features` -> (2, 128),
  no NaN.
- TOP RISK: the augmentation path. His configs use
  `features: pretrained_feats_aug`. `FeatureExtractorAugWrapper._compute` calls
  `precompute_image_embeddings(images, image_shape=...)` (our override ignores
  `image_shape` and only stores normalised frames, returning None) and then
  `wrfeat.WRAugPretrainedFeatures.from_mask_img(...)`. Confirm that path routes
  to our overridden `compute_region_features` and never to `_run_model`/the base
  patch-grid. If it needs patch-grid embeddings, either implement `_run_model`
  or fall back to `features: pretrained_feats` (non-augmented).
- `FeatureExtractorAugWrapper` reads `extractor.final_grid_size` (debug view) and
  `orig_image_size`; our adapter sets both in `__init__`.
