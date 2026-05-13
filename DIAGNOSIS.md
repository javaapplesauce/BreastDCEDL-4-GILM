# DIAGNOSIS: replication gap in BreastDCEDL ViT pCR pipeline

Branch: `fix/preprocessing-and-pooling`
Audience: PI / co-supervisor reviewing the fix set before the next A100 run.

Status before this branch: best val AUC 0.66, test AUC 0.53 across all
cohorts, against the paper's reported 0.74 overall and 0.94 on
HR+/HER2− + clinical. After the changes below, the pipeline still
needs an actual training run to confirm — see "What I could not verify".

## Fixes

### 1. RGB normalization was per-slice and destroyed the DCE signal

`fuse_rgb_slice` called `_minmax_uint8` separately on each of pre,
early-post, and late-post. Per-channel rescaling to [0, 255]
independently means the relative intensity between timepoints — the
entire signal of DCE-MRI — is normalized away. A strongly-enhancing
slice and a non-enhancing slice produce per-channel means within a few
units of each other.

Change: added `_percentile_uint8_volume` in
`src/data/preprocessing.py`. Pools nonzero voxels across all three
patient volumes, computes one (lo, hi) pair from configurable
percentiles, then maps each volume to uint8 with that SHARED range.
`BreastDCEDataset.__getitem__` normalizes whole volumes before
slicing. The old per-slice path is kept behind
`data.normalization=minmax_per_slice` for an explicit ablation. New
default: `percentile` with `percentile_lo=1.0`, `percentile_hi=99.0`.

Verified empirically: on synthetic volumes with R=0.2, G=1.0, B=0.6
intensity at the tumor, percentile output preserves the G > B > R
ordering at tumor voxels and keeps air voxels near zero. Per-slice
MinMax collapses an enhancing slice and a flat slice to identical
per-channel means (max diff < 5 units out of 255); percentile keeps
them >30 units apart. See `tests/test_normalization.py`.

### 2. Patient-level pooling silently mis-aligned labels

`patient_level_eval` and `predict_with_tta` reshaped a flat logits
tensor as `(n_patients, n_slices, -1)`. This assumed three things
were all true:
- the loader returned slices in strict patient-contiguous order
- no `__getitem__` exception ever fired — `BreastDCEDataset._blank`
  silently returned a zero tensor with the patient's label, breaking
  contiguous ordering with a phantom row
- `drop_last=False` didn't trim the tail

Any violation silently mis-aligned labels to predictions. The metric
went down, no error was raised. We have no telemetry showing how often
`_blank` fired in prior runs; even if zero, the design is one bad
NIfTI away from a silent regression.

Change:
- `BreastDCEDataset.__init__` pre-resolves every patient via
  `load_acquisitions` and drops rows that return None or raise, with
  a one-line warning listing up to 10 dropped pids.
- `__getitem__` returns `(img, label, pid)` (or with clinical:
  `(img, label, clinical, pid)`). The `_blank` fallback is gone;
  transient post-construction failures raise loudly with the pid.
- `patient_level_eval(logits, labels, patient_ids)` and
  `predict_with_tta(...) -> (..., patient_ids)` pool via a dict keyed
  on pid. Order-invariant.
- `Trainer.run_epoch` threads pids through and `Trainer.train` feeds
  `vl_pids` to `patient_level_eval`.
- Notebook eval and clinical sub-run cells unpack the 4-tuple and
  reorder `test_df`/`val_df` by the returned pids so the existing
  positional subgroup/cohort code continues to work.

Verified: synthetic-data agreement between new bucket pool and old
reshape pool is exact (< 1e-6) on the happy path; new pool is also
invariant under permutation. Old code is not, and that's the bug.

### 3. Pretrained weight load was only partly auditable

The previous load printed only `missing=N unexpected=M`. The pooler
question raised in the spec — whether `pooler.dense.*` from the paper
checkpoint actually maps into anything that gets used — was not
visible.

Reading the transformers source: `ViTForImageClassification.forward`
returns `logits = self.classifier(sequence_output[:, 0])`, i.e. it
uses the CLS token directly and never calls `self.vit.pooler` on the
classification path. So any `pooler.*` weights in the paper
checkpoint are loaded into `encoder.vit.pooler.*` (the key remap path
gives them a home) but no forward pass reads from them. This is fine
in the sense that nothing is broken, but it does mean the "classifier
warmstarted from paper checkpoint" log is the ONLY paper-specific
weight on the classification path beyond the ImageNet-21k init.

Change: `load_pretrained(path, debug=True)` now prints every
transferred key, every dropped key with a reason (no target key /
shape mismatch / classifier handled separately), and every key we own
that the checkpoint did not provide. Defaults to `debug_pretrained:
true` in the config so the next run emits the full audit; flip off
later for quieter logs.

Verified: model construction still works under `debug=True`; the
listing should make any future "we silently dropped X" failure
visible. The actual list of what gets transferred from the real paper
checkpoint can only be confirmed during a Colab run with the .pth on
disk (see "What I could not verify").

### 4. Augmentation policy was breaking the data

`vertical_flip: true` flips superior/inferior in breast MRI — a
clinically meaningful axis, not a symmetry. `rotation: 15`,
`translate: [0.1, 0.1]`, `scale: [0.9, 1.1]` were on the aggressive
end for fine-tuning a pretrained ViT on medical data. `random_erasing:
0.25` could erase the tumor itself.

Change in `configs/default.yaml`: vertical_flip off, rotation 5,
translate 0.05/0.05, scale 0.95/1.05, random_erasing 0.0.

Verified: config change only. No code path was hardcoded on the old
augmentation values.

### 5. Patience too long, dropout too high

`patience: 10` plus the previous slow-learning phase 1 meant the run
in run9.log waited 26 epochs before stopping at a flat plateau.
`model.dropout: 0.3` is heavy for a small head warming up on a frozen
backbone.

Change: `patience` 10 → 7, `model.dropout` 0.3 → 0.1. No hardcoded
dropout values remain in `src/`.

### 6. Class weights were rescaled to sum to 1

`build_class_weights` returned `weights / weights.sum()`. For a 71/29
split the ratio stayed right but gradient magnitudes shrank by ~3x.
Combined with focal gamma=2, phase 1 head-warmup learned almost
nothing — run9.log shows val_acc 0.301, sens 1.0, spec 0.0 across
epochs 1–6 (predicting pCR for everyone with the head still
collapsed).

Change: `w_c = N / (num_classes * count_c)`. Also threaded
`training.loss_gamma` (default 2.0) through `scripts/train.py` so
gamma in {0, 1, 2} can be ablated without code changes.

Verified: on the actual training split (1083 patients, 29.6% pCR)
the new weights come out [0.711, 1.687], within 5% of the paper-spec
target [0.704, 1.724]. See `tests/test_class_weights.py`.

### 7. Determinism: incomplete seeding

`torch.manual_seed(seed)` and `np.random.seed(seed)` were set but
`random.seed`, `torch.cuda.manual_seed_all`, and DataLoader worker
seeding were not. DataLoader workers re-seed numpy / random per
worker with non-deterministic values, so augmentation choices were
not reproducible run-to-run.

Change: added `random.seed(seed)` and
`torch.cuda.manual_seed_all(seed)` alongside the existing seeds; added
`_worker_init_fn` and passed it to both train and val loaders.
Deliberately NOT setting `torch.use_deterministic_algorithms(True)`
— it breaks some attention ops on A100 and offers limited additional
guarantee. Documented in a comment.

Verified: code change only; full run-to-run bit-equivalence requires
a real GPU and is out of scope here.

## Verification results

```
$ python -m pytest tests/ -v
tests/test_class_weights.py::test_weights_for_71_29_split PASSED         [  6%]
tests/test_class_weights.py::test_weights_for_balanced_split PASSED      [ 13%]
tests/test_class_weights.py::test_weights_handle_empty_class PASSED      [ 20%]
tests/test_class_weights.py::test_weights_are_not_rescaled_to_sum_one PASSED [ 26%]
tests/test_normalization.py::test_percentile_preserves_channel_order_at_tumor PASSED [ 33%]
tests/test_normalization.py::test_percentile_keeps_air_near_zero PASSED  [ 40%]
tests/test_normalization.py::test_per_slice_minmax_destroys_cross_channel_magnitude PASSED [ 46%]
tests/test_normalization.py::test_percentile_handles_all_zero_volumes PASSED [ 53%]
tests/test_normalization.py::test_percentile_handles_degenerate_lo_hi PASSED [ 60%]
tests/test_pooling.py::test_new_pool_matches_legacy_on_happy_path PASSED [ 66%]
tests/test_pooling.py::test_new_pool_invariant_under_reorder PASSED      [ 73%]
tests/test_pooling.py::test_new_pool_handles_unequal_slice_counts PASSED [ 80%]
tests/test_pooling.py::test_new_pool_raises_on_length_mismatch PASSED    [ 86%]
tests/test_smoke.py::test_imports_from_fresh_python PASSED               [ 93%]
tests/test_smoke.py::test_forward_backward_random_input PASSED           [100%]
======================== 15 passed, 2 warnings in 6.04s ========================
```

Smoke test details: `test_forward_backward_random_input` builds
`BreastDCEViT` on CPU with no pretrained weights, pushes a
2×3×224×224 random batch through, computes FocalLoss against random
labels, calls `loss.backward()`, and asserts the loss is finite and
at least one parameter received a non-zero gradient. Passes in ~6s.

## What I could not verify

These need the actual A100 + real data to confirm. Listed in priority
order for the next Colab run:

1. **3-epoch sanity cell** (notebook cell 9a). With the new
   normalization, class weights, and head bias-init, val AUC at
   epoch 3 on a 100-patient stratified subset should be noticeably
   above the previous-pipeline baseline of ~0.55–0.60 at the same
   number of epochs. If it isn't, something in this branch regressed
   and the rest of the run is wasted A100 time.

2. **pretrained-weight debug log**. With `debug_pretrained: true`
   on, the first training run should print every transferred and
   dropped key from `BreastDCEDL_models/...pth`. Check for: pooler
   keys (expected to map to `encoder.vit.pooler.*` and then be unused
   by forward; that's the audit point — confirm shape match exists
   even if forward bypasses), and the head weight+bias pair being
   marked "classifier warmstarted from paper checkpoint".

3. **Pre-resolve overhead**. Pre-resolving 1,000–1,400 patients via
   `load_acquisitions` at dataset construction will add several
   minutes before training starts. If this is unacceptable, the
   correctness goal can be met by a cheaper file-existence check
   (drop patients with fewer than 2 NIfTIs on disk). The full decode
   was kept because the spec asked for it; flag if it's a problem in
   practice.

4. **Full 30-epoch run**. Whether the combined fixes actually close
   the gap to AUC 0.74. Each fix on its own is independently
   defensible; whether they compose to the paper's result is empirical.

## If results still don't improve

Top 3 next investigations, in order of expected payoff:

1. **Slice selection**. The paper uses 7 mid-plane slices; we use 8.
   Per-patient slice indexing currently picks 8 contiguous slices
   centered on the tumor mid-plane, which could include 1 slice past
   the tumor boundary depending on `find_tumor_z_range` width. Verify
   on a few patients via the cell-7 visualization that the chosen
   slices are all in-tumor.

2. **TTA flip semantics**. 4-view TTA includes vertical flip, which
   contradicts Fix 4's removal of vertical_flip from training-time
   augmentation. If vertical flip is genuinely not a symmetry for
   breast MRI, it shouldn't be in TTA either; drop to identity +
   horizontal flip only (2-view).

3. **Cohort-balanced sampling**. Duke is over-represented in raw
   counts and was excluded entirely from the paper's headline number.
   A WeightedRandomSampler weighted by cohort could help; today
   Duke's larger sample count dominates the gradient.

If after all three the test AUC is still well below 0.74, the gap
likely sits in the architecture/training recipe — at that point worth
re-reading paper Section 2 for any detail not in this branch (e.g.
LR schedule, exact freeze/unfreeze timing, head architecture).
