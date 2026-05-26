# Phase 4 Prescription Recognition Pipeline

This toolkit goes from raw prescription photos in `data/raw` to OCR-ready word crops, word-level TrOCR predictions, and structured medicine output.

## Final Decision

Use this as the thesis-ready pipeline:

1. Preprocess full prescription pages.
2. Detect handwritten prescription regions with the trained YOLO region model.
3. Segment lines inside each region with the hybrid OpenCV line segmenter.
4. Segment words from each line crop with connected-component word segmentation.
5. Run word-level TrOCR on the word crops.
6. Apply lexicon/fuzzy matching, dosage/frequency extraction, and drug database validation.

Do not use the YOLO line detector as the final method right now. Your region YOLO is strong, but the line YOLO result is weak on the current small line dataset. Keep line YOLO only as an optional ablation/comparison experiment.

## Augmentation Policy

Raw/full-page augmentation for handwritten-region YOLO is optional and can be skipped for now. Your current region detector already has excellent validation results:

- precision: `0.994`
- recall: `1.000`
- mAP50: `0.995`
- mAP50-95: `0.929`

Only retrain the region model with more raw-page augmentation if it fails on new camera angles, new forms, or different lighting.

Use augmentation where it helps most: word-level OCR training. The script `pipeline/scripts/augment_ocr_dataset.py` creates a train-only augmented OCR dataset using mild rotate/shift/scale/brightness/noise changes. Validation and testing stay unchanged.

## Notebook Order

Run notebooks in this order:

1. `phase4_region_line_segmentation_colab.ipynb`
   - preprocesses `data/raw`
   - trains or reuses `models/region_yolo_best.pt`
   - runs final segmentation into regions, lines, and words
   - output: `data/final_region_hybrid_line_word/word_manifest.csv`

2. `phase4_full_prescription_annotation_tools.ipynb`
   - optional review/annotation notebook for word crops
   - exports `data/custom_word_ocr_dataset`
   - creates `data/custom_word_ocr_dataset_augmented`

3. `phase4_trocr_word_level.ipynb`
   - trains/evaluates word-level TrOCR
   - auto-prefers `data/custom_word_ocr_dataset_augmented`
   - runs final full-prescription inference with region YOLO + hybrid line/word segmentation

Stable model paths:

```text
models/region_yolo_best.pt
models/line_yolo_best.pt   # optional experiment only
```

## Colab Paths

In Colab, keep the repo under:

```text
/content/drive/MyDrive/phase4_project/repo
```

Put raw prescription images in:

```text
/content/drive/MyDrive/phase4_project/repo/data/raw
```

Generated crops, manifests, datasets, and models will remain on Drive.

## Install

```bash
python3 -m pip install -r pipeline/requirements.txt
```

For YOLO training:

```bash
python3 -m pip install -r pipeline/requirements-layout.txt
```

For TrOCR:

```bash
python3 -m pip install -r pipeline/requirements-ocr.txt
```

## Core CLI Flow

### 1. Preprocess Pages

```bash
python3 pipeline/scripts/preprocess_pages.py \
  --input-dir data/raw \
  --output-dir data/processed/pages \
  --manifest-out data/processed/page_manifest.csv
```

### 2. Prepare Region Annotation Package

```bash
python3 pipeline/scripts/prepare_layout_annotation.py \
  --pages-dir data/processed/pages \
  --output-dir data/processed/layout_annotation_package \
  --copy-images
```

Annotate `handwritten_region` boxes, then export YOLO labels if needed.

### 3. Train Region YOLO

If `models/region_yolo_best.pt` already exists and works well, skip retraining.

```bash
python3 pipeline/scripts/prepare_yolo_layout_dataset.py \
  --page-manifest data/processed/page_manifest.csv \
  --region-manifest data/processed/region_manifest.csv \
  --output-dir data/layout_yolo_dataset

python3 pipeline/scripts/train_yolo_layout.py \
  --data-yaml data/layout_yolo_dataset/data.yaml \
  --model yolov8n.pt \
  --epochs 80 \
  --imgsz 960 \
  --batch 8 \
  --weights-out models/region_yolo_best.pt
```

### 4. Final Segmentation: Region YOLO + Hybrid Lines + Words

```bash
python3 pipeline/scripts/run_end_to_end.py \
  --input data/raw \
  --output-dir data/final_region_hybrid_line_word \
  --yolo-model models/region_yolo_best.pt \
  --target-class 0 \
  --ocr-backend none \
  --ocr-unit word \
  --line-padding 6
```

Outputs:

- `page_manifest.csv`
- `region_manifest.csv`
- `line_manifest.csv`
- `word_manifest.csv`
- `segmentation_review.csv`
- region crops
- line crops
- word crops
- context preview images
- `segmentation_overview/` paper/PPT-ready overview images with original, binary, and numbered box panels

The line and word segmenters use the final hybrid method: adaptive connected components, projection/gap analysis, and lightweight low-ink separator refinement. This is the default path for thesis figures and OCR-ready word crops. Keep pretrained detectors such as Kraken/CRAFT/DBNet/PaddleOCR as optional ablations only.

### 5. Build Word OCR Dataset

After reviewing word annotations:

```bash
python3 pipeline/scripts/build_ocr_dataset.py \
  --annotations-csv data/processed/word_annotations.csv \
  --output-root data/custom_word_ocr_dataset \
  --image-path-column word_image_path \
  --label-column medicine_name \
  --approved-status reviewed \
  --seed 42
```

### 6. Augment OCR Training Split

```bash
python3 pipeline/scripts/augment_ocr_dataset.py \
  --input-root data/custom_word_ocr_dataset \
  --output-root data/custom_word_ocr_dataset_augmented \
  --augmentations-per-image 3 \
  --seed 42
```

This augments only `Training/training_words`. It copies validation and testing unchanged.

### 7. Final OCR Inference

After TrOCR training, run:

```bash
python3 pipeline/scripts/run_end_to_end.py \
  --input data/raw \
  --output-dir data/final_demo_trocr \
  --yolo-model models/region_yolo_best.pt \
  --target-class 0 \
  --ocr-backend trocr \
  --ocr-unit word \
  --trocr-model /path/to/fine_tuned/best_model \
  --line-padding 6 \
  --lexicon pipeline/config/drug_lexicon.txt
```

The final output files are:

- `predictions.csv`
- `predictions.json`
- `word_manifest.csv`
- `line_manifest.csv`
- crop folders for thesis screenshots

## Optional Experiments

### YOLO Line Detector

Keep this as an ablation only. Current line YOLO performance was low:

- validation mAP50-95 around `0.090`
- test mAP50-95 around `0.069`

The notebook still contains an optional line YOLO training cell, but do not pass `--line-yolo-model` in the final run unless you are reporting the comparison.

### Raw Augmentation for Region YOLO

Skip it for the current submission unless visual inspection shows failures. The existing region model is already good enough for the next stages.

## Annotation Policy

- Keep the test split clean and unaugmented.
- Double-annotate 10-15 percent of samples if possible.
- Mark rows as `reviewed` only after label and crop quality are acceptable.
- For OCR training, use medicine word crops with filled `medicine_name`.

## What to Send for Review

Send:

- line crops with context previews
- word crops for medicine-name labeling
- annotation CSV or notebook UI

Avoid using isolated auto-word crops as the only source of truth without context. The prescription line context is important for resolving ambiguous handwriting.
