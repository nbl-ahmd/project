# Custom Dataset Annotation Pipeline (Phase 4+)

This toolkit lets you go from `data/raw` prescription photos to an annotation-ready dataset for doctors, and then build an OCR training dataset.

## Notebook Workflows

If you want to run the cropper and annotator tools inside Colab/Jupyter, use:

- `phase4_full_prescription_annotation_tools.ipynb`
  - preprocesses full prescription images
  - prepares CVAT/Label Studio layout annotation files
  - includes an in-notebook manual handwritten-region cropper if CVAT is not used
  - crops handwritten regions
  - segments lines and words
  - provides an in-notebook word annotation UI
  - exports a BD-style custom word OCR dataset

- `phase4_trocr_word_level.ipynb`
  - trains/evaluates TrOCR on cropped word images
  - can be pointed to `data/custom_word_ocr_dataset`
  - includes the final full-prescription inference section

## Key Decision (Your Question)

`header/footer/handwriting` segmentation should be done **before** doctor annotation.

- Do layout segmentation annotation first (by technical annotators, not doctors).
- Then crop handwritten regions and line images.
- Send **line crops + page context** to doctors for text annotation.
- Create single-medicine crops only after verified annotations if still needed.

Do **not** start by sending isolated word crops to doctors. You lose context and introduce wrong auto-crops.

## Folder Structure Used

- Raw input: `data/raw`
- Processed outputs: `data/processed/...`
- Pipeline scripts: `pipeline/scripts`
- Doctor app: `pipeline/app/annotator_app.py`
- Final built OCR dataset: `data/custom_ocr_dataset`

## Install

```bash
python3 -m pip install -r pipeline/requirements.txt
```

For local TrOCR inference, install the optional OCR stack:

```bash
python3 -m pip install -r pipeline/requirements-ocr.txt
```

## Stage 1: Preprocess Raw Prescription Pages

```bash
python3 pipeline/scripts/preprocess_pages.py \
  --input-dir data/raw \
  --output-dir data/processed/pages \
  --manifest-out data/processed/page_manifest.csv
```

What it does:
- resize (optional), deskew, denoise, contrast enhancement
- saves clean page images and a page manifest

## Stage 2: Prepare for Layout Annotation (CVAT/Label Studio)

```bash
python3 pipeline/scripts/prepare_layout_annotation.py \
  --pages-dir data/processed/pages \
  --output-dir data/processed/layout_annotation_package \
  --copy-images
```

Then annotate in CVAT/Label Studio using classes:
- `header`
- `handwritten_region`
- `footer`

Export labels in YOLO format (`.txt` per image) and keep them in:
- `data/processed/layout_yolo_labels`

## Stage 3: Crop Handwritten Regions from YOLO Labels

```bash
python3 pipeline/scripts/crop_regions_from_yolo.py \
  --pages-dir data/processed/pages \
  --labels-dir data/processed/layout_yolo_labels \
  --class-map pipeline/config/layout_classes.txt \
  --target-label handwritten_region \
  --output-dir data/processed/regions \
  --manifest-out data/processed/region_manifest.csv
```

## Stage 4: Segment Line Crops from Handwritten Regions

```bash
python3 pipeline/scripts/segment_lines.py \
  --region-manifest data/processed/region_manifest.csv \
  --output-dir data/processed/line_crops \
  --manifest-out data/processed/line_manifest.csv
```

Outputs:
- line crop images
- context preview images (line highlighted in region)
- line manifest CSV

## Stage 5: Build Doctor Annotation Sheet

```bash
python3 pipeline/scripts/create_annotation_manifest.py \
  --line-manifest data/processed/line_manifest.csv \
  --output-csv data/processed/doctor_annotations.csv \
  --split-into-doctors 3 \
  --doctor-prefix doctor
```

This creates:
- `doctor_annotations.csv` (master sheet)
- `doctor_1_annotations.csv`, `doctor_2_annotations.csv`, ...

## Stage 6A: Doctor Line Annotation Tool (Streamlit)

```bash
streamlit run pipeline/app/annotator_app.py -- \
  --manifest data/processed/line_manifest.csv \
  --annotations data/processed/doctor_annotations.csv \
  --annotator-id doctor_1
```

Doctors will fill:
- `transcription`
- `medicine_name`
- `dosage`
- `frequency`
- `confidence`
- `review_status`
- `notes`

## Stage 6B: Segment and Annotate Word Crops

Use this when you want a word-level dataset similar to:

`data/doctors-handwritten-prescription-bd-dataset/Doctor’s Handwritten Prescription BD dataset`

First segment words from line crops:

```bash
python3 pipeline/scripts/segment_words.py \
  --line-manifest data/processed/line_manifest.csv \
  --output-dir data/processed/word_crops \
  --manifest-out data/processed/word_manifest.csv
```

Create the word annotation CSV:

```bash
python3 pipeline/scripts/create_word_annotation_manifest.py \
  --word-manifest data/processed/word_manifest.csv \
  --output-csv data/processed/word_annotations.csv
```

Run the word annotation app:

```bash
streamlit run pipeline/app/word_annotator_app.py -- \
  --manifest data/processed/word_manifest.csv \
  --annotations data/processed/word_annotations.csv \
  --annotator-id annotator_1
```

Annotators fill:
- `word_text`
- `medicine_name`
- `is_medicine`
- `confidence`
- `review_status`
- `notes`

For medicine-name OCR training, mark only correct medicine word crops as `review_status=reviewed` and fill `medicine_name`.

### Optional: Create a Shareable Package for a Specific Doctor

```bash
python3 pipeline/scripts/package_for_doctors.py \
  --annotations-csv data/processed/doctor_1_annotations.csv \
  --annotator-id doctor_1 \
  --output-dir data/processed/packages/doctor_1
```

## Stage 7: Build Final OCR Dataset from Approved Rows

```bash
python3 pipeline/scripts/build_ocr_dataset.py \
  --annotations-csv data/processed/doctor_annotations.csv \
  --output-root data/custom_ocr_dataset \
  --label-column medicine_name \
  --approved-status reviewed \
  --seed 42
```

For word-level medicine-name OCR, use:

```bash
python3 pipeline/scripts/build_ocr_dataset.py \
  --annotations-csv data/processed/word_annotations.csv \
  --output-root data/custom_word_ocr_dataset \
  --image-path-column word_image_path \
  --label-column medicine_name \
  --approved-status reviewed \
  --seed 42
```

Generated structure:
- `data/custom_ocr_dataset/Training/training_words/*.png`
- `data/custom_ocr_dataset/Validation/validation_words/*.png`
- `data/custom_ocr_dataset/Testing/testing_words/*.png`
- split CSV labels

## Suggested Annotation Policy

- 10–15% samples double-annotated by 2 doctors.
- Resolve conflicts before marking `review_status=reviewed`.
- Keep one master CSV in Drive with versioned backups.

## What to Send Doctors

Send:
- line crops
- context previews
- Streamlit access (or per-doctor CSV package)

Do not send:
- full raw pages for routine transcription
- isolated auto-word crops as primary annotation unit

## Final End-to-End Demo Runner

For the final presentation, use the single-command runner after installing dependencies:

```bash
python3 pipeline/scripts/run_end_to_end.py \
  --input data/raw/1.jpg \
  --output-dir data/final_demo \
  --ocr-backend trocr \
  --ocr-unit word \
  --trocr-model /path/to/fine_tuned/best_model
```

If the fine-tuned TrOCR checkpoint is not available on the current machine, smoke-test the full non-OCR pipeline:

```bash
python3 pipeline/scripts/run_end_to_end.py \
  --input data/raw/1.jpg \
  --output-dir data/final_demo \
  --ocr-unit word \
  --ocr-backend none
```

The runner outputs:
- `page_manifest.csv`
- `region_manifest.csv`
- `line_manifest.csv`
- `word_manifest.csv`
- `predictions.csv`
- `predictions.json`
- cropped region and line images for presentation screenshots

It uses YOLO labels if `--labels-dir data/processed/layout_yolo_labels` is provided. Without labels, it uses a heuristic handwritten-region proposal so the rest of the pipeline can still be demonstrated.

To test only the post-OCR validation stage:

```bash
python3 pipeline/scripts/validate_prescription_text.py \
  --text "Napa 500 mg 1-0-1"
```
