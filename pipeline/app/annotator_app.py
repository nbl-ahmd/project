#!/usr/bin/env python3
"""
Doctor annotation app for line crops.

Run:
streamlit run pipeline/app/annotator_app.py -- \
  --manifest data/processed/line_manifest.csv \
  --annotations data/processed/doctor_annotations.csv \
  --annotator-id doctor_1
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Tuple

import pandas as pd
import streamlit as st


ANNOTATION_COLS = [
    "transcription",
    "medicine_name",
    "dosage",
    "frequency",
    "confidence",
    "annotator_id",
    "review_status",
    "notes",
    "updated_at",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Line crop annotation app")
    parser.add_argument("--manifest", type=Path, required=True, help="line_manifest.csv path")
    parser.add_argument("--annotations", type=Path, required=True, help="annotation csv path")
    parser.add_argument("--annotator-id", type=str, default="doctor_1")
    known, _ = parser.parse_known_args()
    return known


def ensure_annotation_frame(manifest_df: pd.DataFrame, existing_path: Path) -> pd.DataFrame:
    base_cols = [
        "line_id",
        "page_id",
        "line_image_path",
        "context_image_path",
        "region_image_path",
        "page_image_path",
    ]
    missing = set(base_cols) - set(manifest_df.columns)
    optional_missing = missing & {"region_image_path", "page_image_path"}
    for col in optional_missing:
        manifest_df[col] = ""
    missing = missing - optional_missing
    if missing:
        raise ValueError(f"Manifest missing columns: {sorted(missing)}")

    base = manifest_df[base_cols].drop_duplicates("line_id").copy()

    if existing_path.exists():
        ann = pd.read_csv(existing_path)
        if "line_id" not in ann.columns:
            ann = pd.DataFrame(columns=["line_id"] + ANNOTATION_COLS)
    else:
        ann = pd.DataFrame(columns=["line_id"] + ANNOTATION_COLS)

    for c in ANNOTATION_COLS:
        if c not in ann.columns:
            ann[c] = ""

    merged = base.merge(ann[["line_id"] + ANNOTATION_COLS], on="line_id", how="left")
    merged["transcription"] = merged["transcription"].fillna("")
    merged["medicine_name"] = merged["medicine_name"].fillna("")
    merged["dosage"] = merged["dosage"].fillna("")
    merged["frequency"] = merged["frequency"].fillna("")
    merged["confidence"] = merged["confidence"].fillna("medium")
    merged["annotator_id"] = merged["annotator_id"].fillna("")
    merged["review_status"] = merged["review_status"].fillna("pending")
    merged["notes"] = merged["notes"].fillna("")
    merged["updated_at"] = merged["updated_at"].fillna("")
    return merged


def save_annotations(df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df = df[
        [
            "line_id",
            "page_id",
            "line_image_path",
            "context_image_path",
            "region_image_path",
            "page_image_path",
        ]
        + ANNOTATION_COLS
    ].copy()
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    out_df.to_csv(tmp, index=False)
    tmp.replace(out_path)


def get_filtered_indices(df: pd.DataFrame, mode: str, mine_only: bool, annotator_id: str) -> list[int]:
    work = df
    if mine_only:
        work = work[(work["annotator_id"] == "") | (work["annotator_id"] == annotator_id)]

    if mode == "All":
        idxs = work.index.tolist()
    elif mode == "Unannotated":
        idxs = work[work["transcription"].astype(str).str.strip() == ""].index.tolist()
    elif mode == "Pending":
        idxs = work[work["review_status"] == "pending"].index.tolist()
    elif mode == "Reviewed":
        idxs = work[work["review_status"] == "reviewed"].index.tolist()
    else:
        idxs = work.index.tolist()

    return idxs


def resolve_image(path_str: str, manifest_parent: Path) -> Path:
    p = Path(path_str)
    if p.is_absolute():
        return p
    if p.exists():
        return p.resolve()
    return (manifest_parent / p).resolve()


def display_images(line_path: Path, context_path: Path) -> None:
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Line Crop")
        st.image(str(line_path), use_column_width=True)
    with c2:
        st.subheader("Context")
        st.image(str(context_path), use_column_width=True)


def app() -> None:
    args = parse_args()
    st.set_page_config(page_title="Prescription Annotation Tool", layout="wide")
    st.title("Prescription Line Annotation Tool")

    manifest_df = pd.read_csv(args.manifest)
    manifest_parent = args.manifest.parent
    merged = ensure_annotation_frame(manifest_df, args.annotations)

    st.sidebar.header("Controls")
    annotator_id = st.sidebar.text_input("Annotator ID", value=args.annotator_id)
    filter_mode = st.sidebar.selectbox("Filter", ["All", "Unannotated", "Pending", "Reviewed"], index=1)
    mine_only = st.sidebar.checkbox("Show only mine/unassigned", value=True)

    filtered_idxs = get_filtered_indices(merged, filter_mode, mine_only, annotator_id)
    if not filtered_idxs:
        st.info("No rows under current filter.")
        return

    done = int((merged["transcription"].astype(str).str.strip() != "").sum())
    total = len(merged)
    st.sidebar.write(f"Progress: {done}/{total} ({(100.0*done/total):.1f}%)")
    st.sidebar.progress(done / total if total else 0.0)

    if "cursor" not in st.session_state:
        st.session_state.cursor = 0
    st.session_state.cursor = max(0, min(st.session_state.cursor, len(filtered_idxs) - 1))

    nav_col1, nav_col2, nav_col3 = st.columns([1, 1, 3])
    with nav_col1:
        if st.button("Previous") and st.session_state.cursor > 0:
            st.session_state.cursor -= 1
    with nav_col2:
        if st.button("Next") and st.session_state.cursor < len(filtered_idxs) - 1:
            st.session_state.cursor += 1
    with nav_col3:
        st.write(f"Item {st.session_state.cursor + 1}/{len(filtered_idxs)}")

    idx = filtered_idxs[st.session_state.cursor]
    row = merged.loc[idx].copy()

    line_path = resolve_image(str(row["line_image_path"]), manifest_parent)
    context_path = resolve_image(str(row["context_image_path"]), manifest_parent)

    st.caption(f"line_id: {row['line_id']}  |  page_id: {row['page_id']}")
    display_images(line_path, context_path)

    with st.form("annotation_form", clear_on_submit=False):
        transcription = st.text_input("Transcription", value=str(row["transcription"]))
        medicine_name = st.text_input("Medicine Name", value=str(row["medicine_name"]))
        dosage = st.text_input("Dosage", value=str(row["dosage"]))
        frequency = st.text_input("Frequency", value=str(row["frequency"]))
        confidence = st.selectbox(
            "Confidence",
            ["low", "medium", "high"],
            index=["low", "medium", "high"].index(str(row["confidence"]) if str(row["confidence"]) in {"low", "medium", "high"} else "medium"),
        )
        review_status = st.selectbox(
            "Review Status",
            ["pending", "reviewed", "reject"],
            index=["pending", "reviewed", "reject"].index(
                str(row["review_status"]) if str(row["review_status"]) in {"pending", "reviewed", "reject"} else "pending"
            ),
        )
        notes = st.text_area("Notes", value=str(row["notes"]), height=80)

        save_col1, save_col2 = st.columns(2)
        save_clicked = save_col1.form_submit_button("Save")
        save_next_clicked = save_col2.form_submit_button("Save + Next")

    if save_clicked or save_next_clicked:
        merged.at[idx, "transcription"] = transcription.strip()
        merged.at[idx, "medicine_name"] = medicine_name.strip()
        merged.at[idx, "dosage"] = dosage.strip()
        merged.at[idx, "frequency"] = frequency.strip()
        merged.at[idx, "confidence"] = confidence
        merged.at[idx, "review_status"] = review_status
        merged.at[idx, "notes"] = notes.strip()
        merged.at[idx, "annotator_id"] = annotator_id.strip()
        merged.at[idx, "updated_at"] = datetime.utcnow().isoformat(timespec="seconds")
        save_annotations(merged, args.annotations)
        st.success(f"Saved {row['line_id']} -> {args.annotations}")

        if save_next_clicked and st.session_state.cursor < len(filtered_idxs) - 1:
            st.session_state.cursor += 1
            st.rerun()


if __name__ == "__main__":
    app()
