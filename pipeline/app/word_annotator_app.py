#!/usr/bin/env python3
"""Streamlit app for word-level prescription crop annotation."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st


ANNOTATION_COLS = [
    "word_text",
    "medicine_name",
    "is_medicine",
    "confidence",
    "annotator_id",
    "review_status",
    "notes",
    "updated_at",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Word crop annotation app")
    parser.add_argument("--manifest", type=Path, required=True, help="word_manifest.csv path")
    parser.add_argument("--annotations", type=Path, required=True, help="word annotation CSV path")
    parser.add_argument("--annotator-id", type=str, default="annotator_1")
    known, _ = parser.parse_known_args()
    return known


def resolve_image(path_str: str, manifest_parent: Path) -> Path:
    p = Path(path_str)
    if p.is_absolute():
        return p
    if p.exists():
        return p.resolve()
    return (manifest_parent / p).resolve()


def ensure_frame(manifest_df: pd.DataFrame, existing_path: Path) -> pd.DataFrame:
    base_cols = [
        "word_id",
        "line_id",
        "region_id",
        "page_id",
        "word_image_path",
        "line_image_path",
        "line_context_image_path",
    ]
    missing = set(base_cols) - set(manifest_df.columns)
    if missing:
        raise ValueError(f"Manifest missing columns: {sorted(missing)}")

    base = manifest_df[base_cols].drop_duplicates("word_id").copy()
    if existing_path.exists():
        ann = pd.read_csv(existing_path)
        if "word_id" not in ann.columns:
            ann = pd.DataFrame(columns=["word_id"] + ANNOTATION_COLS)
    else:
        ann = pd.DataFrame(columns=["word_id"] + ANNOTATION_COLS)

    for col in ANNOTATION_COLS:
        if col not in ann.columns:
            ann[col] = ""

    merged = base.merge(ann[["word_id"] + ANNOTATION_COLS], on="word_id", how="left")
    for col in ANNOTATION_COLS:
        merged[col] = merged[col].fillna("")
    merged["confidence"] = merged["confidence"].replace("", "medium")
    merged["review_status"] = merged["review_status"].replace("", "pending")
    return merged


def save_annotations(df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cols = [
        "word_id",
        "line_id",
        "region_id",
        "page_id",
        "word_image_path",
        "line_image_path",
        "line_context_image_path",
    ] + ANNOTATION_COLS
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    df[cols].to_csv(tmp, index=False)
    tmp.replace(out_path)


def filtered_indices(df: pd.DataFrame, mode: str, medicine_only: bool, mine_only: bool, annotator_id: str) -> list[int]:
    work = df
    if mine_only:
        work = work[(work["annotator_id"] == "") | (work["annotator_id"] == annotator_id)]
    if medicine_only:
        work = work[work["is_medicine"].astype(str).str.lower().isin(["1", "true", "yes", "y"])]
    if mode == "Unannotated":
        work = work[work["word_text"].astype(str).str.strip() == ""]
    elif mode == "Pending":
        work = work[work["review_status"] == "pending"]
    elif mode == "Reviewed":
        work = work[work["review_status"] == "reviewed"]
    elif mode == "Rejected":
        work = work[work["review_status"] == "reject"]
    return work.index.tolist()


def app() -> None:
    args = parse_args()
    st.set_page_config(page_title="Prescription Word Annotation", layout="wide")
    st.title("Prescription Word Annotation Tool")

    manifest_df = pd.read_csv(args.manifest)
    manifest_parent = args.manifest.parent
    merged = ensure_frame(manifest_df, args.annotations)

    st.sidebar.header("Controls")
    annotator_id = st.sidebar.text_input("Annotator ID", value=args.annotator_id)
    mode = st.sidebar.selectbox("Filter", ["All", "Unannotated", "Pending", "Reviewed", "Rejected"], index=1)
    mine_only = st.sidebar.checkbox("Show only mine/unassigned", value=True)
    medicine_only = st.sidebar.checkbox("Show medicine-marked only", value=False)

    idxs = filtered_indices(merged, mode, medicine_only, mine_only, annotator_id)
    if not idxs:
        st.info("No rows under current filter.")
        return

    done = int((merged["word_text"].astype(str).str.strip() != "").sum())
    total = len(merged)
    st.sidebar.write(f"Progress: {done}/{total} ({(100.0 * done / total):.1f}%)")
    st.sidebar.progress(done / total if total else 0.0)

    if "word_cursor" not in st.session_state:
        st.session_state.word_cursor = 0
    st.session_state.word_cursor = max(0, min(st.session_state.word_cursor, len(idxs) - 1))

    nav1, nav2, nav3 = st.columns([1, 1, 3])
    with nav1:
        if st.button("Previous") and st.session_state.word_cursor > 0:
            st.session_state.word_cursor -= 1
    with nav2:
        if st.button("Next") and st.session_state.word_cursor < len(idxs) - 1:
            st.session_state.word_cursor += 1
    with nav3:
        st.write(f"Item {st.session_state.word_cursor + 1}/{len(idxs)}")

    idx = idxs[st.session_state.word_cursor]
    row = merged.loc[idx].copy()
    word_path = resolve_image(str(row["word_image_path"]), manifest_parent)
    context_path = resolve_image(str(row["line_context_image_path"]), manifest_parent)
    line_path = resolve_image(str(row["line_image_path"]), manifest_parent)

    st.caption(f"word_id: {row['word_id']} | line_id: {row['line_id']} | page_id: {row['page_id']}")
    col1, col2, col3 = st.columns([1, 2, 2])
    with col1:
        st.subheader("Word")
        st.image(str(word_path), use_column_width=True)
    with col2:
        st.subheader("Line")
        st.image(str(line_path), use_column_width=True)
    with col3:
        st.subheader("Word Context")
        st.image(str(context_path), use_column_width=True)

    with st.form("word_annotation_form", clear_on_submit=False):
        word_text = st.text_input("Word Text", value=str(row["word_text"]))
        medicine_name = st.text_input("Medicine Name", value=str(row["medicine_name"]))
        is_medicine = st.selectbox(
            "Is Medicine?",
            ["", "yes", "no"],
            index=["", "yes", "no"].index(str(row["is_medicine"]) if str(row["is_medicine"]) in {"", "yes", "no"} else ""),
        )
        confidence = st.selectbox(
            "Confidence",
            ["low", "medium", "high"],
            index=["low", "medium", "high"].index(str(row["confidence"]) if str(row["confidence"]) in {"low", "medium", "high"} else "medium"),
        )
        review_status = st.selectbox(
            "Review Status",
            ["pending", "reviewed", "reject"],
            index=["pending", "reviewed", "reject"].index(str(row["review_status"]) if str(row["review_status"]) in {"pending", "reviewed", "reject"} else "pending"),
        )
        notes = st.text_area("Notes", value=str(row["notes"]), height=80)

        save_col, save_next_col = st.columns(2)
        save_clicked = save_col.form_submit_button("Save")
        save_next_clicked = save_next_col.form_submit_button("Save + Next")

    if save_clicked or save_next_clicked:
        merged.at[idx, "word_text"] = word_text.strip()
        merged.at[idx, "medicine_name"] = medicine_name.strip()
        merged.at[idx, "is_medicine"] = is_medicine
        merged.at[idx, "confidence"] = confidence
        merged.at[idx, "review_status"] = review_status
        merged.at[idx, "notes"] = notes.strip()
        merged.at[idx, "annotator_id"] = annotator_id.strip()
        merged.at[idx, "updated_at"] = datetime.utcnow().isoformat(timespec="seconds")
        save_annotations(merged, args.annotations)
        st.success(f"Saved {row['word_id']} -> {args.annotations}")

        if save_next_clicked and st.session_state.word_cursor < len(idxs) - 1:
            st.session_state.word_cursor += 1
            st.rerun()


if __name__ == "__main__":
    app()
