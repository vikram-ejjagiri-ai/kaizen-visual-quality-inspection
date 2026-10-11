"""Operator review screen (human-in-the-loop).

Run:  streamlit run app/review_app.py     (after python -m src.pipeline)

In a real line, the camera sends every part to the model. Parts the model
rejects go to this screen: the operator sees WHERE the model found something,
confirms the defect or overrules the model. Every decision is logged, so the
team can measure how often the model is overruled and retrain it later.
"""
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from src.config import CATEGORY, IMAGE_SIZE, MODELS_DIR, REPORTS_DIR
from src.data import to_tensor_batch
from src.patchcore import PatchCore
from src.plotting import overlay

LOG_PATH = REPORTS_DIR / "review_log.csv"

st.set_page_config(page_title="Kaizen Visual Inspection", layout="wide")


@st.cache_resource
def load_model():
    return PatchCore.load(MODELS_DIR / f"patchcore_{CATEGORY}.pt")


@st.cache_data
def load_results():
    results = json.loads((REPORTS_DIR / "results.json").read_text())
    scores = pd.read_csv(REPORTS_DIR / "test_scores.csv")
    return results, scores


def inspect(model, image):
    maps, scores = model.predict_batch(to_tensor_batch([image]))
    return maps[0], float(scores[0])


def log_decision(part, score, model_decision, operator_decision, note):
    row = pd.DataFrame([{"time": datetime.now().isoformat(timespec="seconds"), "part": part,
                         "score": round(score, 3), "model_decision": model_decision,
                         "operator_decision": operator_decision, "note": note}])
    row.to_csv(LOG_PATH, mode="a", header=not LOG_PATH.exists(), index=False)


if not (MODELS_DIR / f"patchcore_{CATEGORY}.pt").exists():
    st.error("No trained model found. Run `python -m src.pipeline` first.")
    st.stop()

model = load_model()
results, scores = load_results()
threshold = results["threshold"]
vmin = float(scores["score"].quantile(0.01))
vmax = float(scores["score"].max())

st.title("Visual Quality Inspection - Operator Review")
st.caption(f"Product: **{CATEGORY}** | decision threshold **{threshold:.2f}** "
           f"(chosen by cost: missed defect = {results['costs']['missed_defect']}, "
           f"false reject = {results['costs']['false_reject']})")

# ---- Choose a part ---------------------------------------------------------
with st.sidebar:
    st.header("Parts to review")
    source = st.radio("Source", ["Review queue (rejected parts)", "All test parts", "Upload image"])
    if source == "Upload image":
        upload = st.file_uploader("Image of a part", type=["png", "jpg", "jpeg", "bmp"])
        part_name, image = (upload.name, Image.open(upload).convert("RGB")) if upload else (None, None)
    else:
        pool = scores if source == "All test parts" else scores[scores["decision"] == "reject"]
        pool = pool.sort_values("score", ascending=False)
        st.write(f"{len(pool)} parts")
        part_name = st.selectbox("Part", pool["path"].tolist(),
                                 format_func=lambda p: Path(p).parent.name + " / " + Path(p).name)
        image = Image.open(ROOT / part_name).convert("RGB") if part_name else None

if image is None:
    st.info("Choose or upload a part in the sidebar.")
    st.stop()

# ---- Inspect ---------------------------------------------------------------
heatmap, score = inspect(model, image)
model_decision = "REJECT" if score >= threshold else "PASS"
shown = np.array(image.resize((IMAGE_SIZE, IMAGE_SIZE)))

c1, c2, c3 = st.columns([1, 1, 0.9])
c1.subheader("Camera image")
c1.image(shown, width="stretch")
c2.subheader("Where the AI looks")
c2.image(overlay(shown, heatmap, vmin, vmax), width="stretch", clamp=True)

with c3:
    st.subheader("AI decision")
    if model_decision == "REJECT":
        st.error(f"REJECT - anomaly score {score:.2f} (threshold {threshold:.2f})")
    else:
        st.success(f"PASS - anomaly score {score:.2f} (threshold {threshold:.2f})")
    st.progress(min(score / (2 * threshold), 1.0), text="Score relative to threshold")
    if source != "Upload image":
        st.caption(f"Ground truth in dataset: **{Path(part_name).parent.name}**")

    st.divider()
    st.subheader("Operator decision")
    note = st.text_input("Note (optional)", placeholder="e.g. scratch on outer ring")
    b1, b2 = st.columns(2)
    if b1.button("Confirm defect", type="primary", width="stretch"):
        log_decision(part_name, score, model_decision, "defect", note)
        st.toast("Logged: defect confirmed")
    if b2.button("Part is OK", width="stretch"):
        log_decision(part_name, score, model_decision, "ok", note)
        st.toast("Logged: part released")

# ---- Review log ------------------------------------------------------------
if LOG_PATH.exists():
    st.divider()
    log = pd.read_csv(LOG_PATH)
    overruled = (log["model_decision"].str.lower().map({"reject": "defect", "pass": "ok"})
                 != log["operator_decision"]).mean()
    st.subheader("Review log")
    st.caption(f"{len(log)} decisions logged | operator overruled the AI in {overruled:.0%} of cases")
    st.dataframe(log.tail(20).iloc[::-1], width="stretch", hide_index=True)
