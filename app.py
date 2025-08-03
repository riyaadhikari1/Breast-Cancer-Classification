# --- Imports ---
import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
import io
import time
import zipfile
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from fpdf import FPDF
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
import os

# --- Setup ---
st.set_page_config(page_title="Breast Cancer Classifier", layout="wide")

# --- Model Load ---


@st.cache_resource
def load_models():
    m1 = load_model("best_model_fold3.keras")
    m2 = load_model("best_model_fold4.keras")
    return m1, m2


model1, model2 = load_models()
CLASS_MAP = {0: "Benign", 1: "Malignant"}

# --- Session State Defaults ---
for key in ["results", "errors", "uploaded_files", "timings", "feedback"]:
    if key not in st.session_state:
        st.session_state[key] = []

# --- Preprocess & Grad-CAM ---


def load_and_preprocess_image(file):
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (256, 256))
    return np.expand_dims(preprocess_input(img_resized.astype(np.float32)), 0), img_rgb


def generate_gradcam(model, x, layer="conv5_block3_out"):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(layer).output, model.output])
    with tf.GradientTape() as tape:
        conv_out, pred = grad_model(x)
        loss = pred[:, 0]
    grads = tape.gradient(loss, conv_out)
    weights = tf.reduce_mean(grads, axis=(0, 1, 2))
    cam = tf.reduce_sum(tf.multiply(conv_out[0], weights), axis=-1)
    heatmap = tf.maximum(cam, 0) / (tf.reduce_max(cam) + 1e-8)
    return heatmap.numpy()


def overlay_gradcam(heatmap, img):
    heatmap = cv2.resize(np.uint8(255*heatmap), (img.shape[1], img.shape[0]))
    colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    return cv2.addWeighted(img, 0.6, colored, 0.4, 0)

# --- PDF Report ---


def generate_pdf_report(results):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", size=16)
    pdf.cell(0, 10, "Breast Cancer Classification Report", ln=True, align="C")
    pdf.ln(10)
    pdf.set_font("Arial", size=12)
    for r in results:
        pdf.cell(0, 8, f"Image: {r['Image']}", ln=True)
        pdf.cell(
            0, 8, f"Prediction: {r['Prediction']} | Confidence: {r['Confidence']}", ln=True)
        pdf.ln(2)
        if os.path.exists(r['GradCAM_Path']):
            try:
                pdf.image(r['GradCAM_Path'], w=120)
                pdf.ln(10)
            except Exception as e:
                pdf.cell(0, 8, f"(Could not add Grad-CAM image: {e})", ln=True)
    pdf_output_path = "report.pdf"
    pdf.output(pdf_output_path)
    with open(pdf_output_path, "rb") as f:
        pdf_bytes = f.read()
    return pdf_bytes


# --- Tabs ---
tabs = st.tabs(["Upload & Classify", "Summary",
               "Visualization", "Feedback", "About / Help"])

# --- Tab 1: Upload & Classify ---
with tabs[0]:
    st.header("Upload & Classify")
    st.markdown("---")

    uploaded = st.file_uploader(
        "Upload images or ZIP file",
        type=["png", "jpg", "jpeg", "zip", 'webp', 'tiff', 'tif', 'bmp'],
        accept_multiple_files=True,
        key="file_uploader"
    )

    if uploaded:
        st.session_state.uploaded_files.clear()
        for f in uploaded:
            if f.name.lower().endswith(".zip"):
                with zipfile.ZipFile(f) as z:
                    for fn in z.namelist():
                        if fn.lower().endswith((".png", "jpg", "jpeg")):
                            file_bytes = io.BytesIO(z.read(fn))
                            file_bytes.name = fn
                            st.session_state.uploaded_files.append(file_bytes)
            else:
                st.session_state.uploaded_files.append(f)

        st.subheader("Preview Uploaded Images")
        cols = st.columns(4)
        for i, file in enumerate(st.session_state.uploaded_files):
            with cols[i % 4]:
                st.image(file, caption=getattr(file, "name",
                         f"Image {i}"), use_container_width=True)

        if st.button("Classify Images"):
            st.session_state.results.clear()
            st.session_state.errors.clear()
            st.session_state.timings.clear()
            progress = st.progress(0)

            for idx, f in enumerate(st.session_state.uploaded_files):
                start = time.time()
                try:
                    x, img = load_and_preprocess_image(f)
                    p1 = model1.predict(x)[0][0]
                    p2 = model2.predict(x)[0][0]
                    chosen_model = model1 if abs(
                        p1-0.5) > abs(p2-0.5) else model2
                    prob = p1 if abs(p1-0.5) > abs(p2-0.5) else p2
                    label = CLASS_MAP[int(prob >= 0.5)]

                    heat = generate_gradcam(chosen_model, x)
                    over = overlay_gradcam(heat, img)
                    _, buf = cv2.imencode(
                        ".png", cv2.cvtColor(over, cv2.COLOR_RGB2BGR))
                    gv = buf.tobytes()
                    gradcam_path = f"gradcam_{idx}.png"
                    with open(gradcam_path, "wb") as f_img:
                        f_img.write(gv)

                    st.session_state.results.append({
                        "Image": getattr(f, "name", f"Image {idx}"),
                        "Prediction": label,
                        "Confidence": f"{prob:.2%}" if prob >= 0.5 else f"{(1-prob):.2%}",
                        "Confidence_Raw": prob,
                        "GradCAM_Path": gradcam_path,
                        "GradCAM_Bytes": gv
                    })
                except Exception as e:
                    st.session_state.errors.append(
                        f"{getattr(f, 'name', 'Image')} - {str(e)}")
                st.session_state.timings.append(time.time() - start)
                progress.progress(
                    (idx + 1) / len(st.session_state.uploaded_files))

            if st.session_state.results:
                df = pd.DataFrame(st.session_state.results)[
                    ["Image", "Prediction", "Confidence"]]
                st.dataframe(df)
                st.download_button(
                    "Download PDF Report",
                    generate_pdf_report(st.session_state.results),
                    file_name="breast_cancer_report.pdf",
                    mime="application/pdf"
                )

            if st.session_state.errors:
                st.markdown("###  Errors")
                for e in st.session_state.errors:
                    st.error(e)

# --- Tab 2: Summary ---
with tabs[1]:
    st.header("Summary")
    st.write(
        f"Total Uploaded Images: **{len(st.session_state.uploaded_files)}**")
    st.write(f"Predictions Made: **{len(st.session_state.results)}**")
    st.write(f"Errors Encountered: **{len(st.session_state.errors)}**")

# --- Tab 3: Visualization ---
with tabs[2]:
    st.header("Visualizations")
    if st.session_state.results:
        st.subheader("Grad-CAM Visualizations")
        cols = st.columns(3)
        for i, res in enumerate(st.session_state.results):
            with cols[i % 3]:
                st.image(res["GradCAM_Bytes"],
                         caption=res["Image"], use_container_width=True)
    else:
        st.info("No results available. Please upload and classify images first.")

# --- Tab 4: Feedback ---
with tabs[3]:
    st.header("Feedback")
    if st.session_state.results:
        for idx, res in enumerate(st.session_state.results):
            key = f"feedback_{idx}_{res['Image']}"
            corr = st.radio(
                f"Is the prediction for **{res['Image']}** correct?", ("Yes", "No"), key=key)
            found = False
            for fb in st.session_state.feedback:
                if fb["Image"] == res["Image"]:
                    fb["Correct"] = corr
                    found = True
                    break
            if not found:
                st.session_state.feedback.append(
                    {"Image": res["Image"], "Correct": corr})
    else:
        st.info("No classification results to provide feedback on.")

# --- Tab 5: About / Help ---
with tabs[4]:
    st.header("ℹAbout / Help")
    st.markdown("""
    ### Project Description
    This project provides a breast cancer histopathological image classifier that uses two deep learning models with Grad-CAM explainability. It assists researchers and practitioners in analyzing biopsy images by providing predicted classes (Benign or Malignant) along with visual explanations of model focus areas.

    ### How to Use
    1. Upload individual image files or a ZIP archive of images using the Upload tab.
    2. Click the "Classify Images" button to run predictions.
    3. View classification results and Grad-CAM visualizations immediately after processing.
    4. Download the comprehensive PDF report with results and visualizations.
    5. Provide feedback on prediction accuracy in the Feedback tab to improve the system.

    ### Ethical Disclaimers
    - This tool is for research and educational purposes only.
    - It is NOT intended for clinical diagnosis or treatment decisions.
    - Users must consult qualified medical professionals for any healthcare decisions.
    - The developers disclaim all liability for misuse or misinterpretation of results.
    """)

# --- Footer ---
st.markdown("---")
st.markdown("<center>Final Year Project</center>", unsafe_allow_html=True)
