import streamlit as st
import cv2
from ultralytics import YOLO
import tempfile
import os

# 1) Load model once
@st.cache_resource
def load_model():
    model = YOLO("yolov8n.pt")   # change to your weights
    return model

def run_inference_on_video(video_path, model):
    cap = cv2.VideoCapture(video_path)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Your existing YOLO inference
        results = model(frame)

        # Your existing drawing / counting logic here
        annotated_frame = results[0].plot()

        # Streamlit display (BGR → RGB)
        rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        yield rgb_frame

    cap.release()

def main():
    st.title("Traffic Surveillance System using YOLO")

    st.sidebar.header("Input")
    source_type = st.sidebar.radio("Source", ["Upload video", "Webcam"])

    model = load_model()

    if source_type == "Upload video":
        uploaded = st.file_uploader("Upload traffic video", type=["mp4", "avi", "mov"])
        if uploaded is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded.read())
            tfile.close()

            stframe = st.empty()
            for frame in run_inference_on_video(tfile.name, model):
                stframe.image(frame, channels="RGB")

            os.unlink(tfile.name)

    else:  # Webcam
        st.warning("Webcam mode may not work on Streamlit Cloud; best for local use.")
        run_webcam = st.checkbox("Start webcam")
        if run_webcam:
            cap = cv2.VideoCapture(0)
            stframe = st.empty()
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                results = model(frame)
                annotated = results[0].plot()
                rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                stframe.image(rgb, channels="RGB")
            cap.release()

if __name__ == "__main__":
    main()
