#pip install streamlit-drawable-canvas
#pip install opencv-python

import streamlit as st
import numpy as np
import cv2
from streamlit_drawable_canvas import st_canvas
from tensorflow.keras.models import load_model

# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="Handwritten Digit Recognition",
    page_icon="✍️",
    layout="centered"
)

# ------------------ LOAD MODEL ------------------
model = load_model("digit_recognization_model.keras")

# ------------------ CUSTOM CSS ------------------
st.markdown("""
<style>

.stApp{
    background-color:#F8FAFC;
}

.main-title{
    text-align:center;
    color:#2563EB;
    font-size:42px;
    font-weight:bold;
}

.sub-title{
    text-align:center;
    color:#475569;
    font-size:18px;
    margin-bottom:25px;
}

.card{
    background:white;
    padding:25px;
    border-radius:18px;
    box-shadow:0px 5px 18px rgba(0,0,0,0.10);
    display:flex;
    justify-content:center;
}

.stButton>button{
    width:100%;
    background:#2563EB;
    color:white;
    border:none;
    border-radius:10px;
    height:50px;
    font-size:20px;
    font-weight:bold;
}

.stButton>button:hover{
    background:#1D4ED8;
    color:white;
}

.result-box{
    background:#ECFDF5;
    padding:15px;
    border-radius:10px;
    border-left:6px solid #10B981;
    text-align:center;
    color:#065F46;
    font-size:26px;
    font-weight:bold;
    margin-top:20px;
}

.footer{
    text-align:center;
    color:gray;
    margin-top:40px;
    font-size:15px;
}

</style>
""", unsafe_allow_html=True)

# ------------------ TITLE ------------------
st.markdown(
    '<div class="main-title">✍️ Handwritten Digit Recognition</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-title">Draw a digit (0–9) inside the canvas and click <b>Predict Digit</b>.</div>',
    unsafe_allow_html=True
)

# ------------------ DRAWING CANVAS ------------------
st.markdown('<div class="card">', unsafe_allow_html=True)

canvas_result = st_canvas(
    fill_color="#00000000",
    stroke_width=12,
    stroke_color="#FFFFFF",
    background_color="#000000",
    width=280,
    height=280,
    drawing_mode="freedraw",
    key="canvas"
)

st.markdown("</div>", unsafe_allow_html=True)

st.write("")

# ------------------ PREDICT ------------------
if st.button("🔍 Predict Digit"):

    if canvas_result.image_data is None:
        st.warning("Please draw a digit first.")
    else:

        img = canvas_result.image_data.astype(np.uint8)

        # Convert RGBA to Grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)

        # Resize to MNIST size
        gray = cv2.resize(gray, (28, 28))

        # Normalize
        gray = gray / 255.0

        # Reshape for ANN model
        gray = gray.reshape(1, 784)

        # Prediction
        prediction = model.predict(gray)

        digit = np.argmax(prediction)
        confidence = np.max(prediction) * 100

        st.markdown(
            f"""
            <div class="result-box">
            Predicted Digit : {digit}
            </div>
            """,
            unsafe_allow_html=True
        )

        st.success(f"Prediction Confidence: {confidence:.2f}%")

        with st.expander("Processed Image"):
            st.image(gray.reshape(28, 28), width=150)

# ------------------ FOOTER ------------------
st.markdown("""
<div class="footer">
Built with ❤️ using <b>Streamlit</b>, <b>TensorFlow</b> & <b>OpenCV</b>
</div>
""", unsafe_allow_html=True)