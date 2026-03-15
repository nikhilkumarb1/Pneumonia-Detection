<| HEAD
"""
Pneumonia Detection -- Streamlit Frontend
Sends chest X-ray images to the FastAPI backend and visualizes results.
"""

import streamlit as st
import requests
from PIL import Image
import io
import plotly.graph_objects as go


# ---------------- API CONFIG ----------------
API_BASE = "http://localhost:8000"

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="PneumoScan - AI Chest X-ray Analysis",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
    .main-title {
        font-size: 40px;
        font-weight: bold;
        color: #1f77b4;
    }
    .result-box {
        padding:20px;
        border-radius:10px;
        background-color:#f2f2f2;
        margin-top:10px;
    }
</style>
""", unsafe_allow_html=True)


# ---------------- SIDEBAR ----------------
st.sidebar.title("🧠 PneumoScan AI")
st.sidebar.markdown("""
This AI system analyzes **Chest X-ray images** to detect possible **Pneumonia**.

Upload an X-ray image and the AI model will predict:

- Pneumonia
- Normal

⚠️ **Disclaimer**  
This tool is for educational purposes only and should not replace professional medical advice.
""")


# ---------------- HEADER ----------------
st.markdown("<p class='main-title'>🫁 PneumoScan - Pneumonia Detection</p>", unsafe_allow_html=True)

st.write("Upload a **Chest X-ray image** and let the AI detect signs of pneumonia.")

st.divider()


# ---------------- IMAGE UPLOAD ----------------
uploaded_file = st.file_uploader(
    "Upload Chest X-ray Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:

    col1, col2 = st.columns(2)

    with col1:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded X-ray", use_container_width=True)

    with col2:

        if st.button("🔍 Analyze Image"):

            with st.spinner("Analyzing X-ray with AI model..."):

                try:

                    files = {
                        "file": (
                            uploaded_file.name,
                            uploaded_file.getvalue(),
                            uploaded_file.type
                        )
                    }

                    response = requests.post(
                        f"{API_BASE}/predict",
                        files=files
                    )

                    if response.status_code != 200:
                        st.error(f"API Error: {response.text}")
                        st.stop()

                    data = response.json()

                except Exception as e:
                    st.error(f"Connection error: {e}")
                    st.stop()

            # ---------------- RESULTS ----------------
            st.success("Analysis Complete")

            st.markdown("### 🧾 Prediction Result")

            st.markdown(
                f"""
                **Label:** {data['label']}  
                **Confidence:** {round(data['confidence']*100,2)}%  
                **Risk Level:** {data['risk_level']}  
                **Inference Time:** {data['inference_time_ms']} ms  
                """
            )

            st.markdown("### 🩺 AI Verdict")
            st.info(data["verdict"])

            # ---------------- PROBABILITY CHART ----------------
            labels = []
            scores = []

            for item in data["all_scores"]:
                labels.append(item["label"])
                scores.append(item["score"])

            fig = go.Figure(
                data=[
                    go.Bar(
                        x=labels,
                        y=scores
                    )
                ]
            )

            fig.update_layout(
                title="Prediction Probabilities",
                xaxis_title="Class",
                yaxis_title="Probability",
                yaxis=dict(range=[0,1])
            )

            st.plotly_chart(fig, use_container_width=True)


# ---------------- HEALTH CHECK ----------------
st.sidebar.markdown("---")

if st.sidebar.button("Check API Status"):

    try:
        r = requests.get(f"{API_BASE}/health")

        if r.status_code == 200:
            data = r.json()

            if data["model_loaded"]:
                st.sidebar.success("Model Loaded")
            else:
                st.sidebar.warning("Model Not Loaded")

        else:
            st.sidebar.error("API not responding")

    except:

"""
Pneumonia Detection -- Streamlit Frontend
Sends chest X-ray images to the FastAPI backend and visualizes results.
"""

import streamlit as st
import requests
from PIL import Image
import io
import plotly.graph_objects as go

# ---------------- API CONFIG ----------------
API_BASE = "http://localhost:8000"

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="PneumoScan - AI Chest X-ray Analysis",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
    .main-title {
        font-size: 40px;
        font-weight: bold;
        color: #1f77b4;
    }
    .result-box {
        padding:20px;
        border-radius:10px;
        background-color:#f2f2f2;
        margin-top:10px;
    }
</style>
""", unsafe_allow_html=True)


# ---------------- SIDEBAR ----------------
st.sidebar.title("🧠 PneumoScan AI")
st.sidebar.markdown("""
This AI system analyzes **Chest X-ray images** to detect possible **Pneumonia**.

Upload an X-ray image and the AI model will predict:

- Pneumonia
- Normal

⚠️ **Disclaimer**  
This tool is for educational purposes only and should not replace professional medical advice.
""")


# ---------------- HEADER ----------------
st.markdown("<p class='main-title'>🫁 PneumoScan - Pneumonia Detection</p>", unsafe_allow_html=True)

st.write("Upload a **Chest X-ray image** and let the AI detect signs of pneumonia.")

st.divider()


# ---------------- IMAGE UPLOAD ----------------
uploaded_file = st.file_uploader(
    "Upload Chest X-ray Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:

    col1, col2 = st.columns(2)

    with col1:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded X-ray", use_container_width=True)

    with col2:

        if st.button("🔍 Analyze Image"):

            with st.spinner("Analyzing X-ray with AI model..."):

                try:

                    files = {
                        "file": (
                            uploaded_file.name,
                            uploaded_file.getvalue(),
                            uploaded_file.type
                        )
                    }

                    response = requests.post(
                        f"{API_BASE}/predict",
                        files=files
                    )

                    if response.status_code != 200:
                        st.error(f"API Error: {response.text}")
                        st.stop()

                    data = response.json()

                except Exception as e:
                    st.error(f"Connection error: {e}")
                    st.stop()

            # ---------------- RESULTS ----------------
            st.success("Analysis Complete")

            st.markdown("### 🧾 Prediction Result")

            st.markdown(
                f"""
                **Label:** {data['label']}  
                **Confidence:** {round(data['confidence']*100,2)}%  
                **Risk Level:** {data['risk_level']}  
                **Inference Time:** {data['inference_time_ms']} ms  
                """
            )

            st.markdown("### 🩺 AI Verdict")
            st.info(data["verdict"])

            # ---------------- PROBABILITY CHART ----------------
            labels = []
            scores = []

            for item in data["all_scores"]:
                labels.append(item["label"])
                scores.append(item["score"])

            fig = go.Figure(
                data=[
                    go.Bar(
                        x=labels,
                        y=scores
                    )
                ]
            )

            fig.update_layout(
                title="Prediction Probabilities",
                xaxis_title="Class",
                yaxis_title="Probability",
                yaxis=dict(range=[0,1])
            )

            st.plotly_chart(fig, use_container_width=True)


# ---------------- HEALTH CHECK ----------------
st.sidebar.markdown("---")

if st.sidebar.button("Check API Status"):

    try:
        r = requests.get(f"{API_BASE}/health")

        if r.status_code == 200:
            data = r.json()

            if data["model_loaded"]:
                st.sidebar.success("Model Loaded")
            else:
                st.sidebar.warning("Model Not Loaded")

        else:
            st.sidebar.error("API not responding")

    except:

        st.sidebar.error("Backend not running")