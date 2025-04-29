import streamlit as st
from inference_sdk import InferenceHTTPClient
from PIL import Image


CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="your_api_key" #Use your ROBOFLOW API KEY 
)


st.set_page_config(page_title="Animal Classifier 🐾", page_icon="🐶", layout="wide")


st.markdown("""
    <style>
        /* Background Gradient */
        .stApp {
            background: linear-gradient(to right, #0f2027, #203a43, #2c5364);
            color: white;
        }
        
        /* Title Styling */
        h1, h2, h3, h4, h5 {
            text-align: center;
            color: white;
        }
        
        /* Uploaded Image Styling */
        .uploaded-img {
            border-radius: 15px;
            box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.2);
        }
        
        /* Prediction Box */
        .prediction-box {
            background-color: #ffffff;
            color: #333;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.1);
            text-align: center;
        }
        
        /* Progress Bar */
        .stProgress > div > div > div > div {
            background-color: #ffcc00 !important;
        }
        
    </style>
""", unsafe_allow_html=True)


st.markdown("<h1>🐾 Animal Image Classifier</h1>", unsafe_allow_html=True)
st.markdown("<h3>Upload an image and let the AI classify it!</h3>", unsafe_allow_html=True)


uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"], label_visibility="collapsed")

if uploaded_file is not None:

    col1, col2 = st.columns([1, 1])

    with col1:
        image = Image.open(uploaded_file)
        st.image(image, caption="📸 Uploaded Image", use_container_width=True, output_format="JPEG")

    with col2:
        with st.spinner("🔍 Classifying..."):
            image_bytes = uploaded_file.read()
            response = CLIENT.infer(image, model_id="animal-classification-7xuap/1")

        # Extract Prediction
        if 'predictions' in response and response['predictions']:
            prediction = response['predictions'][0]
            class_name = prediction['class']
            confidence = prediction['confidence']


            st.markdown(
                f"""
                <div class='prediction-box'>
                    <h2>🧠 Prediction: <span style="color: #007BFF;">{class_name}</span></h2>
                    <h4>Confidence Score:</h4>
                    <progress value="{confidence}" max="1" style="width: 100%; height: 20px;"></progress>
                    <p style="font-size: 18px; font-weight: bold;">{confidence:.2%}</p>
                </div>
                """,
                unsafe_allow_html=True
            )

        else:
            st.error("⚠️ No predictions found. Try a different image.")


st.markdown("<br><h5 style='text-align: center; color: white;'>Powered by Roboflow & Streamlit 🚀</h5>", unsafe_allow_html=True)
