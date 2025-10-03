# app.py
import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from model import Disease_Classifier

# --------------------------
# Load model + class labels
# --------------------------
@st.cache_resource
def load_model():
    checkpoint = torch.load("Disease_Model.pth", map_location=torch.device("cpu"))
    classes = checkpoint["classes"]

    model = Disease_Classifier(label=len(classes), freeze_backbone=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, classes

model, class_names = load_model()

# --------------------------
# Image preprocessing
# --------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def predict(image: Image.Image):
    image = transform(image).unsqueeze(0)  # add batch dimension
    with torch.no_grad():
        outputs = model(image)
        _, preds = torch.max(outputs, 1)
    return class_names[preds.item()]

# --------------------------
# Streamlit UI
# --------------------------
st.set_page_config(page_title="Crop Disease Detector", layout="wide")
st.title("🌱 Kisan Care – Crop Disease Detector")
st.write("Upload a crop leaf image to identify the disease.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Leaf Image", use_column_width=True)

    if st.button("🔍 Predict Disease"):
        with st.spinner("Analyzing..."):
            result = predict(image)
        st.success(f"✅ Prediction: **{result}**")
