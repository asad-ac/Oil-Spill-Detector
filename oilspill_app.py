import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np

st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(to bottom, white, #0f4c81, black);
        color: white;
    }
    </style>
""", unsafe_allow_html=True)



CLASS_NAMES = ['No Oil Spill', 'Oil Spill']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ResNetTransferLearning(nn.Module):
    def __init__(self, num_classes=2):
        super(ResNetTransferLearning, self).__init__()
        self.resnet = models.resnet18(pretrained=False)
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.resnet(x)

def load_model(model_path):
    model = ResNetTransferLearning(num_classes=2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def predict(image, model):
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(image)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probs, 1)
    return CLASS_NAMES[predicted.item()], confidence.item()

st.set_page_config(page_title="Oil Spill Detector", layout="centered")
st.title("Group D: Oil Spill Detection Model")
st.write("Upload any image to detect the presence of an oil spill.")

uploaded_file = st.file_uploader("Choose an image file", type=['jpg', 'jpeg', 'png'])

if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_column_width=True)

    with st.spinner('Currently running analysis...'):
        model = load_model('resnet_oil_spill_model.pth')
        label, confidence = predict(image, model)

    st.success(f"Prediction: **{label}**")
    st.write(f"Confidence: {confidence * 100:.2f}%")
