import streamlit as st
import torch
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import json
import requests
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

# Check if GPU is available
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

st.title("Object Detection with GPUNet by umar igan")
st.markdown("Code: [Github](https://github.com/UmarIgan/gpunetobjectdetect)")


# Model loading section
model_type = st.sidebar.selectbox("Select Model Type", ["GPUNet-0", "GPUNet-1", "GPUNet-2", "GPUNet-P0", "GPUNet-P1", "GPUNet-D1", "GPUNet-D2"])
precision = st.sidebar.selectbox("Select Precision", ["fp32", "fp16"])

values = ['<select>',3, 5, 10, 15, 20, 30]
default_ix = values.index(3)
num_of_results = st.sidebar.selectbox('Select Number of object to detect', values, index=default_ix)

# Load the model
gpunet = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_gpunet', pretrained=True, model_type=model_type, model_math=precision)
utils = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_convnets_processing_utils', force_reload=True)

gpunet.to(device)
gpunet.eval()

# Image upload and inference section
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image.", use_column_width=True)

    # Prepare input data for inference
    #batch = utils.prepare_input(image).to(device)
    convert_tensor = transforms.ToTensor()
    batch = convert_tensor(image).to(device).unsqueeze(0)

    if precision == "fp16":
        batch = batch.half()

    # Run inference
    with torch.no_grad():
        output = torch.nn.functional.softmax(gpunet(batch), dim=1)
    results = utils.pick_n_best(predictions=output, n=num_of_results)

    # Display results
    for i, result in enumerate(results):
        st.subheader(f"Sample {i}")
        for item in result:
            st.text(f"{item[0]}: {item[1]}")

