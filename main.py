import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
from PIL import Image
import requests
from io import BytesIO

def process_single_image_and_get_result(model_name, url):
    single_json = requests.get(url).json()
    prod = single_json["products"]['17387']
    image_url = prod['images'][0]
    prompt = f"""Generate a compelling product description for an e-commerce website. The product is {prod['product']},
    a creation by the brand {prod['brand']}. Focus on its key features, benefits, and how it stands out in the market.
    Highlight the unique ingredients and their benefits for the skin or hair. Emphasize the handcrafted nature and
    the quality of ingredients, appealing to customers looking for luxury, natural skincare products.
    Use persuasive and attractive language to entice potential buyers, aiming to convert interest into purchases.
    Include a call to action that encourages quick buying decisions."""

    # Fetch and open the image from URL
    response = requests.get(image_url)
    image = Image.open(BytesIO(response.content))

    # Encode the image and get result based on model
    if model_name == "moon_dream":
        model_id = "vikhyatk/moondream2"
        revision = "2024-04-02"
        model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, revision=revision)
        tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
        result = "Model 'moon_dream' not fully implemented for image encoding."  # Placeholder as the model API is not standard
    elif model_name == "MiniCPM":
        model = AutoModel.from_pretrained('openbmb/MiniCPM-V-2')
        tokenizer = AutoTokenizer.from_pretrained('openbmb/MiniCPM-V-2')
        model.eval()
        result = "Model 'MiniCPM' not fully implemented for image encoding."  # Placeholder as the model API is not standard
    else:
        raise ValueError("Invalid model name. Please choose 'moon_dream' or 'MiniCPM'.")
    return result

# Streamlit interface
st.title("Product Description Generator")
model_name = st.selectbox("Choose a Model", ["moon_dream", "MiniCPM"])
url = st.text_input("Enter the API URL", "https://api.example.com/data")
if st.button("Generate Description"):
    if url:
        try:
            content = process_single_image_and_get_result(model_name, url)
            st.write(content)
        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.error("Please provide a valid URL.")
