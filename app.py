import streamlit as st
import openai
import os
import requests
import io
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms, models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMG_SIZE = 224
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def get_model():
    model = models.resnet18(pretrained=False)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 1)
    return model

model_path = "cat_dog_classifier.pt"
model = get_model().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

openai.api_key = os.getenv("OPENAI_API_KEY", "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")

st.title("Image Generation with OpenAI API + Cat/Dog Classification")
st.write("Enter the image description you want to generate:")

prompt = st.text_input("Image Description:")

if st.button("Generate and Classify Image") and prompt:
    try:
        response = openai.Image.create(
            prompt=prompt,
            n=1,
            size="1024x1024"
        )
        image_url = response['data'][0]['url']
        st.image(image_url, caption="Generated Image", use_container_width=True)
        
        r = requests.get(image_url)
        img = Image.open(io.BytesIO(r.content)).convert("RGB")
        
        img_tensor = transform(img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(img_tensor)
            prediction = torch.sigmoid(output).item()
        
        result = "Cat" if prediction < 0.5 else "Dog"
        
        st.write(f"Classification result: **{result}** (Confidence: {prediction:.2f})")
        
    except Exception as e:
        st.error(f"Error: {e}")
