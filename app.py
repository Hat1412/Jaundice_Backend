import streamlit as st
import torch
from torchvision import transforms, models
from PIL import Image, PngImagePlugin
import io

st.set_page_config(page_title="Nethvera")
PngImagePlugin.MAX_TEXT_CHUNK = 10485760  # Increase max text chunk size for PNG images

# Load model
@st.cache_resource
def load_model():
    model = models.efficientnet_b0(weights=None)
    num_features = model.classifier[1].in_features
    model.classifier = torch.nn.Sequential(
        torch.nn.Linear(num_features, len(classes)),  # Update for n classes
        torch.nn.Softmax(dim=1)
    )
    model.load_state_dict(torch.load("disease_model_with_scheduler.pth", map_location=torch.device("cpu")))
    model.eval()
    return model

# Prediction function
def predict_image(model, image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(image)
        _, preds = torch.max(outputs, 1)
    return classes[preds[0]]

# Classes
classes = ["Bells Palsy", "Jaundice", "Negative", "Pink Eye", "SLE"]

# Custom styles for hospital theme, font, and buttons
st.markdown(
    """
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css" rel="stylesheet">
    <style>
    body {
        font-family: 'Aptos', sans-serif;
    }
    .stMain {
        background: #FFFFF0;
    }
    .stMarkdown {
        color: #0047AB;
        text-align: center;
        font-size: 2.5rem;
    }
    .contact-icons {
        display: flex;
        justify-content: center;
        gap: 30px;
        margin-top: 20px;
    }
    .contact-icons a {
        font-size: 3rem;
        color: #123b77;
        text-decoration: none;
        transition: color 0.3s ease;
    }
    .contact-icons a:hover {
        color: #6fb4ff;
        font-size: 3.5rem;
    }
    .footer {
        text-align: center;
        text-decoration: bold;
        margin-top: 50px;
        font-size: 0.9rem;
        background-color: #87CEEB;
        color: #000;
    }
    p {
    text-align: center;}
    [datatestid = "stFileUploadDropZone"]{
    background-color: #ffffff;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Home Section
st.markdown('<h1 class="main-title">Welcome to Nethvera</h1>', unsafe_allow_html=True)
st.markdown("""<p style="text-align: left; font-size: 1.2rem;">
Nethvera is a revolutionary AI-powered diagnostic tool designed to detect diseases through non-invasive methods. 

1. **Upload or capture an image**: Provide a clear image of the affected area.
2. **AI Analysis**: Our model analyzes the image and predicts the possible condition.
3. **Receive Results**: Get detailed predictions instantly, helping you take the next steps.
</p>""", unsafe_allow_html=True)
st.markdown('<h1 class="main-title" id="diagnostic">Nethvera Diagnostic Assistant</h1>', unsafe_allow_html=True)
st.markdown("<p>Upload an image or take a photo using your camera to detect potential diseases. </p>", unsafe_allow_html=True)

# Load model
model = load_model()

# Camera input
camera_image = st.camera_input("Take a photo using your camera")
uploaded_file = st.file_uploader("Upload an image file", type=["jpg", "jpeg", "png"])

# Handle predictions
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    with st.spinner("Analyzing..."):
        result = predict_image(model, image)
    st.success(f"Prediction: {result}")

elif camera_image is not None:
    image = Image.open(io.BytesIO(camera_image.getvalue()))
    st.image(image, caption="Captured Image", use_container_width=True)

    with st.spinner("Analyzing..."):
        result = predict_image(model, image)
    st.success(f"Prediction: {result}")
