import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

# --- Configuration ---
NUM_CLASSES = 10 
MODEL_SAVE_PATH = "./resnet50_transfer_learned.pth"
IMG_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the CIFAR-10 class labels
CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer', 
    'dog', 'frog', 'horse', 'ship', 'truck'
]

# --- 1. Model Definition ---

@st.cache_resource # Cache the model to avoid reloading on every interaction
def setup_transfer_learning_model(num_classes: int) -> nn.Module:
    """
    Builds the ResNet-50 structure and loads the saved weights.
    """
    # Load the pre-trained ResNet-50 model structure
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

    # Freeze all parameters 
    for param in model.parameters():
        param.requires_grad = False
    
    # Replace the fully connected layer
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    # Load the fine-tuned parameters from the saved .pth file
    try:
        state_dict = torch.load(MODEL_SAVE_PATH, map_location=DEVICE, weights_only=False)
        model.load_state_dict(state_dict)
        model.eval() # Set to evaluation mode
        return model.to(DEVICE)
    except FileNotFoundError:
        st.error(f"Model file '{MODEL_SAVE_PATH}' not found. Please ensure your training script ran and saved the file.")
        return None
    except Exception as e:
        st.error(f"Error loading model weights: {e}")
        return None


# --- 2. Image Preprocessing ---

def preprocess_image(image: Image.Image) -> torch.Tensor:
    """
    Applies the same transformations used during ResNet-50 fine-tuning.
    """
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        # ImageNet normalization used for pre-trained models
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Apply transformation and add batch dimension (C, H, W -> 1, C, H, W)
    return transform(image).unsqueeze(0).to(DEVICE)


# --- 3. Prediction Function ---

def predict(model: nn.Module, input_tensor: torch.Tensor) -> str:
    """
    Runs the model prediction and returns the class label.
    """
    with torch.no_grad():
        outputs = model(input_tensor)
        # Get the index of the highest probability
        _, predicted_idx = torch.max(outputs.data, 1)
        
        # Get the corresponding class label
        prediction = CIFAR10_CLASSES[predicted_idx.item()]
        
        # Optional: Get confidence score (softmax)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        confidence = probabilities[predicted_idx.item()].item()
        
        return f"{prediction.capitalize()} (Confidence: {confidence:.2f})"


# --- 4. Streamlit App Layout ---

def main():
    st.set_page_config(page_title="ResNet-50 CIFAR-10 TL Classifier", layout="wide")
    
    st.title("🚀 ResNet-50 Transfer Learning Image Classifier")
    st.markdown("Upload an image to classify it into one of the 10 CIFAR-10 categories.")

    # Load model once
    model = setup_transfer_learning_model(NUM_CLASSES)
    
    if model is None:
        st.stop() # Stop execution if model loading failed

    # File Uploader
    uploaded_file = st.file_uploader(
        "Choose an image...", 
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file).convert('RGB')
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.image(image, caption='Uploaded Image', use_column_width=True)
            st.write("")
        
        with col2:
            st.subheader("Model Prediction")
            
            # Preprocess and predict
            with st.spinner('Analyzing image...'):
                input_tensor = preprocess_image(image)
                result = predict(model, input_tensor)
            
            st.success(f"**Predicted Class:** {result}")
            
            # Optional: Displaying the expected class list
            st.markdown("---")
            st.markdown(f"**Model Class Labels:** *{', '.join(CIFAR10_CLASSES)}*")

if __name__ == "__main__":
    main()
