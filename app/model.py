import torch
from config import MODEL_PATH
from SpinalVGG import SpinalVGG

def load_model():
    # Load the trained model onto the appropriate device (GPU if available, else CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SpinalVGG(num_classes=49).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))  # Load model weights
    model.eval()  # Set the model to evaluation mode
    return model

def predict_image(model, image_tensor):
    # Predict the class of a single image tensor
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_tensor = image_tensor.to(device)  # Move the tensor to the appropriate device
    with torch.no_grad():
        output = model(image_tensor)  # Perform prediction
        _, predicted = torch.max(output, 1)  # Get the predicted class
    return predicted.item()  # Return the class as an integer
