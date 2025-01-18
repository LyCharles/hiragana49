from fastapi import FastAPI, File, UploadFile
from typing import List
import torch
from PIL import Image, ImageOps
import io
import numpy as np
from model import load_model

app = FastAPI()

# Load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_model().to(device)
model.eval()


# Image preprocessing function
def preprocess_image(image: Image.Image) -> torch.Tensor:
    # Remove white borders and resize to 28x28 pixels
    image = ImageOps.autocontrast(image)
    image = image.resize((28, 28))

    # Convert image to tensor and normalize
    image_array = np.array(image) / 255.0  # Normalize pixel values to [0, 1]
    image_tensor = torch.tensor(image_array).unsqueeze(0).unsqueeze(0).float()
    # Apply the same normalization used for validation data
    image_tensor = (image_tensor - 0.5) / 0.5  # Equivalent to transforms.Normalize([0.5], [0.5])
    return image_tensor


# Asynchronous single image prediction function
# Using `async` allows the function to handle incoming requests without blocking
# Useful for handling multiple requests concurrently in a web server
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("L")
    image_tensor = preprocess_image(image).to(device)

    # Model prediction
    with torch.no_grad():
        output = model(image_tensor)
        _, predicted = torch.max(output, 1)
    return {"prediction": int(predicted.item())}


# Also asynchronous batch image prediction function
@app.post("/predict_batch")
async def predict_batch(files: List[UploadFile] = File(...)):
    predictions = []
    for file in files:
        image_bytes = await file.read()  # Read each image file asynchronously
        image = Image.open(io.BytesIO(image_bytes)).convert("L")  # Convert to grayscale
        image_tensor = preprocess_image(image).to(device)

        # Model prediction
        with torch.no_grad():
            output = model(image_tensor)
            _, predicted = torch.max(output, 1)
        predictions.append(int(predicted.item()))  # Append each prediction to the list
    return {"predictions": predictions}
