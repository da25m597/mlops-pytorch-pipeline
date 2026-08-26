"""FastAPI serving app for the Fashion-MNIST classifier."""
import io
import os

import torch
import torch.nn.functional as F
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image
from torchvision import transforms

from model import get_model

CHECKPOINT_PATH = os.environ.get("CHECKPOINT_PATH", "/app/checkpoints/FashionMnistClassifier.pt")

CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]

app = FastAPI(title="Fashion-MNIST Classifier")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = None
preprocess = None


def load_model():
    global model, preprocess

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)

    loaded_model = get_model(
        architecture=checkpoint["architecture"],
        num_classes=checkpoint["num_classes"],
    )
    loaded_model.load_state_dict(checkpoint["model_state_dict"])
    loaded_model.to(device)
    loaded_model.eval()

    preprocess = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize(mean=checkpoint["mean"], std=checkpoint["std"]),
    ])

    model = loaded_model


try:
    load_model()
except Exception as exc:
    print(f"Failed to load model from {CHECKPOINT_PATH}: {exc}")


@app.get("/health")
def health():
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "healthy"}


@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        contents = await image.read()
        img = Image.open(io.BytesIO(contents))
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    input_tensor = preprocess(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(input_tensor)
        probabilities = F.softmax(logits, dim=1).squeeze(0).tolist()

    predicted_index = int(torch.argmax(logits, dim=1).item())

    return {
        "predicted_class": CLASS_NAMES[predicted_index],
        "predicted_index": predicted_index,
        "probabilities": {
            CLASS_NAMES[i]: round(prob, 4) for i, prob in enumerate(probabilities)
        },
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
    