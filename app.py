from fastapi import FastAPI, UploadFile, File
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io
from fastapi.middleware.cors import CORSMiddleware



# loading the saved model
checkpoint = torch.load("model_final1.pth", map_location="cpu")

model = models.efficientnet_b0(weights=None)
model.classifier[1] = nn.Linear(
    model.classifier[1].in_features,
    checkpoint["num_classes"]
)

model.load_state_dict(checkpoint["model_state"])
model.eval()

# metadata 
classes = checkpoint["class_names"]
IMG_SIZE = checkpoint["input_size"]
mean = checkpoint["mean"]
std = checkpoint["std"]

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

#api

app = FastAPI(
    title="Skin Disease Detection API",
    description="Upload a skin image and get prediction",
    version="1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow all (for now)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def home():
    return {"message": "API is running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        image = transform(image).unsqueeze(0)

        with torch.no_grad():
            output = model(image)
            probs = torch.softmax(output, dim=1)

        confidence, pred = torch.max(probs, 1)

        #Top-3 predictions
        top3_prob, top3_idx = torch.topk(probs, 3)

        #top3 = []
        #for i in range(1):
        #    top3.append({
        #        "disease": classes[top3_idx[0][i]],
        #        "confidence": float(top3_prob[0][i])
        #    })

        return {
            "prediction": classes[pred.item()],
            "confidence": float(confidence.item()),
        #    "top3": top3
        }

    except Exception as e:
        return {"error": str(e)}