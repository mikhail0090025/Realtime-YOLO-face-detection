import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import torch.nn.init as init
from torchvision import transforms, utils

import numpy as np
from PIL import Image
import os
import math
import torchvision.models as models
from torchvision.ops import generalized_box_iou
from utils import load_checkpoint, get_predictions
from models import YOLOModel

# FastAPI imports
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, Response
from fastapi import UploadFile, File
import io

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_path = "checkpoint.pth"

model = YOLOModel(num_classes=1).to(device)

optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-2)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)
if os.path.exists(model_path):
    model, optimizer, scheduler, epochs, model_loss = load_checkpoint(model, optimizer, scheduler, model_path)

model.eval()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "This is a root of model server. Use /predict endpoint to get predictions."}

@app.post("/predict")
async def predict(tensor_file: UploadFile = File(...)):
    try:
        tensor_bytes = await tensor_file.read()
        image_np = np.load(io.BytesIO(tensor_bytes))  # shape (H, W, C)
        image = torch.from_numpy(image_np).permute(2,0,1).to(torch.uint8).to(device)
        with torch.no_grad():
            result_np = get_predictions(
                image_np, model, device=device, threshold=0.8,
                num_classes=1, max_iou=0.9, target_size=(280,280)
            )
        return {"boxes": result_np[0].tolist(), "classes": result_np[1].tolist()}
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        from fastapi import HTTPException
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health():
    return {"status": "ok"}