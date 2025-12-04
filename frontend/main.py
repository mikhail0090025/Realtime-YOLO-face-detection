from fastapi import FastAPI, Request, HTTPException, UploadFile, File, Response
from pydantic import BaseModel
from sqlalchemy import text
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
import requests
import io
from PIL import Image
import numpy as np

app = FastAPI()

app.mount("/templates", StaticFiles(directory="templates"), name="templates")
app.mount("/styles", StaticFiles(directory="templates/styles"), name="styles")
app.mount("/js", StaticFiles(directory="templates/js"), name="js")

templates = Jinja2Templates(directory="templates")

@app.get("/")
def root():
    return {"message": "Frontend service is up!"}

@app.get("/home", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("main.html", {"request": request})

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Загружаем и подготавливаем изображение
    image = Image.open(io.BytesIO(await file.read())).convert("RGB").resize((240, 240), Image.BILINEAR)
    image_np = np.array(image)  # NumPy-массив (H, W, C)

    # Сохраняем массив в байтовый поток
    tensor_bytes = io.BytesIO()
    np.save(tensor_bytes, image_np)
    tensor_bytes.seek(0)

    # Отправляем как файл
    response = requests.post(
        "http://model_service:8001/predict",
        files={"tensor_file": ("image.npy", tensor_bytes, "application/octet-stream")}
    )

    if response.status_code != 200:
        raise HTTPException(status_code=500, detail="Model service error")

    return JSONResponse(content=response.json())

@app.get("/health")
def health():
    return {"status": "ok"}