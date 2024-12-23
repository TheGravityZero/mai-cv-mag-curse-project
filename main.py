import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import torchvision.transforms as T
import uvicorn

from io import BytesIO
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image


# Устройство
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Загрузка модели
MODEL_PATH = "model.pt"


model = torch.load('model.pt', map_location=device)
model.to(device)
model.eval()

# Инициализация FastAPI
app = FastAPI()

# Предобработка изображения
def preprocess_image(image: Image.Image):
    """Предобработка изображения"""
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    transform = T.Compose([
        T.Resize((768, 1152)),
        T.ToTensor(),
        T.Normalize(mean, std)
    ])
    return transform(image).unsqueeze(0).to(device)

# Постобработка результата
def postprocess_output(output):
    output = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()
    # Скейлинг значений в маске
    # Хотим чтобы пиксели были распределены от 0 до 255. Делим на 23, так как всего 23 класса
    return (output * 255 / 23).astype(np.uint8)

# Эндпоинт для инференса
@app.post("/segment")
async def segment_image(file: UploadFile = File(...)):
    try:
        # Открытие изображения
        image_bytes = await file.read()
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        
        # Предобработка
        input_tensor = preprocess_image(image)
        
        # Инференс
        with torch.no_grad():
            output = model(input_tensor)
        
        # Постобработка
        segmented_mask  = postprocess_output(output)
        
        # Конвертация результата в Base64 или сохранение
        colored_mask = cv2.applyColorMap(segmented_mask, cv2.COLORMAP_HSV)
        segmented_pil_image = Image.fromarray(colored_mask)
        saved_dir = os.getcwd() + "/results/"
        save_path = os.path.join(saved_dir, f"segmented_{file.filename}")
        segmented_pil_image.save(save_path, format="PNG")
        
        return JSONResponse(
            content={
                "message": "Segmentation completed successfully.",
                "saved_path": save_path
            }
        )
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

# Запуск приложения
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
