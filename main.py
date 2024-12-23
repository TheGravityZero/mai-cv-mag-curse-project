import cv2
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
    return output.astype(np.uint8)


# Наложение маски на изображение
def overlay_mask(image: np.ndarray, mask: np.ndarray, alpha=0.6):
    """
    Наложение маски на изображение.
    image: исходное изображение (H, W, 3)
    mask: предсказанная маска (H, W)
    alpha: прозрачность маски
    """
    # Приведение маски к размеру изображения
    resized_mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
    
    # Создание цветной маски
    colored_mask = np.zeros_like(image)
    colored_mask[..., 1] = (resized_mask * 255).astype(np.uint8)  # Зеленый цвет для маски
    
    # Наложение маски
    overlay = ((1 - alpha) * image + alpha * colored_mask).astype(np.uint8)
    return overlay

# Эндпоинт для инференса
@app.post("/segment")
async def segment_image(file: UploadFile = File(...)):
    try:
        # Открытие изображения
        image_bytes = await file.read()
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        original_image = np.array(image)
        
        # Предобработка
        input_tensor = preprocess_image(image)
        
        # Инференс
        with torch.no_grad():
            output = model(input_tensor)
        
        # Постобработка
        segmented_mask  = postprocess_output(output)

        overlaid_image = overlay_mask(original_image, segmented_mask)
        
        # Конвертация результата в Base64 или сохранение
        segmented_pil_image = Image.fromarray(overlaid_image)
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
