from fastapi import FastAPI, UploadFile, File, Form
import shutil
from api.predict import predict

app = FastAPI()

@app.post("/predict")
async def predict_api(
    image: UploadFile = File(...),
    text: str = Form(...)
):
    image_path = f"temp_{image.filename}"
    with open(image_path, "wb") as f:
        shutil.copyfileobj(image.file, f)

    return {"triage_level": predict(image_path, text)}
