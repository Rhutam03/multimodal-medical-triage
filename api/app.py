from fastapi import FastAPI, UploadFile, File, Form
import shutil
from core.inference import predict

app = FastAPI(title="Multimodal Medical Triage API")


@app.get("/")
def root():
    return {"status": "ok"}


@app.post("/predict")
async def predict_api(
    image: UploadFile = File(...),
    text: str = Form(...)
):
    image_path = f"/tmp/{image.filename}"

    with open(image_path, "wb") as f:
        shutil.copyfileobj(image.file, f)

    triage_level = predict(image_path, text)

    return {"triage_level": triage_level}

