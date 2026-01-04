import gradio as gr
import tempfile
from core.inference import predict
import torch


def run_inference(image, text):
    if image is None or text.strip() == "":
        return "Please upload an image and enter text."

    # save image temporarily
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        image.save(tmp.name)
        image_path = tmp.name

    triage_level = predict(image_path, text)

    label_map = {
        0: "Low Urgency",
        1: "Medium Urgency",
        2: "High Urgency"
    }

    return f"Triage Level: {triage_level} ({label_map.get(triage_level)})"


demo = gr.Interface(
    fn=run_inference,
    inputs=[
        gr.Image(type="pil", label="Upload Medical Image"),
        gr.Textbox(lines=4, label="Enter Symptoms / Clinical Text")
    ],
    outputs=gr.Textbox(label="Prediction"),
    title="Multimodal Medical Triage System",
    description="AI-based triage using medical image + patient text"
)

demo.launch()



