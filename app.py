import gradio as gr
import requests
from io import BytesIO

API_URL = "https://lab2-api-latest.onrender.com"

def predict_image(file):
    try:
        # HuggingFace gives a NamedString; get bytes
        file_bytes = file.read() if hasattr(file, "read") else bytes(file, "utf-8")
        files = {"file": ("image.jpg", BytesIO(file_bytes), "image/jpeg")}
        response = requests.post(f"{API_URL}/predict", files=files, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}

iface = gr.Interface(
    fn=predict_image,
    inputs=gr.File(file_types=[".png", ".jpg", ".jpeg"]),
    outputs=gr.JSON(),
    title="Random Image Classifier",
    description="Upload an image and get a random classification prediction from the API."
)

if __name__ == "__main__":
    iface.launch()
