import gradio as gr
import requests

API_URL = "https://lab2-api-latest.onrender.com"

def predict_image(file):
    try:
        # Gradio File object has .name (path on disk)
        with open(file.name, "rb") as f:
            files = {"file": ("image.jpg", f, "image/jpeg")}
            response = requests.post(f"{API_URL}/predict", files=files, timeout=10)
            response.raise_for_status()
            return response.json()
    except Exception as e:
        return {"error": str(e)}

iface = gr.Interface(
    fn=predict_image,
    inputs=gr.File(file_types=[".png", ".jpg", ".jpeg"]),
    outputs=gr.JSON(),
    title="Random Image Classifier",
    description="Upload an image and get a random prediction from the API."
)

if __name__ == "__main__":
    iface.launch()
