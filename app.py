import gradio as gr
import requests
from io import BytesIO

# URL of your deployed Render API
API_URL = "https://lab2-api-latest.onrender.com"

def predict_image(file):
    """
    Sends the uploaded image to the /predict endpoint of the API.
    Works with both local files and BytesIO objects from Gradio.
    """
    try:
        # Gradio provides a _io.BytesIO object
        file_bytes = file.read() if hasattr(file, "read") else file
        files = {"file": ("image.jpg", BytesIO(file_bytes), "image/jpeg")}

        response = requests.post(f"{API_URL}/predict", files=files, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}

# Gradio interface
iface = gr.Interface(
    fn=predict_image,
    inputs=gr.File(file_types=[".png", ".jpg", ".jpeg"]),
    outputs=gr.JSON(),
    title="Random Image Classifier",
    description="Upload an image and get a random classification prediction from the API."
)

if __name__ == "__main__":
    iface.launch()
