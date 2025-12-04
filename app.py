import gradio as gr
import requests

# URL of your deployed Render API
API_URL = "https://lab2-api-latest.onrender.com"

def predict_image(file):
    try:
        # Read the file correctly as bytes
        file_bytes = file.read() if hasattr(file, "read") else None
        if file_bytes is None:
            return {"error": "Invalid file"}

        # Send the file as multipart/form-data
        files = {"file": ("image.jpg", file_bytes, "image/jpeg")}
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
    description="Upload an image and get a random prediction from the API."
)

if __name__ == "__main__":
    iface.launch()
