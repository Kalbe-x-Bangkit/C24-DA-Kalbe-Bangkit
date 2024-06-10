import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import pandas as pd
from google.cloud import storage
import os
import io
import base64
from PIL import Image
import uuid
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse
from fastapi.requests import Request
from starlette.responses import RedirectResponse 
import threading
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
import time
import socket

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] ="./da-kalbe-63ee33c9cdbb.json"
bucket_name = "da-kalbe-ml-result-png"
storage_client = storage.Client()
bucket = storage_client.bucket(bucket_name)

# Function to upload file to Google Cloud Storage
def upload_to_gcs(image: Image, filename: str):
    """Uploads an image to Google Cloud Storage."""
    try:
        blob = bucket.blob(filename)
        image_buffer = io.BytesIO()
        image.save(image_buffer, format='PNG')
        image_buffer.seek(0)
        blob.upload_from_file(image_buffer, content_type='image/png')
        st.write("File ready to be seen in OHIF Viewer.")
    except GoogleCloudError as gce:
        st.error(f"Error uploading to Google Cloud Storage: {gce}")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}") 
    
def upload_folder_images(image, enhanced_image):
    # Create unique folder name
    folder_name = str(uuid.uuid4())

    # Create the folder in Cloud Storage
    bucket.blob(folder_name + '/').upload_from_string('', content_type='application/x-www-form-urlencoded')

    upload_to_gcs(image, folder_name + '/' + 'original_image')
    upload_to_gcs(enhanced_image, folder_name + '/' + enhancement_type)

class GradCAM:
    def __init__(self, model, layer_name):
        self.model = model
        self.layer_name = layer_name
        self.grad_model = tf.keras.models.Model(
            [self.model.inputs],
            [self.model.get_layer(layer_name).output, self.model.output]
        )

    def __call__(self, img_array, cls):
        with tf.GradientTape() as tape:
            conv_outputs, predictions = self.grad_model(img_array)
            loss = predictions[:, cls]

        output = conv_outputs[0]
        grads = tape.gradient(loss, conv_outputs)[0]
        gate_f = tf.cast(output > 0, 'float32')
        gate_r = tf.cast(grads > 0, 'float32')
        guided_grads = gate_f * gate_r * grads

        weights = tf.reduce_mean(guided_grads, axis=(0, 1))

        cam = np.zeros(output.shape[0:2], dtype=np.float32)

        for index, w in enumerate(weights):
            cam += w * output[:, :, index]

        cam = cv2.resize(cam.numpy(), (224, 224))
        cam = np.maximum(cam, 0)
        cam = cam / cam.max()

        return cam

def apply_heatmap(img, heatmap, heatmap_ratio=0.6):
    heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    return np.uint8(heatmap * heatmap_ratio + img * (1 - heatmap_ratio))

def load_image(img_path, df, preprocess=True, height=320, width=320):
    mean, std = get_mean_std_per_batch(img_path, df, height, width)
    x = image.load_img(img_path, target_size=(height, width))
    x = image.img_to_array(x)
    if preprocess:
        x -= mean
        x /= std
        x = np.expand_dims(x, axis=0)
    return x

def get_mean_std_per_batch(image_path, df, height=320, width=320):
    sample_data = []
    for idx, img in enumerate(df.sample(100)["Image Index"].values):
        sample_data.append(
            np.array(image.load_img(image_path, target_size=(height, width))))
    mean = np.mean(sample_data[0])
    std = np.std(sample_data[0])
    return mean, std

def compute_gradcam(img, model, df, labels, layer_name='bn'):
    preprocessed_input = load_image(img, df)
    predictions = model.predict(preprocessed_input)

    top_indices = np.argsort(predictions[0])[-3:][::-1]
    top_labels = [labels[i] for i in top_indices]
    top_predictions = [predictions[0][i] for i in top_indices]

    original_image = load_image(img, df, preprocess=False)

    grad_cam = GradCAM(model, layer_name)

    gradcam_images = []
    for i in range(3):
        idx = top_indices[i]
        label = top_labels[i]
        prob = top_predictions[i]

        gradcam = grad_cam(preprocessed_input, idx)
        gradcam_image = apply_heatmap(original_image, gradcam)
        gradcam_images.append((gradcam_image, f"{label}: p={prob:.3f}"))

    return gradcam_images

def calculate_mse(original_image, enhanced_image):
    mse = np.mean((original_image - enhanced_image) ** 2)
    return mse

def calculate_psnr(original_image, enhanced_image):
    mse = calculate_mse(original_image, enhanced_image)
    if mse == 0:
        return float('inf')
    max_pixel_value = 255.0
    psnr = 20 * np.log10(max_pixel_value / np.sqrt(mse))
    return psnr

def calculate_maxerr(original_image, enhanced_image):
    maxerr = np.max((original_image - enhanced_image) ** 2)
    return maxerr

def calculate_l2rat(original_image, enhanced_image):
    l2norm_ratio = np.sum(original_image ** 2) / np.sum((original_image - enhanced_image) ** 2)
    return l2norm_ratio

def process_image(original_image, enhancement_type, fix_monochrome=True):
    if fix_monochrome and original_image.shape[-1] == 3:
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

    image = original_image - np.min(original_image)
    image = image / np.max(original_image)
    image = (image * 255).astype(np.uint8)

    enhanced_image = enhance_image(image, enhancement_type)

    mse = calculate_mse(original_image, enhanced_image)
    psnr = calculate_psnr(original_image, enhanced_image)
    maxerr = calculate_maxerr(original_image, enhanced_image)
    l2rat = calculate_l2rat(original_image, enhanced_image)

    return enhanced_image, mse, psnr, maxerr, l2rat

def apply_clahe(image):
    clahe = cv2.createCLAHE(clipLimit=40.0, tileGridSize=(8, 8))
    return clahe.apply(image)

def invert(image):
    return cv2.bitwise_not(image)

def hp_filter(image, kernel=None):
    if kernel is None:
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    return cv2.filter2D(image, -1, kernel)

def unsharp_mask(image, radius=5, amount=2):
    def usm(image, radius, amount):
        blurred = cv2.GaussianBlur(image, (0, 0), radius)
        sharpened = cv2.addWeighted(image, 1.0 + amount, blurred, -amount, 0)
        return sharpened
    return usm(image, radius, amount)

def hist_eq(image):
    return cv2.equalizeHist(image)

def enhance_image(image, enhancement_type):
    if enhancement_type == "Invert":
        return invert(image)
    elif enhancement_type == "High Pass Filter":
        return hp_filter(image)
    elif enhancement_type == "Unsharp Masking":
        return unsharp_mask(image)
    elif enhancement_type == "Histogram Equalization":
        return hist_eq(image)
    elif enhancement_type == "CLAHE":
        return apply_clahe(image)
    else:
        raise ValueError(f"Unknown enhancement type: {enhancement_type}")
    
def run_streamlit():
    st.title("Image Enhancement and Quality Evaluation")

    uploaded_file = st.file_uploader("Upload Original Image", type=["png", "jpg", "jpeg"])
    enhancement_type = st.radio("Enhancement Type", ["Invert", "High Pass Filter", "Unsharp Masking", "Histogram Equalization", "CLAHE"])

    if uploaded_file is not None:
        original_image = np.array(image.load_img(uploaded_file, color_mode='rgb' if enhancement_type == "Invert" else 'grayscale'))
        enhanced_image, mse, psnr, maxerr, l2rat = process_image(original_image, enhancement_type)

        st.image(original_image, caption='Original Image', use_column_width=True)
        st.image(enhanced_image, caption='Enhanced Image', use_column_width=True)

        st.write("MSE:", mse)
        st.write("PSNR:", psnr)
        st.write("Maxerr:", maxerr)
        st.write("L2Rat:", l2rat)

        # Save enhanced image to a file
        enhanced_image_path = "enhanced_image.png"
        cv2.imwrite(enhanced_image_path, enhanced_image)

        # Save original image to a file
        original_image_path = "original_image.png"
        cv2.imwrite(original_image_path, original_image)
        
        upload_folder_images(original_image_path, enhanced_image_path)

    st.title("Grad-CAM Visualization")

    uploaded_gradcam_file = st.file_uploader("Upload Image for Grad-CAM", type=["png", "jpg", "jpeg"], key="gradcam")
    if uploaded_gradcam_file is not None:
        df_file = st.file_uploader("Upload DataFrame for Mean/Std Calculation", type=["csv"])
        labels = st.text_area("Labels", placeholder="Enter labels separated by commas")
        model_path = st.text_input("Model Path", 'model/densenet.hdf5')
        pretrained_model_path = st.text_input("Pretrained Model Path", 'model/pretrained_model.h5')

        if df_file and labels and model_path and pretrained_model_path:
            df = pd.read_csv(df_file)
            labels = labels.split(',')
            model = load_model(model_path)
            pretrained_model = load_model(pretrained_model_path)
            gradcam_images = compute_gradcam(uploaded_gradcam_file, pretrained_model, df, labels)

            for idx, (gradcam_image, label) in enumerate(gradcam_images):
                st.image(gradcam_image, caption=f'Grad-CAM {idx+1}: {label}', use_column_width=True)
                # Save gradcam image to a file
                gradcam_image_path = f"gradcam_image_{idx+1}.png"
                cv2.imwrite(gradcam_image_path, gradcam_image)
                # Create unique folder name
                folder_name = str(uuid.uuid4())

                # Create the folder in Cloud Storage
                bucket.blob(folder_name + '/').upload_from_string('', content_type='application/x-www-form-urlencoded')
                
                # Upload the gradcam image to Google Cloud Storage
                destination_blob_name = f"gradcam_images/{uploaded_gradcam_file.name}_{idx+1}.png"
                upload_to_gcs(gradcam_image_path, destination_blob_name)

# FastAPI setup
api_app = FastAPI()

# Add CORS middleware to allow Streamlit to make requests to FastAPI
api_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this to the specific origins you need
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@api_app.post("/process_image")
async def process_image_api(image: UploadFile = File(...), enhancement_type: str = Form(...)):
    if not image:
        return JSONResponse(status_code=400, content={'error': 'No image file provided'})

    allowed_extensions = {'png', 'jpg', 'jpeg'}
    if '.' not in image.filename or image.filename.split('.')[-1].lower() not in allowed_extensions:
        return JSONResponse(status_code=400, content={'error': 'Invalid image file'})

    try:
        image_pil = Image.open(image.file).convert('RGB')
        image_np = np.array(image_pil)
        enhanced_image, mse, psnr, maxerr, l2rat = process_image(image_np, enhancement_type)
        enhanced_image_pil = Image.fromarray(enhanced_image)

        image_buffer = io.BytesIO()
        enhanced_image_pil.save(image_buffer, format='PNG')
        image_buffer.seek(0)

        image_base64 = base64.b64encode(image_buffer.getvalue()).decode('utf-8')
        response = {
            'message': 'Image processed successfully!',
            'processed_image': image_base64,
            'mse': float(mse),
            'psnr': float(psnr),
            'maxerr': float(maxerr),
            'l2rat': float(l2rat)
        }

        return JSONResponse(status_code=200, content=response)
    except Exception as e:
        return JSONResponse(status_code=500, content={'error': f'Error processing image: {str(e)}'})

def find_available_port(start_port=8001, end_port=8100):
    for port in range(start_port, end_port):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(('localhost', port)) != 0:
                return port
    raise RuntimeError("No available ports in the specified range")

def run_fastapi():
    try:
        port = find_available_port(8001, 8100)
        uvicorn.run(api_app, host="0.0.0.0", port=port)
    except Exception as e:
        st.error(f"Failed to start FastAPI: {str(e)}")
    
if __name__ == "__main__":
    # Run FastAPI in a separate thread
    fastapi_thread = threading.Thread(target=run_fastapi, daemon=True)
    fastapi_thread.start()
    
    # Allow FastAPI to start
    time.sleep(2)
    
    # Run Streamlit
    os.system("streamlit run app-api.py")