from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse
from google.cloud import storage
import os
import io
import numpy as np
import cv2
from PIL import Image
import uuid
import base64

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = "./da-kalbe-63ee33c9cdbb.json"
bucket_name = "da-kalbe-ml-result-png"
storage_client = storage.Client()
bucket = storage_client.bucket(bucket_name)

app = FastAPI()

# Function to upload file to Google Cloud Storage
def upload_to_gcs(image: Image, filename: str):
    """Uploads an image to Google Cloud Storage."""
    try:
        blob = bucket.blob(filename)
        image_buffer = io.BytesIO()
        image.save(image_buffer, format='PNG')
        image_buffer.seek(0)
        blob.upload_from_file(image_buffer, content_type='image/png')
    except Exception as e:
        return {'error': f"An unexpected error occurred: {e}"}

def upload_folder_images(image_path, enhanced_image_path):
    # Extract the base name of the uploaded image without the extension
    folder_name = os.path.splitext(os.path.basename(image_path))[0]

    # Create the folder in Cloud Storage
    bucket.blob(folder_name + '/').upload_from_string('', content_type='application/x-www-form-urlencoded')

    # Open the images
    original_image = Image.open(image_path)
    enhanced_image = Image.open(enhanced_image_path)

    # Upload images to GCS
    upload_to_gcs(original_image, folder_name + '/' + 'original_image.png')
    upload_to_gcs(enhanced_image, folder_name + '/' + enhancement_type + '.png')

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

@app.post("/process_image")
async def process_image_api(image: UploadFile = File(...), enhancement_type: str = Form(...)):
    """Processes an uploaded image and returns the enhanced image and metrics."""
    
    if not image:
        return JSONResponse(status_code=400, content={'error': 'No image file provided'})

    allowed_extensions = {'png', 'jpg', 'jpeg'}
    if '.' not in image.filename or image.filename.split('.')[-1].lower() not in allowed_extensions:
        return JSONResponse(status_code=400, content={'error': 'Invalid image file'})

    try:
        # Open the image using Pillow
        image_pil = Image.open(image.file).convert('RGB') 

        # Convert to NumPy array
        image_np = np.array(image_pil)

        # Apply image processing
        enhanced_image, mse, psnr, maxerr, l2rat = process_image(image_np, enhancement_type)

        # Convert processed image back to PIL format for saving
        enhanced_image_pil = Image.fromarray(enhanced_image)

        # Save to in-memory buffer
        image_buffer = io.BytesIO()
        enhanced_image_pil.save(image_buffer, format='PNG') 
        image_buffer.seek(0)

        # Encode to base64
        image_base64 = base64.b64encode(image_buffer.getvalue()).decode('utf-8')

        response = {
            'message': 'Image processed successfully!',
            'processed_image': image_base64,
            'mse': float(mse), 
            'psnr': float(psnr), 
            'maxerr': float(maxerr),  
            'l2rat': float(l2rat)   
        }   
        upload_folder_images(image, enhanced_image_pil)
        return JSONResponse(status_code=200, content=response)

    except Exception as e:
        return JSONResponse(status_code=500, content={'error': f'Error processing image: {str(e)}'})
