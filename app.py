import cv2
import gradio as gr
import numpy as np
from PIL import Image
import io

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
    # Convert image to grayscale if desired
    if fix_monochrome:
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    
    # Normalize the image (assuming it's already a NumPy array)
    image = original_image - np.min(original_image)
    image = image / np.max(original_image)
    image = (image * 255).astype(np.uint8)
    
    # Enhance the image based on selection
    enhanced_image = enhance_image(image, enhancement_type)
    
    # Calculate image quality metrics
    mse = calculate_mse(original_image, enhanced_image)
    psnr = calculate_psnr(original_image, enhanced_image)
    maxerr = calculate_maxerr(original_image, enhanced_image)
    l2rat = calculate_l2rat(original_image, enhanced_image)
    
    return enhanced_image, mse, psnr, maxerr, l2rat

def apply_clahe(image):
    clahe = cv2.createCLAHE(clipLimit=40.0, tileGridSize=(10, 10))
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

def save_image(image):
    img = Image.fromarray(image)
    byte_io = io.BytesIO()
    img.save(byte_io, format='PNG')
    return byte_io.getvalue()

iface = gr.Interface(
    fn=process_image,
    inputs=[
        gr.Image(type="numpy", label="Upload Original Image"),
        gr.Radio(choices=["Invert", "High Pass Filter", "Unsharp Masking", "Histogram Equalization", "CLAHE"], label="Enhancement Type")
    ],
    outputs=[
        gr.Image(type="numpy", label="Enhanced Image"),
        gr.Textbox(label="MSE"),
        gr.Textbox(label="PSNR"),
        gr.Textbox(label="Maxerr"),
        gr.Textbox(label="L2Rat"),
        gr.Button(label="Save Image", type="button", onclick=save_image)
    ],
    title="Image Enhancement and Quality Evaluation"
)

iface.launch()
