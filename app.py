import cv2
import gradio as gr
import numpy as np
from skimage.filters import unsharp_mask

def invert(image):
    return cv2.bitwise_not(image)

def high_pass_filter(image):
    kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
    return cv2.filter2D(image, -1, kernel)

def apply_unsharp_mask(image):
    # Skimage's unsharp_mask returns a float image, so we need to convert it back to uint8
    image = unsharp_mask(image, radius=1, amount=1.5)
    return (image * 255).astype(np.uint8)

def histogram_equalization(image):
    return cv2.equalizeHist(image)

def clahe(image):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(image)

def enhance_image(image, enhancement_type):
    print(f"Enhancement type: {enhancement_type}")
    # Convert image to grayscale if it's not already
    if len(image.shape) == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    print(f"Image shape after conversion to grayscale (if needed): {image.shape}")

    if enhancement_type == "Invert":
        enhanced_image = invert(image)
    elif enhancement_type == "High Pass Filter":
        enhanced_image = high_pass_filter(image)
    elif enhancement_type == "Unsharp Masking":
        enhanced_image = apply_unsharp_mask(image)
    elif enhancement_type == "Histogram Equalization":
        enhanced_image = histogram_equalization(image)
    elif enhancement_type == "CLAHE":
        enhanced_image = clahe(image)

    print(f"Enhanced image shape: {enhanced_image.shape}")
    return enhanced_image

iface = gr.Interface(
    fn=enhance_image,
    inputs=[
        gr.Image(type="numpy", label="Upload Chest X-ray Image"),
        gr.Radio(choices=["Invert", "High Pass Filter", "Unsharp Masking", "Histogram Equalization", "CLAHE"], label="Enhancement Type")
    ],
    outputs=gr.Image(type="numpy", label="Enhanced Image"),
    title="Chest X-ray Image Enhancement"
)

iface.launch()
