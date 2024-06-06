import cv2
import numpy as np
import gradio as gr
from skimage import exposure
from skimage.filters import unsharp_mask

def invert(image):
    return cv2.bitwise_not(image)

def high_pass_filter(image):
    kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
    return cv2.filter2D(image, -1, kernel)

def apply_unsharp_mask(image):
    return unsharp_mask(image, radius=1, amount=1.5)

def histogram_equalization(image):
    return cv2.equalizeHist(image)

def clahe(image):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(image)

def enhance_image(image, enhancement_type):
    if enhancement_type == "Invert":
        return invert(image)
    elif enhancement_type == "High Pass Filter":
        return high_pass_filter(image)
    elif enhancement_type == "Unsharp Masking":
        return apply_unsharp_mask(image)
    elif enhancement_type == "Histogram Equalization":
        return histogram_equalization(image)
    elif enhancement_type == "CLAHE":
        return clahe(image)

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