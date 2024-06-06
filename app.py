import cv2
import gradio as gr
import numpy as np
import matplotlib.pyplot as plt

def process_image(image, enhancement_type, fix_monochrome=True):
    def image2array(path, fix_monochrome=True):
        # Read image with OpenCV
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

        # Image normalization
        image = image - np.min(image)
        image = image / np.max(image)
        image = (image * 255).astype(np.uint8)

    return image

    image = image2array(image)

    def apply_clahe(image):
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(10, 10))
        return clahe.apply(image)

    def invert(image):
        return cv2.bitwise_not(image)

    def hp_filter(image, kernel=None):
        if kernel is None:
            kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])

        return [cv2.filter2D(image, -1, kernel)]

    def unsharp_mask(image, radius=5, amount=2):

        def usm(image, radius, amount):
            blurred = cv2.GaussianBlur(image, (0, 0), radius)
            sharpened = cv2.addWeighted(image, 1.0 + amount, blurred, -amount, 0)
            return sharpened

        return [usm(image, radius, amount)]

    def hist_eq(image):
        return [cv2.equalizeHist(image)]

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
            return apply_clahe(image)

    image = image2array(image, fix_monochrome)
    enhanced_image = enhance_image(image, enhancement_type)
    return enhanced_image

iface = gr.Interface(
    fn=process_image,
    inputs=[
        gr.Image(type="numpy", label="Upload Chest X-ray Image"),
        gr.Radio(choices=["Invert", "High Pass Filter", "Unsharp Masking", "Histogram Equalization", "CLAHE"], label="Enhancement Type")
    ],
    outputs=gr.Image(type="numpy", label="Enhanced Image"),
    title="Chest X-ray Image Enhancement"
)

iface.launch()
