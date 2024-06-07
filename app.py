import cv2
import gradio as gr
import numpy as np

### ENHANCEMENT CODE

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

def adjust_brightness_contrast(image, brightness=0, contrast=0):
    brightness = int((brightness - 50) * 2.55)
    contrast = int((contrast - 50) * 2.55)
    
    if brightness != 0:
        if brightness > 0:
            shadow = brightness
            highlight = 255
        else:
            shadow = 0
            highlight = 255 + brightness
        alpha_b = (highlight - shadow) / 255
        gamma_b = shadow
        image = cv2.addWeighted(image, alpha_b, image, 0, gamma_b)

    if contrast != 0:
        f = 131 * (contrast + 127) / (127 * (131 - contrast))
        alpha_c = f
        gamma_c = 127 * (1 - f)
        image = cv2.addWeighted(image, alpha_c, image, 0, gamma_c)

    return image

def adjust_saturation(image, saturation=1.0):
    if len(image.shape) == 2 or image.shape[2] == 1:
        return image
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv_image[:, :, 1] = np.clip(hsv_image[:, :, 1] * saturation, 0, 255)
    return cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)

def adjust_sharpness(image, sharpness=1.0):
    kernel = np.array([[-1, -1, -1], [-1, 9 * sharpness, -1], [-1, -1, -1]])
    return cv2.filter2D(image, -1, kernel)

def process_image(original_image, enhancement_type, brightness, contrast, saturation, sharpness, fix_monochrome=True):
    # Convert image to grayscale if desired
    if fix_monochrome:
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    
    # Normalize the image (assuming it's already a NumPy array)
    image = original_image - np.min(original_image)
    image = image / np.max(original_image)
    image = (image * 255).astype(np.uint8)
    
    # Enhance the image based on selection
    enhanced_image = enhance_image(image, enhancement_type)
    
    # Adjust brightness, contrast, saturation, and sharpness
    enhanced_image = adjust_brightness_contrast(enhanced_image, brightness, contrast)
    enhanced_image = adjust_saturation(enhanced_image, saturation)
    enhanced_image = adjust_sharpness(enhanced_image, sharpness)
    
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

### DICOM TO CSV EXTRACTOR
def extract_dicom_metadata(dicom_file):
    """Extract metadata from a DICOM file and return as a DataFrame."""
    ds = pydicom.dcmread(dicom_file.name)
    metadata = {elem.keyword: elem.value for elem in ds if elem.keyword}

    # Convert metadata to DataFrame
    df = pd.DataFrame(list(metadata.items()), columns=['Tag', 'Value'])
    return df

def display_metadata(dicom_file):
    """Display DICOM metadata as a table."""
    metadata_df = extract_dicom_metadata(dicom_file)
    return metadata_df



iface1 = gr.Interface(
    fn=process_image,
    inputs=[
        gr.Image(type="numpy", label="Upload Original Image"),
        gr.Radio(choices=["Invert", "High Pass Filter", "Unsharp Masking", "Histogram Equalization", "CLAHE"], label="Enhancement Type"),
        gr.Slider(0, 100, step=1, label="Brightness Level", value=50),
        gr.Slider(0, 100, step=1, label="Contrast Level", value=50),
        gr.Slider(0, 2, step=0.1, label="Saturation Level", value=1.0),
        gr.Slider(0, 10, step=0.1, label="Sharpness Level", value=1.0)
    ],
    outputs=[
        gr.Image(type="numpy", label="Enhanced Image"),
        gr.Textbox(label="MSE"),
        gr.Textbox(label="PSNR"),
        gr.Textbox(label="Maxerr"),
        gr.Textbox(label="L2Rat")
    ],
    title="Image Enhancement and Quality Evaluation"
)

iface2 = gr.Interface(
    fn=display_metadata,
    inputs=gr.File(label="Upload DICOM File"),
    outputs=gr.Dataframe(label="DICOM Metadata"),
    title="DICOM Metadata Extractor",
    description="Upload a DICOM file to extract and view its metadata."
)

iface = gr.TabbedInterface([iface1, iface2], ["Image Enhancement", "DICOM Metadata"])

iface.launch()
