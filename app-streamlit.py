import streamlit as st
import cv2
import numpy as np
import pydicom
from pydicom.dataset import Dataset, FileDataset
from pydicom.uid import generate_uid
from tensorflow.keras.preprocessing import image
from google.cloud import storage
import os
import io
from PIL import Image
import uuid
import pandas as pd
from tensorflow.keras.models import load_model
import tensorflow as tf
from datetime import datetime

# Environment Configuration
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = "./da-kalbe-63ee33c9cdbb.json"
bucket_name = "da-kalbe-ml-result-png"
storage_client = storage.Client()
bucket = storage_client.bucket(bucket_name)

# Utility Functions
def upload_to_gcs(image_data: io.BytesIO, filename: str, content_type='application/dicom'):
    """Uploads an image to Google Cloud Storage."""
    try:
        blob = bucket.blob(filename)
        blob.upload_from_file(image_data, content_type=content_type)
        st.write("File ready to be seen in OHIF Viewer.")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")

def png_to_dicom(image: Image, patient_name="Anonymous", patient_id="0000"):
    """Converts a PNG image to DICOM format."""
    # Create the metadata for the DICOM file
    file_meta = Dataset()
    file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.2" # CT Image Storage
    file_meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
    file_meta.ImplementationClassUID = pydicom.uid.generate_uid()
    file_meta.ImplementationVersionName = "Python PyDICOM"

    # Create the FileDataset (a Dataset with file meta information)
    ds = FileDataset("converted_image.dcm", {}, file_meta=file_meta, preamble=b"\0" * 128)
    ds.PatientName = patient_name
    ds.PatientID = patient_id
    ds.Modality = "CT"  # Other
    ds.ContentDate = str(datetime.today().date()).replace("-", "")
    ds.ContentTime = str(datetime.now().time()).replace(":", "").split(".")[0]

    np_image = np.array(image)
    if image.mode == "L":
        ds.Rows, ds.Columns = np_image.shape
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.BitsAllocated = 8
        ds.BitsStored = 8
        ds.HighBit = 7
        ds.PixelRepresentation = 0
        ds.PixelData = np_image.tobytes()
    elif image.mode == "RGB":
        ds.Rows = image.height
        ds.Columns = image.width
        ds.PhotometricInterpretation = "RGB"
        ds.SamplesPerPixel = 3
        ds.BitsStored = 8
        ds.BitsAllocated = 8
        ds.HighBit = 7
        ds.PixelRepresentation = 0
        ds.PlanarConfiguration = 0
        ds.PixelData = np_image.tobytes()
    else:
        raise ValueError("Unsupported image mode")

    # Set the transfer syntax
    ds.file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian
    return ds

def save_dicom_to_bytes(dicom):
    dicom_bytes = io.BytesIO()
    dicom.save_as(dicom_bytes)
    dicom_bytes.seek(0)
    return dicom_bytes

def upload_folder_images(image_path, enhanced_image_path):
    # Extract the base name of the uploaded image without the extension
    folder_name = f"{uploaded_file.name}"
    # Create the folder in Cloud Storage
    bucket.blob(folder_name + '/').upload_from_string('', content_type='application/x-www-form-urlencoded')
    
    # Open the images
    original_image = Image.open(image_path)
    enhanced_image = Image.open(enhanced_image_path)
    
    # Convert images to DICOM
    original_dicom = png_to_dicom(original_image)
    enhanced_dicom = png_to_dicom(enhanced_image)

    # Convert DICOM to byte stream for uploading
    original_dicom_bytes = io.BytesIO()
    enhanced_dicom_bytes = io.BytesIO()
    original_dicom.save_as(original_dicom_bytes)
    enhanced_dicom.save_as(enhanced_dicom_bytes)
    original_dicom_bytes.seek(0)
    enhanced_dicom_bytes.seek(0)

    # Upload images to GCS
    upload_to_gcs(original_dicom_bytes, folder_name + '/' + 'original_image.dcm', content_type='application/dicom')
    upload_to_gcs(enhanced_dicom_bytes, folder_name + '/' + enhancement_type + '.dcm', content_type='application/dicom')

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

# Streamlit Interface
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
            folder_name = f"{uploaded_file.name}"

            # Create the folder in Cloud Storage
            bucket.blob(folder_name + '/').upload_from_string('', content_type='application/x-www-form-urlencoded')
            
            # Upload the gradcam image to Google Cloud Storage
            destination_blob_name = f"gradcam_images/{uploaded_gradcam_file.name}_{idx+1}.png"
            upload_to_gcs(gradcam_image_path, destination_blob_name)
