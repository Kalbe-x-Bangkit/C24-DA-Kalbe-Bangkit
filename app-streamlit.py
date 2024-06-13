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
import keras

# Environment Configuration ###############################################################

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = "./da-kalbe-63ee33c9cdbb.json"
bucket_name = "da-kalbe-ml-result-png"
storage_client = storage.Client()
bucket_result = storage_client.bucket(bucket_name)
bucket_name_load = "da-ml-models"
bucket_load = storage_client.bucket(bucket_name_load)

# Utility Functions #
# object detection ########################################################################

H = 224
W = 224

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("model-detection.h5", compile=False)
    model.compile(
        loss={
            "bbox": "mse",
            "class": "sparse_categorical_crossentropy"
        },
        optimizer=tf.keras.optimizers.Adam(),
        metrics={
            "bbox": ['mse'],
            "class": ['accuracy']
        }
    )
    return model

def preprocess_image(image):
    """ Preprocess the image to the required size and normalization. """
    image = cv2.resize(image, (W, H))
    image = (image - 127.5) / 127.5  # Normalize to [-1, +1]
    image = np.expand_dims(image, axis=0).astype(np.float32)
    return image

def predict(model, image):
    """ Predict bounding box and label for the input image. """
    pred_bbox, pred_class = model.predict(image)
    pred_label_confidence = np.max(pred_class, axis=1)[0]
    pred_label = np.argmax(pred_class, axis=1)[0]
    return pred_bbox[0], pred_label, pred_label_confidence

def draw_bbox(image, bbox):
    """ Draw bounding box on the image. """
    h, w, _ = image.shape
    x1, y1, x2, y2 = bbox
    x1, y1, x2, y2 = int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h)
    image = cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
    return image

st.title("Chest X-ray Disease Detection")

st.write("Upload a chest X-ray image and click on 'Detect' to find out if there's any disease.")

model = load_model()

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    st.image(image, caption='Uploaded Image.', use_column_width=True)

    if st.button('Detect'):
        st.write("Processing...")
        input_image = preprocess_image(image)
        pred_bbox, pred_label, pred_label_confidence = predict(model, input_image)

        label_mapping = {
            0: 'Atelectasis',
            1: 'Cardiomegaly',
            2: 'Effusion',
            3: 'Infiltrate',
            4: 'Mass',
            5: 'Nodule',
            6: 'Pneumonia',
            7: 'Pneumothorax'
        }

        if pred_label_confidence < 0.01:
            st.write("May not detect a disease.")
        else:
            pred_label_name = label_mapping[pred_label]
            st.write(f"Prediction Label: {pred_label_name}")
            st.write(f"Prediction Bounding Box: {pred_bbox}")
            st.write(f"Prediction Confidence: {pred_label_confidence:.2f}")

            output_image = draw_bbox(image.copy(), pred_bbox)
            st.image(output_image, caption='Detected Image.', use_column_width=True)

# Upload to GCS ###########################################################################

# Dictionaries to track InstanceNumbers and StudyInstanceUIDs per filename
instance_numbers = {}
study_uids = {}

def upload_to_gcs(image_data: io.BytesIO, filename: str, content_type='application/dicom'):
    """Uploads an image to Google Cloud Storage."""
    try:
        blob = bucket_result.blob(filename)
        blob.upload_from_file(image_data, content_type=content_type)
        st.write("File ready to be seen in OHIF Viewer.")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")

def load_dicom_from_gcs(dicom_name: str = "dicom_00000001_000.dcm"):
    # Get the blob object
    blob = bucket_load.blob(dicom_name)

    # Download the file as a bytes object
    dicom_bytes = blob.download_as_bytes()

    # Wrap bytes object into BytesIO (file-like object)
    dicom_stream = io.BytesIO(dicom_bytes)

    # Load the DICOM file
    ds = pydicom.dcmread(dicom_stream)

    return ds

def png_to_dicom(image_path: str, image_name: str, file_name: str, instance_number: int = 1, dicom: str = None, study_instance_uid: str = None, ):
    global instance_numbers, study_uids
    """Converts a PNG image to DICOM, setting related Study/Series UIDs.

    Args:
        image_path (str): Path to the PNG image file.
        image_name (str): Desired filename for the output DICOM file.
        dicom (str, optional): Path to a template DICOM file. If None,
                                a default template will be used.
                                Defaults to None.
        instance_number (int): Instance number to assign to the DICOM file.
                                Defaults to 1.

    Returns:
        pydicom.Dataset: The modified DICOM dataset.

    Raises:
        ValueError: If the PNG image mode is unsupported.
    """
    # Load the template DICOM file
    ds = load_dicom_from_gcs() if dicom is None else load_dicom_from_gcs(dicom)

    # Process the image
    jpg_image = Image.open(image_path)  # the PNG or JPG file to be replaced
    print("Image Mode:", jpg_image.mode)

    if jpg_image.mode in ('L', 'RGBA', 'RGB'):
        if jpg_image.mode == 'RGBA':
            np_image = np.array(jpg_image.getdata(), dtype=np.uint8)[:,:3]
        else:
            np_image = np.array(jpg_image.getdata(),dtype=np.uint8)

        ds.Rows = jpg_image.height
        ds.Columns = jpg_image.width
        ds.PhotometricInterpretation = "MONOCHROME1" if jpg_image.mode == 'L' else "RGB"
        ds.SamplesPerPixel = 1 if jpg_image.mode == 'L' else 3
        ds.BitsStored = 8
        ds.BitsAllocated = 8
        ds.HighBit = 7
        ds.PixelRepresentation = 0
        ds.PixelData = np_image.tobytes()

        if not hasattr(ds, 'PatientName') or ds.PatientName == '':
            ds.PatientName = os.path.splitext(file_name)[0]  # Remove extension

        ds.SeriesDescription = 'original image' if image_name == 'original_image.dcm' else enhancement_type

        if hasattr(ds, 'StudyDescription'):
            del ds.StudyDescription

        if study_instance_uid:
            ds.StudyInstanceUID = study_instance_uid
        else:
            # Check if a StudyInstanceUID exists for the file name
            if file_name in study_uids:
                ds.StudyInstanceUID = study_uids[file_name]
                print(f"Reusing StudyInstanceUID for '{file_name}'")
            else:
                # Generate a new StudyInstanceUID and store it
                new_study_uid = generate_uid()
                study_uids[file_name] = new_study_uid
                ds.StudyInstanceUID = new_study_uid
                print(f"New StudyInstanceUID generated for '{file_name}'")

        # Generate a new SeriesInstanceUID and SOPInstanceUID for the added image
        ds.SeriesInstanceUID = generate_uid()
        ds.SOPInstanceUID = generate_uid()

        if hasattr(ds, 'InstanceNumber'):
            instance_numbers[file_name] = int(ds.InstanceNumber) + 1
        else:
            # Manage InstanceNumber based on filename
            if file_name in instance_numbers:
                instance_numbers[file_name] += 1
            else:
                instance_numbers[file_name] = 1
        ds.InstanceNumber = int(instance_numbers[file_name])

        ds.save_as(image_name)
    else:
        raise ValueError(f"Unsupported image mode: {jpg_image.mode}")

    return ds


def upload_folder_images(original_image_path, enhanced_image_path, file_name):
    # Convert images to DICOM if result is png
    if not original_image_path.lower().endswith('.dcm'):
        original_dicom = png_to_dicom(original_image_path, "original_image.dcm", file_name=file_name)
    else:
        original_dicom = pydicom.dcmread(original_image_path)
    study_instance_uid = original_dicom.StudyInstanceUID

    # Use StudyInstanceUID as folder name
    folder_name = study_instance_uid

    # Create the folder in Cloud Storage
    bucket_result.blob(folder_name + '/').upload_from_string('', content_type='application/x-www-form-urlencoded')

    enhancement_name = enhancement_type.split('_')[-1]

    enhanced_dicom = png_to_dicom(enhanced_image_path, enhancement_name + ".dcm", study_instance_uid=study_instance_uid, file_name=file_name)

    # Convert DICOM to byte stream for uploading
    original_dicom_bytes = io.BytesIO()
    enhanced_dicom_bytes = io.BytesIO()
    original_dicom.save_as(original_dicom_bytes)
    enhanced_dicom.save_as(enhanced_dicom_bytes)
    original_dicom_bytes.seek(0)
    enhanced_dicom_bytes.seek(0)

    # Upload images to GCS
    upload_to_gcs(original_dicom_bytes, folder_name + '/' + 'original_image.dcm', content_type='application/dicom')
    upload_to_gcs(enhanced_dicom_bytes, folder_name + '/' + enhancement_name + '.dcm', content_type='application/dicom')

# Grad cam ################################################################################

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

# Image enhancement #######################################################################

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

# Other Utils #############################################################################

def redirect_button(url):
    button = st.button('Go to OHIF Viewer')
    if button:
        st.markdown(f'<meta http-equiv="refresh" content="0;url={url}" />', unsafe_allow_html=True)

###########################################################################################
########################### Streamlit Interface ###########################################
###########################################################################################

st.title("Image Enhancement and Quality Evaluation")

# uploaded_file = st.file_uploader("Upload Original Image", type=["png", "jpg", "jpeg"])
# enhancement_type = st.radio("Enhancement Type", ["Invert", "High Pass Filter", "Unsharp Masking", "Histogram Equalization", "CLAHE"])

st.sidebar.title("Configuration")
uploaded_file = st.sidebar.file_uploader("Upload Original Image", type=["png", "jpg", "jpeg", "dcm"])
enhancement_type = st.sidebar.selectbox(
    "Enhancement Type", 
    ["Invert", "High Pass Filter", "Unsharp Masking", "Histogram Equalization", "CLAHE"]
)

if uploaded_file is None:
    uploaded_file = st.file_uploader("Upload Original Image", type=["png", "jpg", "dcm"])
    enhancement_type = st.radio("Enhancement Type", ["Invert", "High Pass Filter", "Unsharp Masking", "Histogram Equalization", "CLAHE"])

# File uploader for DICOM files
if uploaded_file is not None:
    if hasattr(uploaded_file, 'name'):
        file_name = uploaded_file.name
        file_extension = uploaded_file.name.split(".")[-1]  # Get the file extension
        if file_extension.lower() == "dcm":
            # Process DICOM file
            dicom_data = pydicom.dcmread(uploaded_file)
            pixel_array = dicom_data.pixel_array
            # Process the pixel_array further if needed
            # Extract all metadata
            metadata = {elem.keyword: elem.value for elem in dicom_data if elem.keyword}
            metadata_dict = {str(key): str(value) for key, value in metadata.items()}
            df = pd.DataFrame.from_dict(metadata_dict, orient='index', columns=['Value'])

            # Display metadata in the left-most column
            with st.expander("Lihat Metadata"):
                st.write("Metadata:")
                st.dataframe(df)

            # Read the pixel data
            pixel_array = dicom_data.pixel_array
            img_array = pixel_array.astype(float)
            img_array = (np.maximum(img_array, 0) / img_array.max()) * 255.0  # Normalize to 0-255
            img_array = np.uint8(img_array)  # Convert to uint8
            img = Image.fromarray(img_array)

            col1, col2 = st.columns(2)
            # Check the number of dimensions of the image
            if img_array.ndim == 3:
                n_slices = img_array.shape[0]
                if n_slices > 1:
                    slice_ix = st.sidebar.slider('Slice', 0, n_slices - 1, int(n_slices / 2))
                    # Display the selected slice
                    st.image(img_array[slice_ix, :, :], caption=f"Slice {slice_ix}", use_column_width=True)
                else:
                    # If there's only one slice, just display it
                    st.image(img_array[0, :, :], caption="Single Slice Image", use_column_width=True)
            elif img_array.ndim == 2:
                # If the image is 2D, just display it
                with col1:
                    st.image(img_array, caption="Original Image", use_column_width=True)
            else:
                st.error("Unsupported image dimensions")

            original_image = img_array

            # Example: convert to grayscale if it's a color image
            if len(pixel_array.shape) > 2:
                pixel_array = pixel_array[:, :, 0]  # Take only the first channel
            # Perform image enhancement and evaluation on pixel_array
            enhanced_image, mse, psnr, maxerr, l2rat = process_image(pixel_array, enhancement_type)
        else:
            # Process regular image file
            original_image = np.array(keras.utils.load_img(uploaded_file, color_mode='rgb' if enhancement_type == "Invert" else 'grayscale'))
            # Perform image enhancement and evaluation on original_image
            enhanced_image, mse, psnr, maxerr, l2rat = process_image(original_image, enhancement_type)
            col1, col2 = st.columns(2)
            with col1:
                st.image(original_image, caption="Original Image", use_column_width=True)
        with col2:
            st.image(enhanced_image, caption='Enhanced Image', use_column_width=True)

        col1, col2 = st.columns(2)
        col3, col4 = st.columns(2)

        col1.metric("MSE", round(mse,3))
        col2.metric("PSNR", round(psnr,3))
        col3.metric("Maxerr", round(maxerr,3))
        col4.metric("L2Rat", round(l2rat,3))

        # Save enhanced image to a file
        enhanced_image_path = "enhanced_image.png"
        cv2.imwrite(enhanced_image_path, enhanced_image)

        # Save original image to a file
        original_image_path = "original_image.png"
        cv2.imwrite(original_image_path, original_image)

        if st.button("Send to OHIF"):
            upload_folder_images(original_image_path, enhanced_image_path, file_name)

    # Add the redirect button
    redirect_button("https://new-ohif-viewer-k7c3gdlxua-et.a.run.app/")


st.title("Grad-CAM Visualization")
st.write("Grad Cam visualization")

# uploaded_file = st.file_uploader("Upload Image for Grad-CAM", type=["png", "jpg", "jpeg"], key="gradcam")
if uploaded_file is not None:
    st.session_state["gradcam"] = uploaded_file
    df_file = st.file_uploader("Upload DataFrame for Mean/Std Calculation", type=["csv"])
    labels = st.text_area("Labels", placeholder="Enter labels separated by commas")
    model_path = st.text_input("Model Path", 'model/densenet.hdf5')
    pretrained_model_path = st.text_input("Pretrained Model Path", 'model/pretrained_model.h5')

    if df_file and labels and model_path and pretrained_model_path:
        df = pd.read_csv(df_file)
        labels = labels.split(',')
        model = load_model(model_path)
        pretrained_model = load_model(pretrained_model_path)
        gradcam_images = compute_gradcam(uploaded_file, pretrained_model, df, labels)

        for idx, (gradcam_image, label) in enumerate(gradcam_images):
            st.image(gradcam_image, caption=f'Grad-CAM {idx+1}: {label}', use_column_width=True)
            # Save gradcam image to a file
            gradcam_image_path = f"gradcam_image_{idx+1}.png"
            cv2.imwrite(gradcam_image_path, gradcam_image)
            # Create unique folder name
            folder_name = f"{uploaded_file.name}"

            # Create the folder in Cloud Storage
            bucket_result.blob(folder_name + '/').upload_from_string('', content_type='application/x-www-form-urlencoded')

            # Upload the gradcam image to Google Cloud Storage
            destination_blob_name = f"gradcam_images/{uploaded_file.name}_{idx+1}.png"
            upload_to_gcs(gradcam_image_path, destination_blob_name)
