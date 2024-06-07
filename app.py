import cv2
import gradio as gr
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.preprocessing import image
from keras.models import load_model

# Load the pre-trained models
model_path = 'model/densenet.hdf5'
pretrained_model_path = 'model/pretrained_model.h5'
model = load_model(model_path)
pretrained_model = load_model(pretrained_model_path)

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

def get_mean_std_per_batch(image_path, df, H=320, W=320):
    sample_data = []
    for idx, img in enumerate(df.sample(100)["Image Index"].values):
        sample_data.append(
            np.array(image.load_img(image_path, target_size=(H, W))))
    mean = np.mean(sample_data[0])
    std = np.std(sample_data[0])
    return mean, std

def load_image(img_path, df, preprocess=True, H=320, W=320):
    mean, std = get_mean_std_per_batch(img_path, df, H=H, W=W)
    x = image.load_img(img_path, target_size=(H, W))
    x = image.img_to_array(x)
    if preprocess:
        x -= mean
        x /= std
        x = np.expand_dims(x, axis=0)
    return x

def grad_cam(input_model, img_array, cls, layer_name):
    grad_model = tf.keras.models.Model(
        [input_model.inputs],
        [input_model.get_layer(layer_name).output, input_model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
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

def compute_gradcam(img, model, df, labels, layer_name='bn'):
    preprocessed_input = load_image(img, df)
    predictions = model.predict(preprocessed_input)

    top_indices = np.argsort(predictions[0])[-3:][::-1]
    top_labels = [labels[i] for i in top_indices]
    top_predictions = [predictions[0][i] for i in top_indices]

    plt.figure(figsize=(20, 20))
    plt.subplot(4, 4, 1)
    plt.title("Original Image")
    plt.axis('off')
    original_image = load_image(img, df, preprocess=False)
    plt.imshow(original_image, cmap='gray')

    for i in range(3):
        idx = top_indices[i]
        label = top_labels[i]
        prob = top_predictions[i]

        gradcam = grad_cam(model, preprocessed_input, idx, layer_name)

        plt.subplot(4, 4, i+2)
        plt.title(f"{label}: p={prob:.3f}")
        plt.axis('off')
        plt.imshow(original_image, cmap='gray')
        plt.imshow(gradcam, cmap='jet', alpha=0.5)

    plt.show()

def gradio_interface_gradcam(img, df, labels, model):
    labels = labels.split(',')
    compute_gradcam(img, model, df, labels)
    return img

iface = gr.Interface(
    fn=process_image,
    inputs=[
        gr.Image(type="numpy", label="Upload Original Image"),
        gr.Radio(choices=["Invert", "High Pass Filter", "Unsharp Masking", "Histogram Equalization", "CLAHE"], label="Enhancement Type"),
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

iface_gradcam = gr.Interface(
    fn=gradio_interface_gradcam,
    inputs=[
        gr.Image(type="numpy", label="Upload Image for Grad-CAM"),
        gr.Dataframe(label="Dataframe for Mean/Std Calculation"),
        gr.Textbox(label="Labels", lines=5, placeholder="Enter labels separated by commas"),
        gr.Model(label="Model")
    ],
    outputs=[
        gr.Plot(label="Grad-CAM Visualization")
    ],
    title="Grad-CAM Visualization"
)

# Combine interfaces
iface_combined = gr.TabbedInterface([iface, iface_gradcam], ["Image Enhancement and Quality Evaluation", "Grad-CAM Visualization"])

iface_combined.launch()
