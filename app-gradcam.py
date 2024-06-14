import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from tensorflow.python.keras.models import load_model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

class GradCAM(object):
    
    def __init__(self, model, alpha=0.8, beta=0.3):
        self.model = model
        self.alpha = alpha
        self.beta = beta

    def apply_heatmap(self, heatmap, image): 
        heatmap = cv2.resize(heatmap, image.shape[:-1]) 
        heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)   
        superimposed_img = cv2.addWeighted(np.array(image).astype(np.float32), self.alpha, 
                                           np.array(heatmap).astype(np.float32), self.beta, 0)
        return np.array(superimposed_img).astype(np.uint8)
    
    def gradCAM(self, x_test=None, name='block5_conv3', index_class=0):
        with tf.GradientTape() as tape:
            last_conv_layer = self.model.get_layer(name) 
            grad_model = tf.keras.Model([self.model.input], [self.model.output, last_conv_layer.output])
            model_out, last_conv_layer = grad_model(np.expand_dims(x_test, axis=0))   
            class_out = model_out[:, index_class]  
            grads = tape.gradient(class_out, last_conv_layer)   
            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
            last_conv_layer = last_conv_layer[0]
            heatmap = last_conv_layer @ pooled_grads[..., tf.newaxis]
            heatmap = tf.squeeze(heatmap)
        heatmap = np.maximum(heatmap, 0)
        heatmap /= np.max(heatmap)    
        heatmap = np.array(heatmap) 
        return self.apply_heatmap(heatmap, x_test)

# Streamlit app
st.title("Grad-CAM Visualization")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Load the uploaded image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        st.image(img, caption='Uploaded Image.', use_column_width=True)

        # Preprocess the image for the model (assuming the model expects 224x224 images)
        img_resized = cv2.resize(img, (224, 224))
        img_array = np.expand_dims(img_resized, axis=0)

        # Load the model
        model_path = 'model_renamed.h5'  # Update this path to your model's path
        model = tf.keras.models.load_model(model_path)

        # Initialize GradCAM
        grad_cam = GradCAM(model)

        # Compute GradCAM heatmap
        heatmap_img = grad_cam.gradCAM(img_array[0])

        # Display the GradCAM heatmap
        st.image(heatmap_img, caption='Grad-CAM Heatmap.', use_column_width=True)

    except Exception as e:
        st.error(f"Error: {e}")