import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import itertools
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

class GradCAM(object):
    
    def __init__(self, model, alpha=0.8, beta=0.3):
        
        self.model = model
        self.alpha = alpha
        self.beta = beta
 
    def apply_heatmap(self, heatmap, image): 
        # Resize the heatmap to match the size of the original image
        heatmap = cv2.resize(heatmap, image.shape[:-1]) 
        
        # Apply color map (JET) to the heatmap
        heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)   
        
        # Combine the original image and heatmap
        superimposed_img = cv2.addWeighted(np.array(image).astype(np.float32), self.alpha, 
                                           np.array(heatmap).astype(np.float32), self.beta, 0)
        
        return np.array(superimposed_img).astype(np.uint8)
    
    def gradCAM(self, x_test=None, name='black_max_pool_2', index_class=0):
        
        with tf.GradientTape() as tape:
             # Get the specified layer from the model
            last_conv_layer = self.model.get_layer(name) 
            
            # Create a new model that outputs the predicted class and the output of the specified layer
            grad_model = tf.keras.Model([self.model.input], [self.model.output, last_conv_layer.output])
              
            # Get the model predictions and the output of the specified layer for the input image
            model_out, last_conv_layer = grad_model(np.expand_dims(x_test, axis=0))   
            
            # Extract the predicted class output
            class_out = model_out[:, index_class]  
               
            # Compute the gradients of the predicted class output with respect to the last convolutional layer
            grads = tape.gradient(class_out, last_conv_layer)   
            
            # Compute the average gradient values across the spatial dimensions (8x8) and the channels (1152)
            pooled_grads = tf.reduce_mean(grads, axis=(0, 3)) # (1, 8, 8, 1152) -> (8, 8)
            
            # Compute the average activation values across the spatial dimensions (8x8) and the channels (1152)
            last_conv_layer = tf.reduce_mean(last_conv_layer, axis=(0, 3))
        
        # Element-wise multiplication of the gradients and the activation values
        heatmap = tf.multiply(pooled_grads, last_conv_layer)  
        
        # Set negative values to zero
        heatmap = np.maximum(heatmap, 0)
        
        # Normalize the heatmap values between 0 and 1
        heatmap /= np.max(heatmap)    
        heatmap = np.array(heatmap) 
        
        # Apply the heatmap on the input image and return the result
        return self.apply_heatmap(heatmap, x_test)  