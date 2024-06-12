from tensorflow.keras import layers as L
from tensorflow.keras.models import Model
from tensorflow.keras.applications import MobileNetV2
from glob import glob

def build_model(input_shape):
    """ Inputs """
    inputs = L.Input(input_shape)
 
    """ Backbone """
    backbone = MobileNetV2(
        include_top=False,
        weights="imagenet",
        input_tensor=inputs,
        alpha=1.0
    )
    backbone.trainable = True
    for layer in backbone.layers[:100]:
        layer.trainable = False
 
    # backbone.summary()
  
    """ Detection Head """
    # x = backbone.output
    # x = L.Conv2D(256, kernel_size=1, padding="same")(x)
    # x = L.BatchNormalization()(x)
    # x = L.Activation("relu")(x)
    # x = L.GlobalAveragePooling2D()(x)
    # x = L.Dropout(0.5)(x)
    # x = L.Dense(4, activation="sigmoid")(x)
    x = backbone.output
    x = L.Conv2D(256, kernel_size=3, padding="same")(x)
    x = L.BatchNormalization()(x)
    x = L.Activation("relu")(x)
    x = L.Conv2D(128, kernel_size=3, padding="same")(x)
    x = L.BatchNormalization()(x)
    x = L.Activation("relu")(x)
    x = L.GlobalAveragePooling2D()(x)
    x = L.Dropout(0.5)(x)
    x = L.Dense(4, activation="sigmoid")(x)
 
    """ Model """
    model = Model(inputs, x)
    return model

if __name__ == "__main__":
    input_shape = (512, 512, 3)
    model = build_model(input_shape)
    # model.save('files/model.h5')
    model.summary()










# from tensorflow.keras import layers as L
# from tensorflow.keras.models import Model
# from tensorflow.keras.applications import MobileNetV2
# from glob import glob

# def build_model(input_shape):
#     """ Inputs """
#     inputs = L.Input(input_shape)
 
#     """ Backbone """
#     backbone = MobileNetV2(
#         include_top=False,
#         weights="imagenet",
#         input_tensor=inputs,
#         alpha=1.0
#     )
#     backbone.trainable = False
#     # backbone.summary()
 
#     """ Detection Head """
#     x = backbone.output
#     x = L.Conv2D(64, kernel_size=1, padding="same")(x)
#     x = L.Activation("relu")(x)
#     x = L.Conv2D(32, kernel_size=1, padding="same")(x)
#     x = L.Dropout(0.5)(x)
#     x = L.GlobalMaxPooling2D()(x)
#     x = L.Dense(4, activation="sigmoid")(x)
 
#     """ Model """
#     model = Model(inputs, x)
#     return model

# if __name__ == "__main__":
#     input_shape = (512, 512, 3)
#     model = build_model(input_shape)
#     model.save('files/mamas_model.h5')
#     model.summary()