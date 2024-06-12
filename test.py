import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import cv2
from tqdm import tqdm
import tensorflow as tf
from train import load_dataset, create_dir

H = 512
W = 512

def cal_iou(y_true, y_pred):
    x1 = max(y_true[0], y_pred[0])
    y1 = max(y_true[1], y_pred[1])
    x2 = min(y_true[2], y_pred[2])
    y2 = min(y_true[3], y_pred[3])

    intersection_area = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)
 
    true_area = (y_true[2] - y_true[0] + 1) * (y_true[3] - y_true[1] + 1)
    bbox_area = (y_pred[2] - y_pred[0] + 1) * (y_pred[3] - y_pred[1] + 1)

    iou = intersection_area / float(true_area + bbox_area - intersection_area)
    return iou

if __name__ == "__main__":
    np.random.seed(42)
    tf.random.set_seed(42)

    create_dir("object_detection_test_samples")

    model = tf.keras.models.load_model(os.path.join("files", "model.h5"))

    dataset_path = "."
    # (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_dataset(dataset_path)
    (train_x, train_y, train_labels), (valid_x, valid_y, valid_labels), (test_x, test_y, test_labels) = load_dataset(dataset_path)
    print(f"Train: {len(train_x)} - {len(train_y)}")
    print(f"Valid: {len(valid_x)} - {len(valid_y)}")
    print(f"Test : {len(test_x)} - {len(test_y)}")

    mean_iou = []
    # for image, true_bbox in tqdm(zip(test_x, test_y), total=len(test_x)):
    for image, true_bbox, label in tqdm(zip(test_x, test_y, test_labels), total=len(test_x)):
        name = image.split("/")[-1].split(".")[0]

        image = cv2.imread(image, cv2.IMREAD_COLOR)
        x = cv2.resize(image, (W, H))
        x = (x - 127.5) / 127.5
        x = np.expand_dims(x, axis=0)

        true_x1, true_y1, true_x2, true_y2 = true_bbox

        pred_bbox = model.predict(x, verbose=0)[0]

        pred_x1 = int(pred_bbox[0] * image.shape[1])
        pred_y1 = int(pred_bbox[1] * image.shape[0])
        pred_x2 = int(pred_bbox[2] * image.shape[1])
        pred_y2 = int(pred_bbox[3] * image.shape[0])

        iou = cal_iou(true_bbox, [pred_x1, pred_y1, pred_x2, pred_y2])
        mean_iou.append(iou)

        image = cv2.rectangle(image, (true_x1, true_y1), (true_x2, true_y2), (255, 0, 0), 5) ## BLUE
        image = cv2.rectangle(image, (pred_x1, pred_y1), (pred_x2, pred_y2), (0, 0, 255), 5) ## RED

        x = int(image.shape[1] * 0.05)
        y = int(image.shape[0] * 0.05)
        font_size = int(image.shape[0] * 0.001)
        cv2.putText(image, f"IoU: {iou:.4f}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_size, (255, 0, 0), 3)
        cv2.putText(image, f"Label: {label}", (x, y + 30), cv2.FONT_HERSHEY_SIMPLEX, font_size, (255, 255, 255), 3)

        cv2.imwrite(f"results/{name}.png", image)

    score = np.mean(mean_iou, axis=0)
    print(f"Mean IoU: {score:.4f}")



# import os
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
 
# import numpy as np
# import cv2
# from tqdm import tqdm
# import tensorflow as tf
# from train import load_dataset, create_dir

# H = 512
# W = 512

# def cal_iou(y_true, y_pred):
#     x1 = max(y_true[0], y_pred[0])
#     y1 = max(y_true[1], y_pred[1])
#     x2 = min(y_true[2], y_pred[2])
#     y2 = min(y_true[3], y_pred[3])

#     # intersection_area = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)
 
#     # true_area = (y_true[2] - y_true[0] + 1) * (y_true[3] - y_true[1] + 1)
#     # bbox_area = (y_pred[2] - y_pred[0] + 1) * (y_pred[3] - y_pred[1] + 1)

#     intersection_area = max(0, x2 - x1) * max(0, y2 - y1)
 
#     true_area = (y_true[2] - y_true[0]) * (y_true[3] - y_true[1])
#     bbox_area = (y_pred[2] - y_pred[0]) * (y_pred[3] - y_pred[1])
 
#     iou = intersection_area / float(true_area + bbox_area - intersection_area)
#     return iou

# if __name__ == "__main__":
#     """ Seeding """
#     np.random.seed(42)
#     tf.random.set_seed(42)

#     """ Directory for storing files """
#     create_dir("results")

#     """ Load the model """
#     model = tf.keras.models.load_model(os.path.join("files", "model.h5"))

#     dataset_path = '.'
#     (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_dataset(dataset_path)
#     print(f"Train: {len(train_x)} - {len(train_y)}")
#     print(f"Valid: {len(valid_x)} - {len(valid_y)}")
#     print(f"Test : {len(test_x)} - {len(test_y)}")

#     """ Prediction """
#     mean_iou = []
#     for image, true_bbox in tqdm(zip(test_x, test_y), total=len(test_y)):
#         """ Extract the name """
#         # print(image)
#         name = image.split("/")[-1].split(".")[0]

#         """ Reading the image """
#         image = cv2.imread(image, cv2.IMREAD_COLOR)
#         orig_h, orig_w, _ = image.shape
#         x = cv2.resize(image, (W, H))
#         x = (x - 127.5) / 127.5
#         x = np.expand_dims(x, axis=0)

#         """ Bounding box """
#         true_x1, true_y1, true_x2, true_y2 = true_bbox

#         """ Prediction """
#         pred_bbox = model.predict(x, verbose=0)[0]
#         # print(pred_bbox)

#         """ Rescale the bbox points """
#         # pred_x1 = int(pred_bbox[0] * image.shape[1])
#         # pred_y1 = int(pred_bbox[1] * image.shape[0])
#         # pred_x2 = int(pred_bbox[2] * image.shape[1])
#         # pred_y2 = int(pred_bbox[3] * image.shape[0])

#         pred_x1 = int(pred_bbox[0] * orig_w)
#         pred_y1 = int(pred_bbox[1] * orig_h)
#         pred_x2 = int(pred_bbox[2] * orig_w)
#         pred_y2 = int(pred_bbox[3] * orig_h)

#         """ True BBox GPT """
#         true_x1 = int(true_bbox[0] * orig_w)
#         true_y1 = int(true_bbox[1] * orig_h)
#         true_x2 = int(true_bbox[2] * orig_w)
#         true_y2 = int(true_bbox[3] * orig_h)

#         """ IoU """
#         # iou = cal_iou(true_bbox, [pred_x1, pred_y1, pred_x2, pred_y2])
#         # mean_iou.append(iou)
#         iou = cal_iou([true_x1, true_y1, true_x2, true_y2], [pred_x1, pred_y1, pred_x2, pred_y2])
#         mean_iou.append(iou)

#         """ Plot them on image """
#         # image = cv2.rectangle(image, (true_x1, true_y1), (true_x2, true_y2), (255, 0, 0), 10) ## BLUE
#         # image = cv2.rectangle(image, (pred_x1, pred_y1), (pred_x2, pred_y2), (0, 0, 255), 10) ## RED
#         image = cv2.rectangle(image, (true_x1, true_y1), (true_x2, true_y2), (255, 0, 0), 2) ## BLUE
#         image = cv2.rectangle(image, (pred_x1, pred_y1), (pred_x2, pred_y2), (0, 0, 255), 2) ## RED


#         # x = int(image.shape[1] * 0.05)
#         # y = int(image.shape[0] * 0.05)
#         # font_size = int(image.shape[0] * 0.001)
#         # cv2.putText(image, f"IoU: {iou:.4f}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_size, (255, 0, 0), 3)
#         x_text = int(image.shape[1] * 0.05)
#         y_text = int(image.shape[0] * 0.05)
#         font_size = int(image.shape[0] * 0.001)
#         cv2.putText(image, f"IoU: {iou:.4f}", (x_text, y_text), cv2.FONT_HERSHEY_SIMPLEX, font_size, (255, 0, 0), 2)

#         """ Saving the image """
#         cv2.imwrite(f"results/{name}.png", image)

#         # break

#     """ Mean IoU """
#     score = np.mean(mean_iou, axis=0)
#     print(f"Mean IoU: {score:.4f}")