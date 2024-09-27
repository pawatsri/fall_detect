import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import torch

# Load validation image and label files
val_image_dir = 'Fall_detect/fall_dataset/images/val'
train_img_files = os.listdir(val_image_dir)
train_img_files.sort()

# Assuming labels are also stored in the same directory, adjust as necessary
train_label_files = os.listdir(val_image_dir)
train_label_files.sort()

# Debugging: Check the number of images and labels
print("Number of validation images:", len(train_img_files))
print("Number of validation labels:", len(train_label_files))

# Prepare lists for images and classes
complete_images = []
complete_class = []

# Load images and bounding boxes from labels
for i in range(len(train_img_files)):
    img_path = os.path.join(val_image_dir, train_img_files[i])
    img = plt.imread(img_path)
    
    with open(os.path.join(val_image_dir, train_label_files[i]), 'r') as file:
        r = file.readlines()
    
    bounding_boxes = []
    for j in r:
        j = j.split()
        bounding_boxes.append([int(j[0]), float(j[1]), float(j[2]), float(j[3]), float(j[4])])  # Class ID, x_center, y_center, width, height
    
    # Convert bounding box coordinates
    for box in bounding_boxes:
        image_height, image_width, _ = img.shape
        xmin, ymin, width, height = box[1:]
        
        xmin = int((box[1] - box[3] / 2) * image_width)  # Convert center x to xmin
        ymin = int((box[2] - box[4] / 2) * image_height)  # Convert center y to ymin
        width = int(width * image_width)
        height = int(height * image_height)
        
        # Append class and cropped image
        complete_class.append(box[0])
        complete_images.append(img[max(0, ymin):min(image_height, ymin + height), max(0, xmin):min(image_width, xmin + width)])

# Resize images to a preferred size
pref_size = (128, 128)
for i in range(len(complete_images)):
    complete_images[i] = cv2.resize(complete_images[i], pref_size)

# Display an example image
img = complete_images[86]
plt.imshow(img)
plt.axis('off')  # Hide axes
plt.show()

# Create a DataFrame for evaluation
df = pd.DataFrame()
df['Images'] = complete_images
df['Class'] = complete_class
df['Images'] /= 255  # Normalize pixel values to [0, 1]

# Prepare test data for model evaluation
X_test = np.array(df['Images'].tolist())
y_test = np.array(df['Class'])

# Evaluate the trained model (assuming 'model' is already defined and trained)
# Ensure your model is loaded before this line
model.evaluate(X_test, y_test)

# Load YOLOv5 model for object detection
yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
yolo_model.classes = [0]  # Set classes to detect, adjust as necessary

# Optional: Test YOLOv5 on a sample image
test_image = cv2.cvtColor(X_test[0], cv2.COLOR_RGB2BGR)  # Convert back to BGR for OpenCV
results = yolo_model(test_image)

# Visualize YOLOv5 results
results.show()
