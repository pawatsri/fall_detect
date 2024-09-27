import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import torch
from PIL import Image
import time
from tensorflow.keras.applications import VGG16
from tensorflow.keras import layers, models

# Set the image and label directory paths
train_img_files = os.listdir('Fall_detect/fall_dataset/images/train')
train_img_files.sort()
r1 = 'Fall_detect/fall_dataset/images/train/'

train_label_files = os.listdir('Fall_detect/fall_dataset/labels/train')
train_label_files.sort()
r2 = 'Fall_detect/fall_dataset/labels/train/'

# Print the number of files in each directory
print(f"Number of images: {len(train_img_files)}")
print(f"Number of labels: {len(train_label_files)}")

complete_images = []
complete_class = []

# Load images and their bounding boxes
for i in range(len(train_img_files)):
    img = plt.imread(r1 + train_img_files[i])
    with open(r2 + train_label_files[i], 'r') as file:
        r = file.readlines()
    
    bounding_boxes = []
    for j in r:
        j = j.split()
        bounding_boxes.append([int(j[0]), float(j[1]), float(j[2]), float(j[3]), float(j[4])])
    
    for box in bounding_boxes:
        image_height, image_width, _ = img.shape
        xmin, ymin, width, height = box[1:]
        xmin = int(xmin * image_width)
        ymin = int(ymin * image_height)
        width = int(width * image_width)
        height = int(height * image_height)
        complete_class.append(box[0])
        complete_images.append(img[ymin - height//2 : ymin + height//2, xmin - width//2 : xmin + width//2])

# Resize images
pref_size = (128, 128)
for i in range(len(complete_images)):
    complete_images[i] = cv2.resize(complete_images[i], pref_size)

# Show an example image
img = complete_images[86]
plt.imshow(img)
plt.show()

# Normalize images and create a dataframe
df = pd.DataFrame()
df['Images'] = complete_images
df['Class'] = complete_class
df['Images'] = df['Images'] / 255.0

# Define the VGG16 model
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(128, 128, 3))

# Freeze the layers of the base model
for layer in base_model.layers:
    layer.trainable = False

# Create the classifier model
model = models.Sequential([
    base_model,
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(3, activation='softmax')  # 3 classes for fall detection
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Prepare the training data
X_train = np.array(df['Images'].tolist())
y_train = np.array(df['Class'])

# Train the model
model.fit(X_train, y_train, epochs=20)

# Save the model
model.save('Fall_detect/my_model.keras')

# YOLO model setup for bounding box detection
yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
yolo_model.classes = [0]  # Set YOLO class to person

# Modified get_boxes to handle both image paths and frames
def get_boxes(image):
    if isinstance(image, np.ndarray):
        # If the image is a numpy array (video frame), convert it to a PIL image
        img = Image.fromarray(image)
    else:
        # If it's a file path, load the image
        img = Image.open(image)

    image_width, image_height = img.size
    results = yolo_model(img, size=640)
    
    bounding_boxes = []
    for i, detection in enumerate(results.pandas().xyxy[0].values):  
        x_min, y_min, x_max, y_max, confidence, class_id, *extra_values = detection.tolist()
        if confidence > 0.5:
            x_center = (x_min + x_max) / 2 / image_width
            y_center = (y_min + y_max) / 2 / image_height
            norm_width = (x_max - x_min) / image_width
            norm_height = (y_max - y_min) / image_height
            bounding_boxes.append([x_center, y_center, norm_width, norm_height])
    return bounding_boxes

# Modified pred to accept frames directly
def pred(image):
    # Use get_boxes() to detect objects in the frame
    bounding_boxes = get_boxes(image)
    
    complete_images = []
    image_height, image_width, _ = image.shape
    
    # Crop bounding boxes from the image
    for box in bounding_boxes:
        xmin, ymin, width, height = box[:]
        xmin = int(xmin * image_width)
        ymin = int(ymin * image_height)
        width = int(width * image_width)
        height = int(height * image_height)
        cropped_img = image[ymin-height//2:ymin+height//2, xmin-width//2:xmin+width//2]
        complete_images.append(cropped_img)
    
    # Run fall detection on each cropped image
    for cropped_img in complete_images:
        cropped_img_resized = cv2.resize(cropped_img, pref_size)
        cropped_img_resized = cropped_img_resized / 255.0
        cropped_img_resized = np.expand_dims(cropped_img_resized, axis=0)
        predictions = model.predict(cropped_img_resized)
        k = np.argmax(predictions)
        
        if k == 1:
            print("Fall detected")
            return 1  # Return 1 for fall detected
        elif k == 0:
            print("No fall detected. Person is walking or standing")
        else:
            print("No fall detected. Person is sitting")
    
    return 1  # Return 1 for no fall detected

# Real-time fall detection with webcam
cap = cv2.VideoCapture(0)

# Initialize variables
snapshot_interval = 2.5  # Time interval between snapshots in seconds
last_snapshot_time = time.time()

while True:
    ret, frame = cap.read()

    if not ret:
        print("Error: Unable to capture frame")
        break

    cv2.imshow('Webcam', frame)

    current_time = time.time()
    if current_time - last_snapshot_time >= snapshot_interval:
        # Perform fall detection on the current frame
        fall_detected = pred(frame)

        # Print the result
        if fall_detected == 1:
            print("Fall detected!")
        else:
            print("No fall detected")

        last_snapshot_time = current_time

    # Press 'q' to exit the video stream
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()