import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

dataset_path = r"C:\Users\S.Sanjaikumar\.cache\kagglehub\datasets\gti-upm\leapgestrecog\versions\1\leapGestRecog"
print("Dataset Path:", dataset_path)


subject_folder = os.path.join(dataset_path, "00") 
gesture_folder = os.path.join(subject_folder, "01_palm") 

images = os.listdir(gesture_folder)
print(f"Number of images in '{gesture_folder}':", len(images))

sample_image_path = os.path.join(gesture_folder, images[0])



img_size = 100 
#X for images and Y for lables
X = [] 
y = []  

# Map gesture names to numeric labels
gesture_labels = {
    '01_palm': 0,
    '02_l': 1,
    '03_fist': 2,
    '04_fist_moved': 3,
    '05_thumb': 4,
    '06_index': 5,
    '07_ok': 6,
    '08_palm_moved': 7,
    '09_c': 8,
    '10_down': 9
}

# Loop through all subjects and gestures
for subject in os.listdir(dataset_path):
    subject_path = os.path.join(dataset_path, subject)
    
    # Skip any unexpected folders
    if not os.path.isdir(subject_path):
        continue

    for gesture in os.listdir(subject_path):
        gesture_path = os.path.join(subject_path, gesture)

        # Skip non-gesture folders
        if gesture not in gesture_labels:
            continue

        # Process each image
        for img_name in os.listdir(gesture_path):
            img_path = os.path.join(gesture_path, img_name)

            # Read, resize, normalize
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (img_size, img_size)) / 255.0

            # Append to dataset
            X.append(img)
            y.append(gesture_labels[gesture])

# Convert to numpy arrays
X = np.array(X).reshape(-1, img_size, img_size, 1)  # Add channel dimension
y = np.array(y)

# Print data shape
print("Dataset loaded:")
print("X shape:", X.shape)
print("y shape:", y.shape)

from sklearn.model_selection import train_test_split

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Print the shapes of the splits
#print("Training set shape:", X_train.shape, y_train.shape)
#print("Testing set shape:", X_test.shape, y_test.shape)



# Build the CNN model
model = keras.Sequential([
    # Convolutional layers
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 1)),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    # Flatten and Dense layers
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),  # Helps prevent overfitting
    layers.Dense(10, activation='softmax')  # 10 classes for 10 gestures
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

#model.summary()

history = model.fit(X_train, y_train, validation_split=0.2, epochs=20, batch_size=32)

test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {test_accuracy:.4f}, Test Loss: {test_loss:.4f}')

model.save('hand_gesture_model.h5')
print("Model saved as 'hand_gesture_model.h5'")

# Accuracy plot
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training and Validation Accuracy')
plt.savefig('accuracy_plot.png')
print("Accuracy plot saved as 'accuracy_plot.png'")
plt.show()

# Loss plot
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')
plt.savefig('loss_plot.png')
print("Loss plot saved as 'loss_plot.png'")
plt.show()





