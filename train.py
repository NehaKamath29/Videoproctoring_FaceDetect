import numpy as np
import cv2
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam
from keras.utils import to_categorical
from livenessnet import LivenessNet  # Assuming you have defined this class somewhere
import os
import pickle

# Define script parameters
dataset_path = "liveness_output_img"
model_path = "liveness_model.h5"
label_encoder_path = "le.pickle"
batch_size = 32
epochs = 50
learning_rate = 1e-4

# Function to load and preprocess dataset
def load_dataset(dataset_path):
    data = []
    labels = []
    
    # Iterate through both "Live" and "Non_Live" folders
    for folder in ["live", "non_live"]:
        folder_path = os.path.join(dataset_path, folder)
        
        # Walk through all the files in the current folder
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if file.endswith(".jpeg") or file.endswith(".png"):
                    image_path = os.path.join(root, file)
                    
                    # Assign appropriate label based on the folder name
                    label = folder.lower()
                    #print(label)
                    image = cv2.imread(image_path)
                    image = cv2.resize(image, (32, 32))
                    data.append(image)
                    labels.append(label)
    
    data = np.array(data, dtype="float") / 255.0
    labels = np.array(labels)
    return data, labels

# Load and preprocess dataset
data, labels = load_dataset(dataset_path)

# Encode labels
le = LabelEncoder()
labels = le.fit_transform(labels)
num_classes = len(le.classes_)
labels = to_categorical(labels, num_classes)

# Split dataset into training and testing sets
trainX, testX, trainY, testY = train_test_split(data, labels, test_size=0.25, random_state=42)

# Build and compile the model
model = LivenessNet.build(width=32, height=32, depth=3, classes=num_classes)
opt = Adam(learning_rate=learning_rate)
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

# Train the model
H = model.fit(trainX, trainY, validation_data=(testX, testY),
              batch_size=batch_size, epochs=epochs)

# Evaluate the model
_, accuracy = model.evaluate(testX, testY)
print("Accuracy: {:.2f}%".format(accuracy * 100))

# Save the trained model
model.save(model_path)
print("Model saved at:", model_path)

# Save the label encoder
with open(label_encoder_path, "wb") as le_file:
    le_file.write(pickle.dumps(le))
print("Label Encoder saved at:", label_encoder_path)



# Print actual image paths and labels
# for i in range(len(data)):
#     print(i)
#     folder_index = np.where(labels[i] == 1)[0][0]  # Assuming 1 corresponds to "Non_Live" and 0 to "Live"
#     folder_name = "Non_Live" if folder_index == 1 else "Live"
#     file_name = f"img{i+1}.jpeg" if folder_index == 1 else f"live_img{i+1}.jpeg"
#     image_path = os.path.join(dataset_path, folder_name, file_name)
#     print(f"Image Path: {image_path}, Label: {labels[i]}")
