import streamlit as st
import numpy as np
import pandas as pd
import os
import cv2
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import to_categorical

# Function for image preprocessing
def preprocess_image(image_path, target_size=(64, 64)):
    img = cv2.imread(image_path)
    print("img_path:", image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    img = cv2.resize(img, target_size)
    img = img / 255.0  # Normalize pixel values to [0, 1]
    return img

# Load your dataset from folders
@st.cache
@st.cache(allow_output_mutation=True)
def load_dataset(data_dir):
    data = []    # List to store image data
    labels = []  # List to store corresponding labels

    classes = os.listdir(data_dir)

    for class_name in classes:
        class_path = os.path.join(data_dir, class_name)
        for item_name in os.listdir(class_path):
            item_path = os.path.join(class_path, item_name)
            img = preprocess_image(item_path, target_size=(64, 64))
            data.append(img)
            labels.append(class_name)

    data = np.array(data)
    labels = np.array(labels)

    return data, labels

@st.cache_data(allow_output_mutation=True)
def load_cached_dataset(data_dir):
    return load_dataset(data_dir)

@st.cache_data(allow_output_mutation=True)
def split_cached_dataset(data):
    return split_dataset(data)


# Split dataset into training and testing sets
@st.cache_data(allow_output_mutation=True)
def split_dataset(data):
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
    return train_data, test_data


# Build a simple CNN model (you can replace this with your custom XceptionNet)
def build_model(input_shape, num_classes):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Train the model
def train_model(train_data, target_size, num_classes, epochs=10):
    for img_path in train_data:
        print("Processing image:", img_path)
        X_train = np.array([preprocess_image(img_path, target_size)])
        print("X_train shape:", X_train.shape)

        # Add the rest of your code...
        y_train = to_categorical(train_data['label'], num_classes=num_classes)

        input_shape = (target_size[0], target_size[1], 3)
        model = build_model(input_shape, num_classes)
        model.fit(X_train, y_train, epochs=epochs)
        return model

# Evaluate the model
def evaluate_model(model, test_data, target_size, num_classes):
    X_test = np.array([preprocess_image(img_path, target_size) for img_path in test_data['image_path']])
    y_test = to_categorical(test_data['label'], num_classes=num_classes)

    predictions = model.predict(X_test)
    y_pred = np.argmax(predictions, axis=1)

    accuracy = accuracy_score(test_data['label'], y_pred)
    return accuracy

# Main Streamlit app
def main():
    st.title("Oral Cancer Detection Web App")

    # Load dataset from folders
    data_dir = 'C:\\Users\\Huawei\\OneDrive\\Documents\\dentaldesease\\oralcancer\\uploads'
    data = load_dataset(data_dir)

    # Split dataset
    train_data, test_data = split_dataset(data)

    # Display some sample data
    st.write("Sample Data:")
    if isinstance(train_data, tuple):
        st.write("Train Data:")
        st.write(train_data[0][:5], train_data[1][:5])
        st.write("Test Data:")
        st.write(test_data[0][:5], test_data[1][:5])
    else:
        st.write(data[:5])

    # Training the model
    st.write("Training the model...")
    model = train_model(train_data, target_size=(64, 64), num_classes=len(np.unique(data[1])), epochs=5)
    st.write("Training complete!")

    # Evaluate the model
    st.write("Evaluating the model...")
    accuracy = evaluate_model(model, test_data, target_size=(64, 64), num_classes=len(np.unique(data[1])))
    st.write(f"Accuracy: {accuracy * 100:.2f}%")

if __name__ == '__main__':
    main()
