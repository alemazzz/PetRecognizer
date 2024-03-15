import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from sklearn.model_selection import train_test_split
import cv2
import re
import sys

IMG_WIDTH = 30
IMG_HEIGHT = 30
EPOCHS = 50
NUM_CATEGORIES = 37
    
    
def load_data(data_dir):
    images, labels = list(), list()
    
    # Iterate for each file in the given directory and skip if the file is corrupted
    for file_name in os.listdir(os.path.join(data_dir)):
        i = int(float(file_name.replace('.ppm', '')))
        
        try:
            image = cv2.resize(cv2.imread(os.path.join(data_dir, file_name)), (IMG_WIDTH, IMG_HEIGHT))
        except:
            continue
        
        # Append image and relative labels
        images.append(image)
        labels.append(i)
    
    return images, labels

def get_model():
    model = keras.Sequential()
    
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)))
    model.add(MaxPooling2D((2, 2)))
    
    model.add(Conv2D(32, (2, 2), activation='relu'))
    model.add(MaxPooling2D((3, 3)))
    
    model.add(Flatten())
    
    model.add(Dense(128, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.3))
    
    model.add(Dense(NUM_CATEGORIES, activation='softmax'))
    
    model.summary()
    
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics='accuracy')
    
    return model

def main():
    # Check command-line arguments
    if len(sys.argv) not in [2, 3]:
        sys.exit('Usage: python main.py data_directory [model.h5]')
        
    # Load data     
    images, labels = load_data(sys.argv[1])
    
    # one-hot encoding for the labels
    labels = keras.utils.to_categorical(labels)
    
    # Split train and test data
    x_train, x_test, y_train, y_test = train_test_split(np.array(images), np.array(labels), test_size=0.4)
    
    # Create a neural network model
    model = get_model()
    
    # Fit the model
    model.fit(x_train, y_train, epochs=EPOCHS)
    
    # Evaluate the model
    model.evaluate(x_test, y_test, verbose=2)
    
    # Save the model if the user wants to
    if len(sys.argv) == 3:
        filename = sys.argv[2]
        model.save(filename)
        print(f'Model saved to {filename}.')
        
if __name__ == '__main__':
    main()