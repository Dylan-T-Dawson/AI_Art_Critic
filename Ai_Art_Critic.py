import os
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import ResNet50
from keras import layers, models
import pandas as pd

# Set the path to your dataset
data_dir = r'C:\Users\dtda230\Desktop\AI_Art_Critic\data\Clarity'

# Function to load images and labels from a directory
def load_data(directory):
    images = []
    labels = []
    label_dict = {'blurry': 'blurry', 'opaque': 'opaque', 'vivid': 'vivid'}

    for label in label_dict.keys():
        label_path = os.path.join(directory, label)
        for filename in os.listdir(label_path):
            img_path = os.path.join(label_path, filename)
            images.append(img_path)
            labels.append(label_dict[label])

    return images, labels

# Load the paths and labels
images, labels = load_data(data_dir)

# Split the data into training and testing sets
train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=0.2, stratify=labels, random_state=42)

# Create a DataFrame from the paths and labels
train_df = pd.DataFrame({'image_path': train_images, 'label': train_labels})
test_df = pd.DataFrame({'image_path': test_images, 'label': test_labels})

# Create a data generator for training
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

# Create a data generator for testing
test_datagen = ImageDataGenerator(rescale=1./255)

# Load and prepare the training data
train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    x_col='image_path',
    y_col='label',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

# Load and prepare the testing data
test_generator = test_datagen.flow_from_dataframe(
    dataframe=test_df,
    x_col='image_path',
    y_col='label',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

# Load the pre-trained ResNet model without the top (fully connected) layers
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Create a new model on top
model = models.Sequential()
model.add(base_model)
model.add(layers.GlobalAveragePooling2D())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(3, activation='softmax'))  # 3 classes: opaque, blurry, vivid

# Freeze the pre-trained layers
for layer in base_model.layers:
    layer.trainable = False

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_generator, epochs=10)

# Evaluate the model on the test set
eval_result = model.evaluate(test_generator)
print("Test Accuracy:", eval_result[1])