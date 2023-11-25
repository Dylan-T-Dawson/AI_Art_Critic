import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import ResNet50, DenseNet121, Xception
from keras import layers, models
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import pandas as pd

# Set the path to your dataset
data_dir = r'/content/data'

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

# Convert labels to numpy array
labels = np.array(labels)

# Create a DataFrame from the paths and labels
df = pd.DataFrame({'image_path': images, 'label': labels})

# Create a data generator
datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

# Define the model with hyperparameters
def create_model(train_generator, test_generator, dropout, lr, batch_size, dense_units, epochs, optimizer):
    base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    model = models.Sequential()
    model.add(base_model)
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dense(dense_units, activation='relu'))
    model.add(layers.Dropout(dropout))
    model.add(layers.Dense(3, activation='softmax'))
    for layer in base_model.layers:
        layer.trainable = False
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(train_generator, epochs=epochs)
    return model

# Define the objective function for optimization
def objective(params):
    dropout = params['dropout']
    lr = params['lr']
    batch_size = params['batch_size']
    dense_units = params['dense_units']
    epochs = params['epochs']
    optimizer = params['optimizer']

    # Split the data into train and test sets
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    train_generator = datagen.flow_from_dataframe(
        dataframe=train_df,
        x_col='image_path',
        y_col='label',
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical'
    )

    test_generator = datagen.flow_from_dataframe(
        dataframe=test_df,
        x_col='image_path',
        y_col='label',
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical'
    )

    # Train the model
    model = create_model(train_generator, test_generator, dropout, lr, batch_size, dense_units, epochs, optimizer)

    # Evaluate the model on the test set
    eval_result = model.evaluate(test_generator)
    accuracy = eval_result[1]
    print(f"Raw Accuracy: {accuracy}")

    return {'loss': -accuracy, 'status': STATUS_OK}

# Option to either tune hyperparameters or enter your own
tune_hyperparameters = False

if tune_hyperparameters:
    # Define the hyperparameter search space
    space = {
        'dropout': hp.uniform('dropout', 0.2, 0.6),
        'lr': hp.loguniform('lr', np.log(0.000001), np.log(0.0001)),
        'batch_size': hp.choice('batch_size', [32, 64, 128]),
        'dense_units': hp.choice('dense_units', [256]),
        'epochs': hp.choice('epochs', [10, 15]),
        'optimizer': hp.choice('optimizer', ['adam']),
    }

    # Perform Bayesian optimization manually
    trials = Trials()
    best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=5, trials=trials)

    # Print the best hyperparameters
    print("Best performing hyperparameters:")
    print(best)
else:
    # Enter your own hyperparameters
    manual_hyperparameters = {
        'dropout': 0.75,
        'lr': 0.001,
        'batch_size': 32,
        'dense_units': 256,
        'epochs': 15,
        'optimizer': 'adam'
    }  # Change these values as needed

    # Evaluate the model with manual hyperparameters
    manual_result = objective(manual_hyperparameters)

    # Print the result with manual hyperparameters
    print("Result with manual hyperparameters:")
    print(manual_result)
