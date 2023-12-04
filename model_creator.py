import os
import numpy as np
from sklearn.model_selection import StratifiedKFold
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import ResNet50, DenseNet121, Xception
from keras import layers, models
from keras.models import save_model
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import pandas as pd

# Function to load images and labels from a directory
def load_data(directory):
    images = []
    labels = []
    #label_dict = {'blurry': 'blurry', 'opaque': 'opaque', 'vivid': 'vivid'}
    label_list = [label for label in os.listdir(directory) if os.path.isdir(os.path.join(directory, label))]
    print(label_list)
    for label in label_list:
        label_path = os.path.join(directory, label)
        for filename in os.listdir(label_path):
            img_path = os.path.join(label_path, filename)
            images.append(img_path)
            labels.append(label)

    return images, labels, label_list

random_state=30

className = "Clarity"

# Set the path to your dataset
data_dir = r'AI_Art_Critic/data/' + className

# Load the paths and labels
images, labels, label_list = load_data(data_dir)

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

# Set the number of folds
num_folds = 5
kfold = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=random_state)

# Define the hyperparameter search space
batch_size = [32, 64, 128]
epochs = [5, 10, 15]
space = {
    'dropout': hp.uniform('dropout', 0.1, 0.9),
    'lr': hp.loguniform('lr', np.log(0.000001), np.log(0.0001)),
    'batch_size': hp.choice('batch_size', batch_size),
    'epochs': hp.choice('epochs', epochs),
}

def create_model_from_params(params):
    dropout = params['dropout']
    lr = params['lr']
    batch_size = params['batch_size']
    epochs = params['epochs']

    train_generator = datagen.flow_from_dataframe(
        dataframe=df,
        x_col='image_path',
        y_col='label',
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical'
    )

    model = create_model(train_generator, dropout, lr, batch_size, epochs)
    return model

# Define the model with hyperparameters
def create_model(train_generator, dropout, lr, batch_size, epochs):
    base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    model = models.Sequential()
    model.add(base_model)
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dropout(dropout))
    model.add(layers.Dense(len(label_list), activation='softmax'))
    for layer in base_model.layers:
        layer.trainable = False
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(train_generator, epochs=epochs)
    return model


# Define the objective function for optimization
def objective(params):
    dropout = params['dropout']
    lr = params['lr']
    batch_size = params['batch_size']
    epochs = params['epochs']

    # List to store accuracy values for each fold
    accuracy_values = []

    # Perform k-fold cross-validation
    for train, test in kfold.split(df['image_path'], df['label']):
        train_df = df.iloc[train]
        test_df = df.iloc[test]

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
        model = create_model(train_generator, dropout, lr, batch_size, epochs)

        # Evaluate the model on the test set
        eval_result = model.evaluate(test_generator)
        accuracy_values.append(eval_result[1])

    # Calculate and return the average accuracy
    average_accuracy = np.mean(accuracy_values)
    print(f"Raw Accuracy: {average_accuracy}")

    return {'loss': -average_accuracy, 'status': STATUS_OK}

trials = Trials()
best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=5, trials=trials)

# Print the best hyperparameters
print("Best performing hyperparameters:", best)

model_save_path = 'AI_Art_Critic/' + className + '.keras'

bestParams = {
    'dropout': best['dropout'],
    'lr': best['lr'],
    'batch_size': batch_size[best['batch_size']],
    'epochs': epochs[best['epochs']],
}

print(bestParams)
model = create_model_from_params(bestParams)
model.save(model_save_path)
