from tensorflow import keras
from keras.preprocessing import image
from keras.models import load_model
from keras import backend as K
from keras import models
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.applications import DenseNet121
from keras.applications.densenet import preprocess_input, decode_predictions
import numpy as np
import os 


perspectiveLabels = ['Deep', 'Shallow']
clarityLabels = ["Vivid", "Blurry", "Opaque"]
colorLabels = ['Neon', 'Pastel', 'Black and White']
detailLabels = ['Detailed', 'Simple']
styleLabels = ['Expressionism', 'Surrealism', 'Renaissance', 'Romanticism', 'Impressionism', 'Baroque', 'Art Nouveau', 'Post Impressionism', 'Realism']


modelMap = {
    "Detail": {'dropout': 0.5624932368766788, 'labels': detailLabels}, 
    "Clarity": {'dropout': 0.4254156052828808, 'labels': clarityLabels}, 
    "Perspective": {'dropout': 0.672488921445092, 'labels': perspectiveLabels}, 
    "Style": {'dropout': 0.6336432510546972, 'labels': styleLabels}, 
    "Color": {'dropout': 0.6693562156687544, 'labels': colorLabels}
}

compiledModels = {}

#Errors were thrown when importing the model and the denseNet weights, so a work-around was to import it layer by layer.
def create_model(classes, modelMapValue, path):
    global compiledModels

    # Check if the model has already been compiled
    if path in compiledModels:
        return compiledModels[path]

    base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    model = models.Sequential()
    model.add(base_model)
    model.add(GlobalAveragePooling2D())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(modelMapValue['dropout']))
    model.add(Dense(classes, activation='softmax'))
    
    for layer in base_model.layers:
        layer.trainable = False

    for layer in model.layers:
        try:
            if isinstance(layer, keras.layers.Layer):
                layer.load_weights(path, by_name=True)
        except Exception as e:
            pass

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Save the compiled model in the global map
    compiledModels[path] = model

    return model


def load_and_classify(img):
    returnString = ""
    returnMap = {}
    try:
        responseMap = {}
        for key, value in modelMap.items():
            
            returnMap[key] = []

            modelPath = "/model_path/" + key + ".keras"

            # Get the directory of the current script
            script_directory = os.path.dirname(os.path.realpath(__file__))

            # Construct the full path to the model file
            modelPath = os.path.join(script_directory, modelPath)

            # Load the Keras model
            model = create_model(len(value['labels']), value, modelPath)


            img = img.resize((224, 224))  # Ensure input size is correct
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array /= 255.0  # Normalize the image data

            # Make a prediction
            predictions = model.predict(img_array)

            for i in range(len(predictions[0])):
                print(" " + str(value['labels'][i]) +  " : " +  str(predictions[0][i]) + "\n")
                returnMap[key].append(str(predictions[0][i]))
          
        return createResponse(returnMap)

    except Exception as e:
        return f"Please input a valid .png file." + str(e)
    
#Crafts a response based on the probabilities.
def createResponse(returnMap):
    try:
        style_scores = returnMap['Style']
        max_style_index = np.argmax(style_scores)
        max_style_label = styleLabels[max_style_index]
        style_return_string = f"The image seems to be in the interesting style of {max_style_label}. "

        clarity_scores = returnMap['Clarity']
        max_clarity_index = np.argmax(clarity_scores)
        max_clarity_label = clarityLabels[max_clarity_index]
        clarity_return_string = f"The image appears {max_clarity_label}, "

        perspective_scores = returnMap['Perspective']
        max_perspective_index = np.argmax(perspective_scores)
        max_perspective_label = perspectiveLabels[max_perspective_index]
        perspective_return_string = f"and exhibits a {max_perspective_label} perspective. "

        detail_scores = returnMap['Detail']
        max_detail_index = np.argmax(detail_scores)
        max_detail_label = detailLabels[max_detail_index]
        detail_return_string = f"The {max_detail_label} image uses a"

        color_scores = returnMap['Color']
        max_color_index = np.argmax(color_scores)
        max_color_label = colorLabels[max_color_index]
        color_return_string = f" {max_color_label} color palette."

        return style_return_string + clarity_return_string + perspective_return_string + detail_return_string + color_return_string
    except Exception as e:
        return f"An error occurred: {e}"