import os
import numpy as np
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing import image

def predictCovid19(dataPath, modelName=None):

    if modelName == None or modelName == "resnet50":
        modelJsonPath = "/Models/model_resnet50.json"
        modelWeightPath = "/Models/model_resnet_best_weight.h5"

    if modelName == "densenet121":
        modelJsonPath = "/Models/model_densenet121.json"
        modelWeightPath = "/Models/model_densenet_best_weight.h5"

    if modelName == "mobilenet":
        modelJsonPath = "/Models/model_mobilenet.json"
        modelWeightPath = "/Models/model_mobilenet_best_weight.h5"

    oriImg = image.load_img(dataPath, target_size=(224,224))
    # normalize
    img = image.img_to_array(oriImg) / 255.0
    img = img.reshape((1,) + img.shape)

    # load pre-train model(json & weights)
    with open(os.path.join(os.getcwd(), "app") + modelJsonPath, "r") as f:
        json = f.read()
    
    model = model_from_json(json)   
    model.load_weights(os.path.join(os.getcwd(), "app") + modelWeightPath)

    # predict
    result = model.predict(img)
    pred = np.argmax(result, axis=1)
    
    trans_dict = {
        0:"COVID-19 Negative",
        1:"COVID-19 Positive"
    }
    
    pred = [trans_dict[j] for j in list(pred)]

    return pred