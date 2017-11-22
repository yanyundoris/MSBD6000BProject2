from keras.models import load_model
import numpy as np
from ImagePreprocess import LoadImage2Array
import keras

def Load_trained_model(model_path = ""):
    model = load_model(model_path)
    return model

def Get_prediction(predict_data, model, save_file = ""):
    prediction_model = model.predict(predict_data)
    np.savetxt(save_file, prediction_model)

