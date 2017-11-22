from keras.models import load_model
import numpy as np
from PreprocessingStage import LoadTestImage2Array, LoadImage2Array
from sklearn.metrics import classification_report
import keras


num_classes = 5

model = load_model('keras_trained_model_batchnormal.h5')


# You need to change data dir
test_data = LoadTestImage2Array('test.txt',
                                '/Users/yanyunliu/PycharmProjects/TensorFlowTutorial/data',
                                image_size = 32, use_gray = False, use_keras = True)

test_data = test_data.astype('float32')
test_data /= 255

prediction_model = model.predict(test_data)
prediction_model = prediction_model.argmax(axis=-1)

target_names = ['class 0', 'class 1', 'class 2','class 3', 'class 4']

np.savetxt("keras_prediction.txt", np.array(prediction_model), "%1.0f")

