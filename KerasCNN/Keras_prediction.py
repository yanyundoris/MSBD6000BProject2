from keras.models import load_model
import numpy as np
from Loading_data import LoadImage2Array
import keras


num_classes = 5

model = load_model('/Users/yanyunliu/PycharmProjects/TensorFlowTutorial/MSBD6000BProject2/KerasCNN/keras_cifar10_trained_model_batchnormal.h5')

# val_data, val_label = LoadImage2Array('/disk02/data/eLearning/yyliu/DeepLearningProject/val.txt', image_size = 32, use_gray = False, use_keras = True)

# val_label = keras.utils.to_categorical(val_label, num_classes)
# val_data = val_data.astype('float32')
# val_data /= 255

# scores = model.evaluate(val_data,val_label, verbose=1)
# print scores


test_data, test_label = LoadImage2Array('/Users/yanyunliu/PycharmProjects/TensorFlowTutorial/data/sample_truth_onelabel.txt', image_size = 32, use_gray = False, use_keras = True)

test_label = keras.utils.to_categorical(test_label, num_classes)
test_data = test_data.astype('float32')
test_data /= 255
test_label = keras.utils.to_categorical(test_label, num_classes)

# scores = model.evaluate(test_data,test_label, verbose=1)
# scores = model.evaluate(test_data,test_label, verbose=1)
print model.predict(test_data)
prediction_model = model.predict(test_data)

np.savetxt("keras_prediction.txt", np.array(prediction_model), "%1.0f")

# train_data, train_label = LoadImage2Array('/disk02/data/eLearning/yyliu/DeepLearningProject/train.txt', image_size = 32, use_gray = False, use_keras = True)
#
# # train_label = keras.utils.to_categorical(train_label, num_classes)
# train_data = train_data.astype('float32')
# train_data /= 255


# scores = model.evaluate(test_data,test_label, verbose=1)
# prediction_model = model.predict(test_data)
# print prediction_model
# np.savetxt("train_features.txt",prediction_model,"%1.3f")
# np.savetxt("train_label.txt", np.array(train_label), "%1.0f")
