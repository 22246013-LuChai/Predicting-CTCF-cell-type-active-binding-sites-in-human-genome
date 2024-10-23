from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

import tensorflow as tf
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import pandas as pd
import numpy as np
from keras.utils import to_categorical

trainn = pd.read_csv('train2.csv')                     #训练集矩阵
train = np.array(trainn)
rows1, columns = train.shape
trainb=train.reshape(rows1,15,20)
trainn_lable = pd.read_csv('train2-lable.csv')         #训练集lable
train_lable = np.array(trainn_lable)
y_train = tf.keras.utils.to_categorical(train_lable,2) #转换训练集lable为onehot编码

'''读取测试集'''
testt = pd.read_csv('test2.csv')                       #测试集矩阵
test = np.array(testt)
rows2, columns = test.shape
testb=test.reshape(rows2,15,20)
testt_lable = pd.read_csv('test2-lable.csv')           #测试集lable
test_lable = np.array(testt_lable)
x_test = tf.keras.utils.to_categorical(test_lable,2)   #转换测试集lable为onehot编码

##载入模型所需包
from keras import models
from tensorflow.keras.callbacks import ModelCheckpoint
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Reshape
def model():
     model = models.Sequential()
     model.add(Reshape((15,20,1),input_shape=(15,20)))
     model.add(Conv2D(20,(1,1),activation='tanh'))
     model.add(MaxPooling2D((1,1)))
     model.add(Conv2D(50,(1,1),activation='tanh'))
     model.add(MaxPooling2D((1,1)))
     model.add(Flatten())
     model.add(Dense(500,activation='tanh'))
     model.add(Dense(40))
     model.add(Dense(2,activation='softmax'))
     return model


model = model()

model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics=['acc'])
model.summary()

#tf.config.run_functions_eagerly(True)
model.fit(trainb,y_train,batch_size=40,epochs=20,callbacks = None,verbose=1)

test_loss, test_acc = model.evaluate(testb, x_test)
print(test_acc)

# Set a random seed for reproducibility
seed = 42
np.random.seed(seed)
random.seed(seed)
tf.random.set_seed(seed)

'''绘制ROC曲线'''
from sklearn.metrics import roc_curve
import matplotlib
import matplotlib.pyplot as plt
y_pred = model.predict(testb)[:,1]
fpr_keras, tpr_keras, thresholds_keras = roc_curve(test_lable, y_pred)

from sklearn.metrics import auc
auc_keras = auc(fpr_keras, tpr_keras)

predictions_df = pd.DataFrame({'True Label': test_lable, 'Predicted Probability': y_pred})
predictions_df.to_csv('predicted_values2.csv', index=False)

plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_keras, tpr_keras, label='Keras (area = {:.3f})'.format(auc_keras))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.savefig('figure2.png')

'''绘制PR曲线'''
from sklearn.metrics import precision_recall_curve, auc
import matplotlib
import matplotlib.pyplot as plt

# Get the predicted probabilities for the positive class
y_pred = model.predict(testb)[:, 1]

# Compute precision, recall, and thresholds for the PR curve
precision, recall, thresholds = precision_recall_curve(test_lable, y_pred)

# Compute the AUC for the Precision-Recall curve
auc_pr = auc(recall, precision)

# Plot the Precision-Recall curve
plt.figure(2)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(recall, precision, label='AUPR (area = {:.3f})'.format(auc_pr))
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc='best')

# Save the figure
plt.savefig('figure_pr2.png')
