import xgboost
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics
from xgboost import plot_importance
from matplotlib import pyplot
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import learning_curve
from sklearn.metrics import mean_squared_error
import seaborn as sns

trainn = pd.read_csv('train1.csv')                     #训练集矩阵
train = np.array(trainn)
print(train.shape)
trainn_lable = pd.read_csv('train1-lable.csv')         #训练集lable
train_lable = np.array(trainn_lable)
#y_train = tf.keras.utils.to_categorical(train_lable,2) #转换训练集lable为onehot编码

'''读取测试集'''
testt = pd.read_csv('test1.csv')                       #测试集矩阵
test = np.array(testt)
print(test.shape)
testt_lable = pd.read_csv('test1-lable.csv')           #测试集lable
test_lable = np.array(testt_lable)
#test_lable = tf.keras.utils.to_categorical(test_lable,2)

num_round=1000
bst=XGBClassifier(max_depth=7,n_estimators=num_round,eta = 0.1,objective='binary:logistic')


eval_set = [(train,train_lable),(test,test_lable)]
bst.fit(train,train_lable,early_stopping_rounds=10,eval_metric=['error','logloss'],eval_set=eval_set)

bst.save_model('local_xgb.model')
#模型读取
modelfile='local_xgb.model'
xgbst=xgboost.Booster({'nthread':8},model_file=modelfile)
#模型预测
# dtest= xgboost.DMatrix(data = test_x, label = test_y)
dtest= xgboost.DMatrix(data = test)#.predict()里必须是DMatrix格式
y_pred=c=xgbst.predict(dtest)
test_predictions=[round(value) for value in y_pred]

#模型效果验证
test_accuracy = accuracy_score(test_lable, test_predictions)
print("Test Accuracy: %.2f%%" % (test_accuracy * 100.0))
test_auc=metrics.roc_auc_score(test_lable,y_pred)
print("auc:%.2f%%"%(test_auc*100.0))
test_recall=metrics.recall_score(test_lable,test_predictions)
print("recall:%.2f%%"%(test_recall*100.0))
test_f1=metrics.f1_score(test_lable,test_predictions)
print("f1:%.2f%%"%(test_f1*100.0))
test_precision=metrics.precision_score(test_lable,test_predictions)
print("precision:%.2f%%"%(test_precision*100.0))

#ROC曲线
fpr,tpr,threshold=metrics.roc_curve(test_lable,y_pred)
#lw是line width
pyplot.plot(fpr,tpr,color='blue',lw=2,label='ROC curve(area=%.2f%%)'%(test_auc*100.0))
#假正率为横坐标，真正率为纵坐标做曲线
pyplot.legend(loc="lower right")
pyplot.plot([0,1],[0,1],color='navy',lw=2,linestyle='--')
pyplot.xlabel('False Positive Rate')
pyplot.ylabel('True Positive Rate')
pyplot.title('ROC curve')
pyplot.savefig('XGboost_figure1.png')
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import PrecisionRecallDisplay
prec, recall, _ = precision_recall_curve(test_lable,y_pred, pos_label=clf.classes_[0])
pr_display = PrecisionRecallDisplay(precision=prec, recall=recall).plot()


