import numpy as np
import pandas as pd

import sklearn.decomposition
import sklearn.grid_search

from sklearn.cross_validation import train_test_split
import scipy as sp
import math
import random
import csv
from sklearn.metrics import precision_score, recall_score, confusion_matrix, classification_report,accuracy_score, f1_score

from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC


#data = pd.read_csv("/home/vinod/Documents/Task/eclipse-metrics-files-3.0.csv")

data = pd.read_csv('/home/vinod/Documents/Task/eclipse-metrics-files-3.0.csv', delimiter=';', encoding="utf-8-sig")

inputColumns = ['pre','ACD','FOUT_avg','FOUT_max','FOUT_sum','MLOC_avg','MLOC_max','MLOC_sum','NBD_avg','NBD_max','NBD_sum','NOF_avg','NOF_max','NOF_sum','NOI','NOM_avg','NOM_max','NOM_sum','NOT','NSF_avg','NSF_max','NSF_sum','NSM_avg','NSM_max','NSM_sum','PAR_avg','PAR_max','PAR_sum','TLOC','VG_avg','VG_max','VG_sum','AnonymousClassDeclaration','ArrayAccess','ArrayCreation','ArrayInitializer','ArrayType','Assignment','Block','BooleanLiteral','BreakStatement','CastExpression','CatchClause','CharacterLiteral','ClassInstanceCreation','ConditionalExpression','ConstructorInvocation','ContinueStatement','DoStatement','EmptyStatement','ExpressionStatement','FieldAccess','FieldDeclaration','ForStatement','IfStatement','ImportDeclaration','InfixExpression','Initializer','Javadoc','LabeledStatement','MethodDeclaration','MethodInvocation','NullLiteral','NumberLiteral','ParenthesizedExpression','PostfixExpression','PrefixExpression','PrimitiveType','QualifiedName','ReturnStatement','SimpleName','SimpleType','SingleVariableDeclaration','StringLiteral','SuperConstructorInvocation','SuperFieldAccess','SuperMethodInvocation','SwitchCase','SwitchStatement','SynchronizedStatement','ThisExpression','ThrowStatement','TryStatement','TypeDeclaration','TypeDeclarationStatement','TypeLiteral','VariableDeclarationExpression','VariableDeclarationFragment','VariableDeclarationStatement','WhileStatement','InstanceofExpression','Modifier','SUM','NORM_AnonymousClassDeclaration','NORM_ArrayAccess','NORM_ArrayCreation','NORM_ArrayInitializer','NORM_ArrayType','NORM_Assignment','NORM_Block','NORM_BooleanLiteral','NORM_BreakStatement','NORM_CastExpression','NORM_CatchClause','NORM_CharacterLiteral','NORM_ClassInstanceCreation','NORM_CompilationUnit','NORM_ConditionalExpression','NORM_ConstructorInvocation','NORM_ContinueStatement','NORM_DoStatement','NORM_EmptyStatement','NORM_ExpressionStatement','NORM_FieldAccess','NORM_FieldDeclaration','NORM_ForStatement','NORM_IfStatement','NORM_ImportDeclaration','NORM_InfixExpression','NORM_Initializer','NORM_Javadoc','NORM_LabeledStatement','NORM_MethodDeclaration','NORM_MethodInvocation','NORM_NullLiteral','NORM_NumberLiteral','NORM_PackageDeclaration','NORM_ParenthesizedExpression','NORM_PostfixExpression','NORM_PrefixExpression','NORM_PrimitiveType','NORM_QualifiedName','NORM_ReturnStatement','NORM_SimpleName','NORM_SimpleType','NORM_SingleVariableDeclaration','NORM_StringLiteral','NORM_SuperConstructorInvocation','NORM_SuperFieldAccess','NORM_SuperMethodInvocation','NORM_SwitchCase','NORM_SwitchStatement','NORM_SynchronizedStatement','NORM_ThisExpression','NORM_ThrowStatement','NORM_TryStatement','NORM_TypeDeclaration','NORM_TypeDeclarationStatement','NORM_TypeLiteral','NORM_VariableDeclarationExpression','NORM_VariableDeclarationFragment','NORM_VariableDeclarationStatement','NORM_WhileStatement','NORM_InstanceofExpression','NORM_Modifier']



data0 = data.loc[data['post'] == 0]
data1 = data.loc[data['post'] == 1]
data2 = data.loc[data['post'] == 2]
data3 = data.loc[data['post'] == 3]
data4 = data.loc[data['post'] == 4]
data5 = data.loc[data['post'] == 5]
data6 = data.loc[data['post'] == 6]
data7 = data.loc[data['post'] == 7]
data8 = data.loc[data['post'] == 8]
data9 = data.loc[data['post'] == 9]
data10 = data.loc[data['post'] == 10]
data11 = data.loc[data['post'] == 11]
data12 = data.loc[data['post'] == 12]
data13 = data.loc[data['post'] == 13]
data14 = data.loc[data['post'] == 14]
data15 = data.loc[data['post'] == 15]
data16 = data.loc[data['post'] == 16]
data17 = data.loc[data['post'] == 17]

X0 = data0[inputColumns]
y0 = data0['post']
X_train0, X_test0, y_train0, y_test0 = train_test_split(X0, y0, test_size=0.30, random_state=42)

X1 = data1[inputColumns]
y1 = data1['post']
X_train1, X_test1, y_train1, y_test1 = train_test_split(X1, y1, test_size=0.30, random_state=42)

X2 = data2[inputColumns]
y2 = data2['post']
X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y2, test_size=0.30, random_state=42)

X3 = data3[inputColumns]
y3 = data3['post']
X_train3, X_test3, y_train3, y_test3 = train_test_split(X3, y3, test_size=0.30, random_state=42)

X4 = data4[inputColumns]
y4 = data4['post']
X_train4, X_test4, y_train4, y_test4 = train_test_split(X4, y4, test_size=0.30, random_state=42)

X5 = data5[inputColumns]
y5 = data5['post']
X_train5, X_test5, y_train5, y_test5 = train_test_split(X5, y5, test_size=0.30, random_state=42)

X6 = data6[inputColumns]
y6 = data6['post']
X_train6, X_test6, y_train6, y_test6 = train_test_split(X6, y6, test_size=0.30, random_state=42)

X7 = data7[inputColumns]
y7 = data7['post']
X_train7, X_test7, y_train7, y_test7 = train_test_split(X7, y7, test_size=0.30, random_state=42)

X8 = data8[inputColumns]
y8 = data8['post']
X_train8, X_test8, y_train8, y_test8 = train_test_split(X8, y8, test_size=0.30, random_state=42)

X9 = data9[inputColumns]
y9 = data9['post']
X_train9, X_test9, y_train9, y_test9 = train_test_split(X9, y9, test_size=0.30, random_state=42)

X10 = data10[inputColumns]
y10 = data10['post']
X_train10, X_test10, y_train10, y_test10 = train_test_split(X10, y10, test_size=0.30, random_state=42)

X11 = data11[inputColumns]
y11 = data11['post']
X_train11, X_test11, y_train11, y_test11 = train_test_split(X11, y11, test_size=0.30, random_state=42)

X12 = data12[inputColumns]
y12 = data12['post']
X_train12, X_test12, y_train12, y_test12 = train_test_split(X12, y12, test_size=0.30, random_state=42)

X13 = data13[inputColumns]
y13 = data13['post']
X_train13, X_test13, y_train13, y_test13 = train_test_split(X13, y13, test_size=0.30, random_state=42)

X14 = data14[inputColumns]
y14 = data14['post']
X_train14, X_test14, y_train14, y_test14 = train_test_split(X14, y14, test_size=0.30, random_state=42)

X15 = data15[inputColumns]
y15 = data15['post']
X_train15, X_test15, y_train15, y_test15 = train_test_split(X15, y15, test_size=0.30, random_state=42)

X16 = data16[inputColumns]
y16 = data16['post']
X_train16, X_test16, y_train16, y_test16 = train_test_split(X16, y16, test_size=0.30, random_state=42)

X17 = data17[inputColumns]
y17 = data17['post']
X_train17, X_test17, y_train17, y_test17 = train_test_split(X17, y17, test_size=0.30, random_state=42)

Xtrain = [X_train0,X_train1, X_train2, X_train3,X_train4,X_train5,X_train6,X_train7,X_train8,X_train9,X_train10,X_train11,X_train12,X_train13,X_train14,X_train15,X_train16,X_train17]

Xtest = [X_test0,X_test1, X_test2, X_test3,X_test4,X_test5,X_test6,X_test7,X_test8,X_test9,X_test10,X_test11,X_test12,X_test13,X_test14,X_test15,X_test16,X_test17]

ytrain = [y_train0, y_train1, y_train2, y_train3,y_train4,y_train5,y_train6,y_train7,y_train8,y_train9,y_train10,y_train11,y_train12,y_train13,y_train14,y_train15,y_train16,y_train17]

ytest = [y_test0, y_test1, y_test2, y_test3,y_test4,y_test5,y_test6,y_test7,y_test8,y_test9,y_test10,y_test11,y_test12,y_test13,y_test14,y_test15,y_test16,y_test17]


X_train = pd.concat(Xtrain)
X_test = pd.concat(Xtest)
y_train= pd.concat(ytrain)
y_test = pd.concat(ytest)

#Normalize data
X_Normalize = pd.concat([X_train,X_test])
mean = X_Normalize.mean(axis=0)
std = (X_Normalize - mean).std()

X_train = (X_train - mean) / std
X_test = (X_test - mean) / std



test_Y1 = y_test.reshape(y_test.shape[0], 1)
y_test = np.transpose(test_Y1)

print X_train.shape,X_test.shape,y_train.shape,y_test.shape




def getSVC():

	classifiersPool = []
	classifier = SVC(C=100, cache_size=200, class_weight=None, coef0=0.0, degree=3, gamma=0.1, kernel='rbf', max_iter=-1, probability=True, random_state=None, shrinking=True, tol=0.001, verbose=False)
	classifier.fit(X_train,y_train.values)
	classifiersPool.append(classifier)
	return classifiersPool


def getAdaBoostClassifier():

	classifiersPool = []
	classifier = AdaBoostClassifier(n_estimators=500)
	classifier.fit(X_train,y_train.values)
	classifiersPool.append(classifier)
	return classifiersPool




def getGradientBoostingClassifier():

	classifiersPool = []
	classifier = GradientBoostingClassifier(n_estimators=50, learning_rate=1.0, max_depth=1, random_state=0)
	classifier.fit(X_train,y_train.values)
	classifiersPool.append(classifier)
	return classifiersPool



def getPredictions(classifiers, X_test):
	totalPredictions = []
	for classifier in classifiers:
		predictions = classifier.predict(X_test)
		totalPredictions.append(predictions)
	return totalPredictions





classifiers = getSVC()
predictions = getPredictions(classifiers, X_test)
pred = np.asarray(predictions)

print "--------------------------------------------------"
print "SVM - SVC"
print "--------------------------------------------------"
print 'Accuracy:', accuracy_score(y_test.flatten(), pred.flatten())
print 'F1 score:', f1_score(y_test.flatten(), pred.flatten())
print 'Recall:', recall_score(y_test.flatten(), pred.flatten())
print 'Precision:', precision_score(y_test.flatten(), pred.flatten())
print '\n clasification report:\n', classification_report(y_test.flatten(), pred.flatten())

#print 'SVC score: %f' % accuracy_score(y_test.flatten(), pred.flatten())


classifiers = getAdaBoostClassifier()
predictions = getPredictions(classifiers, X_test)
pred = np.asarray(predictions)

print "--------------------------------------------------"
print "AdaboostClassifier"
print "--------------------------------------------------"
print 'Accuracy:', accuracy_score(y_test.flatten(), pred.flatten())
print 'F1 score:', f1_score(y_test.flatten(), pred.flatten())
print 'Recall:', recall_score(y_test.flatten(), pred.flatten())
print 'Precision:', precision_score(y_test.flatten(), pred.flatten())
print '\n clasification report:\n', classification_report(y_test.flatten(), pred.flatten())

#print 'Adaboost score: %f' % accuracy_score(y_test.flatten(), pred.flatten())


classifiers = getGradientBoostingClassifier()
predictions = getPredictions(classifiers, X_test)
pred = np.asarray(predictions)

print "--------------------------------------------------"
print "GradientBoostingClassifier"
print "--------------------------------------------------"
print 'Accuracy:', accuracy_score(y_test.flatten(), pred.flatten())
print 'F1 score:', f1_score(y_test.flatten(), pred.flatten())
print 'Recall:', recall_score(y_test.flatten(), pred.flatten())
print 'Precision:', precision_score(y_test.flatten(), pred.flatten())
print '\n clasification report:\n', classification_report(y_test.flatten(), pred.flatten())

#print 'Gradient boosting score: %f' % accuracy_score(y_test.flatten(), pred.flatten())




# Neural network used the library skflow which is based on tensorflow

import skflow

model = skflow.TensorFlowDNNClassifier(hidden_units=[10, 20, 10], n_classes=17, batch_size=100, steps=3000, optimizer="SGD", learning_rate=0.01)

model.fit(X_train, y_train.values)


#y_test = test_data['Y']
y_prediction = model.predict(X_test)



print "prediction accuracy:", np.sum(y_test.flatten() == y_prediction)*1./len(y_test.flatten())

