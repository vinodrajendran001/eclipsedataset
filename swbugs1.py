#import statements

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

#load data

data = pd.read_csv('/home/vinod/Documents/Task/eclipse-metrics-files-3.0.csv', delimiter=';', encoding="utf-8-sig")


def getRandomForestClassifier():

	classifiersPool = []
	classifier = RandomForestClassifier(n_estimators=100, max_depth=None, min_samples_split=1, random_state=0)
	classifier.fit(X_train,y_train.values)

	classifiersPool.append(classifier)
	return classifiersPool

def getDecisionTreeClassifier():

	classifiersPool = []
	classifier = DecisionTreeClassifier(max_depth=None, min_samples_split=1, random_state=0)
	classifier.fit(X_train,y_train.values)
	classifiersPool.append(classifier)
	return classifiersPool

def getExtraTreesClassifier():

	classifiersPool = []
	classifier = ExtraTreesClassifier(n_estimators=200, max_depth=None, min_samples_split=1, random_state=0)
	classifier.fit(X_train,y_train.values)
	classifiersPool.append(classifier)
	return classifiersPool


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

#input features to be considered

inputColumns = ['pre','ACD','FOUT_avg','FOUT_max','FOUT_sum','MLOC_avg','MLOC_max','MLOC_sum','NBD_avg','NBD_max','NBD_sum','NOF_avg','NOF_max','NOF_sum','NOI','NOM_avg','NOM_max','NOM_sum','NOT','NSF_avg','NSF_max','NSF_sum','NSM_avg','NSM_max','NSM_sum','PAR_avg','PAR_max','PAR_sum','TLOC','VG_avg','VG_max','VG_sum','AnonymousClassDeclaration','ArrayAccess','ArrayCreation','ArrayInitializer','ArrayType','Assignment','Block','BooleanLiteral','BreakStatement','CastExpression','CatchClause','CharacterLiteral','ClassInstanceCreation','ConditionalExpression','ConstructorInvocation','ContinueStatement','DoStatement','EmptyStatement','ExpressionStatement','FieldAccess','FieldDeclaration','ForStatement','IfStatement','ImportDeclaration','InfixExpression','Initializer','Javadoc','LabeledStatement','MethodDeclaration','MethodInvocation','NullLiteral','NumberLiteral','ParenthesizedExpression','PostfixExpression','PrefixExpression','PrimitiveType','QualifiedName','ReturnStatement','SimpleName','SimpleType','SingleVariableDeclaration','StringLiteral','SuperConstructorInvocation','SuperFieldAccess','SuperMethodInvocation','SwitchCase','SwitchStatement','SynchronizedStatement','ThisExpression','ThrowStatement','TryStatement','TypeDeclaration','TypeDeclarationStatement','TypeLiteral','VariableDeclarationExpression','VariableDeclarationFragment','VariableDeclarationStatement','WhileStatement','InstanceofExpression','Modifier','SUM','NORM_AnonymousClassDeclaration','NORM_ArrayAccess','NORM_ArrayCreation','NORM_ArrayInitializer','NORM_ArrayType','NORM_Assignment','NORM_Block','NORM_BooleanLiteral','NORM_BreakStatement','NORM_CastExpression','NORM_CatchClause','NORM_CharacterLiteral','NORM_ClassInstanceCreation','NORM_CompilationUnit','NORM_ConditionalExpression','NORM_ConstructorInvocation','NORM_ContinueStatement','NORM_DoStatement','NORM_EmptyStatement','NORM_ExpressionStatement','NORM_FieldAccess','NORM_FieldDeclaration','NORM_ForStatement','NORM_IfStatement','NORM_ImportDeclaration','NORM_InfixExpression','NORM_Initializer','NORM_Javadoc','NORM_LabeledStatement','NORM_MethodDeclaration','NORM_MethodInvocation','NORM_NullLiteral','NORM_NumberLiteral','NORM_PackageDeclaration','NORM_ParenthesizedExpression','NORM_PostfixExpression','NORM_PrefixExpression','NORM_PrimitiveType','NORM_QualifiedName','NORM_ReturnStatement','NORM_SimpleName','NORM_SimpleType','NORM_SingleVariableDeclaration','NORM_StringLiteral','NORM_SuperConstructorInvocation','NORM_SuperFieldAccess','NORM_SuperMethodInvocation','NORM_SwitchCase','NORM_SwitchStatement','NORM_SynchronizedStatement','NORM_ThisExpression','NORM_ThrowStatement','NORM_TryStatement','NORM_TypeDeclaration','NORM_TypeDeclarationStatement','NORM_TypeLiteral','NORM_VariableDeclarationExpression','NORM_VariableDeclarationFragment','NORM_VariableDeclarationStatement','NORM_WhileStatement','NORM_InstanceofExpression','NORM_Modifier']


X = data[inputColumns]
y = data['post']

#Uncomment this to perform dimensionality reduction
'''
#dimensionality Reduction using PCA
from sklearn.decomposition import RandomizedPCA
pca = RandomizedPCA(copy=True, iterated_power=3, n_components=60,random_state=None, whiten=False)
pca.fit(X)
#RandomizedPCA(copy=True, iterated_power=3, n_components=64,random_state=None, whiten=False)
#print(pca.explained_variance_ratio_)

X = pca.transform(X)
'''
print X.shape
#-------------------------------

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)


test_Y1 = y_test.reshape(y_test.shape[0], 1)
y_test = np.transpose(test_Y1)

classifiers = getDecisionTreeClassifier()
predictions = getPredictions(classifiers, X_test)
pred = np.asarray(predictions)

print "--------------------------------------------------"
print "Decision Tree"
print "--------------------------------------------------"
print 'Accuracy:', accuracy_score(y_test.flatten(), pred.flatten())
print 'F1 score:', f1_score(y_test.flatten(), pred.flatten())
print 'Recall:', recall_score(y_test.flatten(), pred.flatten())
print 'Precision:', precision_score(y_test.flatten(), pred.flatten())
print '\n clasification report:\n', classification_report(y_test.flatten(), pred.flatten())

#print 'Decision Tree score: %f' % accuracy_score(y_test.flatten(), pred.flatten())



classifiers = getRandomForestClassifier()
predictions = getPredictions(classifiers, X_test)
pred = np.asarray(predictions)

print "--------------------------------------------------"
print "Random Forest"
print "--------------------------------------------------"
print 'Accuracy:', accuracy_score(y_test.flatten(), pred.flatten())
print 'F1 score:', f1_score(y_test.flatten(), pred.flatten())
print 'Recall:', recall_score(y_test.flatten(), pred.flatten())
print 'Precision:', precision_score(y_test.flatten(), pred.flatten())
print '\n clasification report:\n', classification_report(y_test.flatten(), pred.flatten())

#print 'Random Forest score: %f' % accuracy_score(y_test.flatten(), pred.flatten())

classifiers = getExtraTreesClassifier()
predictions = getPredictions(classifiers, X_test)
pred = np.asarray(predictions)

print "--------------------------------------------------"
print "ExtraTreesClassifier"
print "--------------------------------------------------"
print 'Accuracy:', accuracy_score(y_test.flatten(), pred.flatten())
print 'F1 score:', f1_score(y_test.flatten(), pred.flatten())
print 'Recall:', recall_score(y_test.flatten(), pred.flatten())
print 'Precision:', precision_score(y_test.flatten(), pred.flatten())
print '\n clasification report:\n', classification_report(y_test.flatten(), pred.flatten())

#print 'Extra Trees classifier score: %f' % accuracy_score(y_test.flatten(), pred.flatten())

# For the following classifiers Normalization of data is required

mean = X.mean(axis=0)
std = (X - mean).std()
X = (X - mean) / std

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

test_Y1 = y_test.reshape(y_test.shape[0], 1)
y_test = np.transpose(test_Y1)


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




# Deep Learning - DNN used the library skflow which is based on tensorflow

import skflow

model = skflow.TensorFlowDNNClassifier(hidden_units=[10, 20, 10], n_classes=17, batch_size=100, steps=3000, optimizer="SGD", learning_rate=0.01)

model.fit(X_train, y_train.values)


#y_test = test_data['Y']
y_prediction = model.predict(X_test)



print "prediction accuracy:", np.sum(y_test.flatten() == y_prediction)*1./len(y_test.flatten())





