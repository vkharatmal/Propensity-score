import pandas as pd
import numpy as np
from sklearn.model_selection import cross_validate
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn import metrics
import lightgbm as lgb
import pickle
from sklearn.calibration import CalibratedClassifierCV


def train():
	# getting filename to import
	filename=input('Please enter filename with extension:')

	# Importing file
	if filename.split('.')[1]=='xlsx' or filename.split('.')[1]=='xls':
		df=pd.read_excel(filename)
	elif filename.split('.')[1]=='csv':
		df=pd.read_csv(filename)
	else:
		return(print('Incorrect file format'),-1)


	# Train Test Split of data
	xTr,xTe,yTr,yTe=train_test_split(df.drop(columns=['Product_Activation']),
	                                 df.Product_Activation,random_state=42)
	xTrt=xTr.drop(columns=['Identifier'])
	xTet=xTe.drop(columns=['Identifier'])

	params = {
	'max_bin' : [128],
	'num_leaves': [8],
	'reg_alpha' : [1.2],
	'reg_lambda' : [1.2],
	'min_data_in_leaf' : [50],
	'bagging_fraction' : [0.5],
	'learning_rate' : [0.001]
	}

	lgbm_model = lgb.LGBMClassifier(n_jobs=-1, n_estimators=7000, **params)
	lgbm_model.fit(xTrt,yTr,eval_set=[(xTet,yTe)],eval_metric='auc',verbose=0,early_stopping_rounds=80)


	# Model score
	print('Generic LGBM model Test score:',lgbm_model.score(xTet,yTe))
	print('Generic LGBM model Train score:',lgbm_model.score(xTrt,yTr))
	print('f1:',metrics.f1_score(yTe,lgbm_model.predict(xTet)),
	'\nrecall:',metrics.recall_score(yTe,lgbm_model.predict(xTet)),
	'\nprecision:',metrics.precision_score(yTe,lgbm_model.predict(xTet)))
	fpr, tpr, thresholds = metrics.roc_curve(yTe, lgbm_model.predict_proba(xTet)[:,1])
	print('AUROC:',metrics.auc(fpr, tpr))



	# Caliberating model probabilities
	d_df=pd.read_csv('Undersampled.csv')
	d_df_val=d_df.drop(columns=['Product_Activation','Identifier'])
	dfxtr,dfxte,dfytr,dfyte=train_test_split(d_df_val,d_df.Product_Activation,random_state=4)

	calibrated_clf = CalibratedClassifierCV(base_estimator=lgbm_model, cv=3)
	calibrated_clf.fit(dfxtr, dfytr)
	cal_prob=calibrated_clf.predict_proba(xTet)
	#print(pd.DataFrame(cal_prob))

	print('Generic caliberated LGBM model Test score:',calibrated_clf.score(xTet,yTe))
	print('Generic caliberated LGBM model Train score:',calibrated_clf.score(xTrt,yTr))
	print('f1:',metrics.f1_score(yTe,cal_prob),
	'\nrecall:',metrics.recall_score(yTe,cal_prob),
	'\nprecision:',metrics.precision_score(yTe,cal_prob)
	fpr, tpr, thresholds = metrics.roc_curve(yTe, calibrated_clf.predict_proba(xTet)[:,1])
	print('AUROC:',metrics.auc(fpr, tpr))
	
	
	# Prediction results
	Result=pd.DataFrame(xTe.Identifier)
	Result['Propensity']=round(pd.DataFrame(lgbm_model.predict_proba(xTet))[1],4)*100
	Result['Product_Activation Prediction']=pd.DataFrame(lgbm_model.predict(xTet))
	Result['Caliberated prob']=pd.DataFrame(cal_prob)[1]


	# Saving into file
	Result.to_csv('Train_Results.csv',index=False)
	pickle.dump(calibrated_clf,open('Caliberated_model.sav','wb'))

	return(print('Results saved into file "Train_Results.csv" and Model is saved as "Caliberated_model.sav"'))

def predict():
	# getting filename to import
	filename=input('Please enter filename with extension:')

	# Importing file
	if filename.split('.')[1]=='xlsx' or filename.split('.')[1]=='xls':
		df=pd.read_excel(filename)
	elif filename.split('.')[1]=='csv':
		df=pd.read_csv(filename)
	else:
		return('Incorrect file format',-1)

	# Importing saved model using Pickle
	lgbm_model=pickle.load(open('Caliberated_model_og.sav','rb'))


	# Precition and saving results
	results=pd.DataFrame(df.Identifier)
	results['Product Activation']=lgbm_model.predict(df.drop(columns=['Product_Activation','Identifier']))
	results['Propensity']=round(pd.DataFrame(lgbm_model.predict_proba(df.drop(columns=['Product_Activation','Identifier'])))[1],4)*100
	results.to_csv('Prediction_Results.csv',index=False)

	return(print('Results saved into file "Prediction_Results.csv".'))
