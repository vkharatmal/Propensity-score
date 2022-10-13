import pandas as pd
import numpy as np
from sklearn.model_selection import cross_validate
from sklearn.model_selection import RepeatedStratifiedKFold
# import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn import metrics
import lightgbm as lgb
# sns.set(style='white')
import pickle



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

	lgbm_model = lgb.LGBMClassifier(n_jobs=-1, n_estimators=7000, 
	                     **params)
	lgbm_model.fit(xTrt,yTr,eval_set=[(xTet,yTe)],eval_metric='auc',verbose=0,early_stopping_rounds=80)

	# Model score
	print('Generic LGBM model Test score:',lgbm_model.score(xTet,yTe))
	print('Generic LGBM model Train score:',lgbm_model.score(xTrt,yTr))
	print('f1:',metrics.f1_score(yTe,lgbm_model.predict(xTet)),
	'\nrecall:',metrics.recall_score(yTe,lgbm_model.predict(xTet)),
	'\nprecision:',metrics.precision_score(yTe,lgbm_model.predict(xTet)))
	fpr, tpr, thresholds = metrics.roc_curve(yTe, lgbm_model.predict_proba(xTet)[:,1])
	print('AUROC:',metrics.auc(fpr, tpr))

	# Prediction results
	Result=pd.DataFrame(xTe.Identifier)
	# Result=pd.DataFrame()
	Result['Propensity']=round(pd.DataFrame(lgbm_model.predict_proba(xTet))[1],4)*100
	Result['Product_Activation Prediction']=pd.DataFrame(lgbm_model.predict(xTet))

	# Saving into file
	Result.to_csv('Train_Results.csv',index=False)
	pickle.dump(lgbm_model,open('lgbmC_w_selection.sav','wb'))
	return(print('Results saved into file "Train_Results.csv".'))

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
	lgbm_model=pickle.load(open('lgbmf_model.sav','rb'))

	# Precition and saving results
	results=pd.DataFrame(df.Identifier)
	results['Product Activation']=lgbm_model.predict(df.drop(columns=['Product_Activation','Identifier']))
	results['Propensity']=round(pd.DataFrame(lgbm_model.predict_proba(df.drop(columns=['Product_Activation','Identifier'])))[1],4)*100
	results.to_csv('Prediction_Results.csv',index=False)

	return(print('Results saved into file "Prediction_Results.csv".'))
