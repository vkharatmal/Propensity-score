import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
import numpy as np
import pickle
df_smote=pd.read_csv('dfsmote_w_id.csv')
xTr,xTe,yTr,yTe=train_test_split(df_smote.drop(columns=['Product_Activation']),
                                 df_smote.Product_Activation,random_state=42)


val=pd.read_csv('validation.csv')

xTrm=xTr.drop(columns=['Identifier'])
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
lgbm_model.fit(xTrm,yTr,eval_set=[(xTet,yTe)],eval_metric='auc',verbose=0,early_stopping_rounds=80)

print(lgbm_model.predict(val.drop(columns=['Identifier','Product_Activation'])))

pickle.dump(lgbm_model,open('lgbmf_model.sav','wb'))
# parameters={'learning_rate':np.arange(0.001,1.0,0.002),'max_depth':range(-50,-1,1),
# "bagging_freq": range(1,5,1), "bagging_fraction": np.arange(0.1,0.75,0.05)}

'''
grid=RandomizedSearchCV(estimator=lgbm_model,param_distributions=parameters, 
                        n_iter = 10, cv = 3, verbose = 1, random_state = 42, 
                               n_jobs = -1).fit(xTrm,yTrm)
print(grid.best_params_)
'''