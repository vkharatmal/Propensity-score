from imblearn.over_sampling import SMOTE
import pandas as pd

df=pd.read_csv('Product Training and Testing.csv')
X_train_smt = df.drop(columns=['Product_Activation'])
y_train_smt = df.Product_Activation
smt = SMOTE()
xTrain_smt, yTrain_smt = smt.fit_resample(X_train_smt, y_train_smt)

df_smote=pd.DataFrame(xTrain_smt)
df_smote['Product_Activation']=yTrain_smt
df_smote.to_csv('dfsmote_w_id.csv',index=False)