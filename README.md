# Ex-07-Feature-Selection
## AIM
To Perform the various feature selection techniques on a dataset and save the data to a file. 

# Explanation
Feature selection is to find the best set of features that allows one to build useful models.
Selecting the best features helps the model to perform well. 

# ALGORITHM
### STEP 1
Read the given Data
### STEP 2
Clean the Data Set using Data Cleaning Process
### STEP 3
Apply Feature selection techniques to all the features of the data set
### STEP 4
Save the data to the file


# CODE:
```
Developed by:Logeshwari.P
Register number:212221230055
#Importing libraries
from sklearn.datasets import load_boston
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
%matplotlib inline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.linear_model import RidgeCV, LassoCV, Ridge, Lasso

from sklearn.datasets import load_boston
boston = load_boston()

print(boston['DESCR'])

import pandas as pd
df = pd.DataFrame(boston['data'] )
df.head()

df.columns = boston['feature_names']
df.head()

df['PRICE']= boston['target']
df.head()

df.info()

plt.figure(figsize=(10, 8))
sns.distplot(df['PRICE'], rug=True)
plt.show()

#FILTER METHODS

X=df.drop("PRICE",1)
y=df["PRICE"]

from sklearn.feature_selection import SelectKBest, chi2
X, y = load_boston(return_X_y=True)
X.shape

#1.Variance Threshold
from sklearn.feature_selection import VarianceThreshold
selector = VarianceThreshold()
selector.fit_transform(X)

#2.Information gain/Mutual Information
from sklearn.feature_selection import mutual_info_regression
mi = mutual_info_regression(X, y);
mi = pd.Series(mi)
mi.sort_values(ascending=False)
mi.sort_values(ascending=False).plot.bar(figsize=(10, 4))

#3.SelectKBest Model
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import SelectKBest,SelectPercentile
skb = SelectKBest(score_func=f_classif, k=2) 
X_data_new = skb.fit_transform(X, y)
print('Number of features before feature selection: {}'.format(X.shape[1]))
print('Number of features after feature selection: {}'.format(X_data_new.shape[1]))

#4.Correlation Coefficient
cor=df.corr()
sns.heatmap(cor,annot=True)

#5.Mean Absolute Difference
mad=np.sum(np.abs(X-np.mean(X,axis=0)),axis=0)/X.shape[0]
plt.bar(np.arange(X.shape[1]),mad,color='teal')

#Processing data into array type.
from sklearn import preprocessing
lab = preprocessing.LabelEncoder()
y_transformed = lab.fit_transform(y)
print(y_transformed)

#6.Chi Square Test
X = X.astype(int)
chi2_selector = SelectKBest(chi2, k=2)
X_kbest = chi2_selector.fit_transform(X, y_transformed)
print('Original number of features:', X.shape[1])
print('Reduced number of features:', X_kbest.shape[1])

#7.SelectPercentile method
X_new = SelectPercentile(chi2, percentile=10).fit_transform(X, y_transformed)
X_new.shape

#WRAPPER METHOD

#1.Forward feature selection

from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.linear_model import LinearRegression
sfs = SFS(LinearRegression(),
          k_features=10,
          forward=True,
          floating=False,
          scoring = 'r2',
          cv = 0)
sfs.fit(X, y)
sfs.k_feature_names_

#2.Backward feature elimination

sbs = SFS(LinearRegression(),
         k_features=10,
         forward=False,
         floating=False,
         cv=0)
sbs.fit(X, y)
sbs.k_feature_names_

#3.Bi-directional elimination

sffs = SFS(LinearRegression(),
         k_features=(3,7),
         forward=True,
         floating=True,
         cv=0)
sffs.fit(X, y)
sffs.k_feature_names_

#4.Recursive Feature Selection

from sklearn.feature_selection import RFE
lr=LinearRegression()
rfe=RFE(lr,n_features_to_select=7)
rfe.fit(X, y)
print(X.shape, y.shape)
rfe.transform(X)
rfe.get_params(deep=True)
rfe.support_
rfe.ranking_

#EMBEDDED METHOD

#1.Random Forest Importance

from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier().fit(X,y_transformed)
importances=model.feature_importances_

final_df=pd.DataFrame({"Features":pd.DataFrame(X).columns,"Importances":importances})
final_df.set_index("Importances")
final_df=final_df.sort_values("Importances")
final_df.plot.bar(color="teal")
```

# OUPUT:
![p1](https://user-images.githubusercontent.com/94211349/170408527-e6e7d293-668c-4c23-b7ae-6233aca3a0f2.jpeg)
![p2](https://user-images.githubusercontent.com/94211349/170408544-24165394-ca75-4ed1-9b78-324ef4ed02e4.jpeg)
![p3](https://user-images.githubusercontent.com/94211349/170408559-ddc56b21-7ef4-458f-ae61-a827493fede4.jpeg)
![p4](https://user-images.githubusercontent.com/94211349/170408571-4232c6a7-7c6b-4b3c-9a5c-cf3920b2aba5.jpeg)
![p5](https://user-images.githubusercontent.com/94211349/170408587-7c6676aa-6d09-4e09-b91a-fd25c155d05a.jpeg)
![p6](https://user-images.githubusercontent.com/94211349/170408639-2a320a99-c3df-4c8b-9991-c8bbf8487d2b.jpeg)
![p7](https://user-images.githubusercontent.com/94211349/170408681-e53a652a-bd50-4678-900c-d0d13c6ceb54.jpeg)
## FILTER METHOD:
![p8](https://user-images.githubusercontent.com/94211349/170408708-0dac4215-2bc5-43b8-805c-4e0282f9df1e.jpeg)
## Information gain/Mutual Information:
![p9](https://user-images.githubusercontent.com/94211349/170408749-8a66f9a9-195d-4297-9d2d-1130988f05a2.jpeg)
## SelectKBest Model:
![p10](https://user-images.githubusercontent.com/94211349/170408811-b5c23421-b52f-4666-af3f-1605c9592a8d.jpeg)
## Mean Absolute Differeence:
![p11](https://user-images.githubusercontent.com/94211349/170408823-7ffbff81-65fc-4df2-8056-5c9bde9f7d7e.jpeg)
![p12](https://user-images.githubusercontent.com/94211349/170408832-87a9e4e7-748f-4f38-9bd3-ea64c27800e3.jpeg)
## Chi Square Test:
![p13](https://user-images.githubusercontent.com/94211349/170408878-4f4ab2cf-416c-406f-8e0e-420d01f77337.jpeg)
## Select Percentaile Method:
![p14](https://user-images.githubusercontent.com/94211349/170408917-c6c53903-9152-43e3-b6ce-6e200f478eed.jpeg)

## WRAPPER METHOD:

## Forward feature selection:
![p15](https://user-images.githubusercontent.com/94211349/170408935-1e04247c-fe81-4140-ba37-aa85958d7ab1.jpeg)
## Bacward feature selection:
![p16](https://user-images.githubusercontent.com/94211349/170408952-74a98f6b-735b-4897-a491-f29a82641ead.jpeg)
## Bi-directioanal elimination:
![p17](https://user-images.githubusercontent.com/94211349/170408970-0a7d72fb-96ea-45f9-9f3b-a2e884b726e5.jpeg)
## Recursive feature selection:
![p18](https://user-images.githubusercontent.com/94211349/170408992-5ed0583b-4421-4f8b-a722-74410e117cf0.jpeg)

## EMBEDDED METHOD:

## Random forest imporatance:

![p19](https://user-images.githubusercontent.com/94211349/170409010-2fbdb2de-2217-4af3-a698-e872634f98a8.jpeg)



