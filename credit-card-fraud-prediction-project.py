import numpy as np
import pandas as pd
from pandas import Series, DataFrame

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

import statsmodels.api as sm 
import statsmodels.formula.api as smf

from datetime import datetime

from sklearn.metrics import roc_curve, auc, classification_report

#dataset link (from kaggle): https://www.kaggle.com/datasets/kelvinkelue/credit-card-fraud-prediction



df = pd.read_csv('C:/Users/samue/OneDrive/Desktop/PROG/fraud_prediction/credit_card_data.csv', sep=',')
df.head(5)
print(df.columns)
print(df.dtypes)



#useless columns were dropped
data = df.drop(columns=['Unnamed: 0','trans_date_trans_time','cc_num','merchant','first','last','street','city','state','zip','lat','long','job','trans_num',
                        'unix_time','merch_lat','merch_long'])


data['gender'] = data['gender'].map({'F':0, 'M':1}) #females=0 , males=1
data['dob'] = pd.to_datetime(data['dob'])
today = datetime.now()
data['age'] = data['dob'].apply(lambda x: today.year - x.year) #cardholder's date of birth to age
data = data.drop(columns=['dob'])

category_values = data['category'].unique()
print('Categories:', category_values)
category_mapping = {            #mapping dictionary to relate each business category with an int value
    'personal_care': 1,
    'health_fitness': 2,
    'misc_pos': 3,
    'travel': 4,
    'kids_pets': 5,
    'shopping_pos': 6,
    'food_dining': 7,
    'home': 8,
    'entertainment': 9,
    'shopping_net': 10,
    'misc_net': 11,
    'grocery_pos': 12,
    'gas_transport': 13,
    'grocery_net': 14
}
data['category'] = data['category'].map(category_mapping)

fraud = data['is_fraud']
data['fraud'] = fraud  #fraud column at the end of the DataFrame
data = data.drop(columns=['is_fraud'])

data.rename(columns={'amt':'amount'}, inplace=True)

#columns were dropped after testing the predictive model
data = data.drop(columns=['gender'])
data = data.drop(columns=['city_pop'])



#classification - logistic regression
y = data['fraud']
x = data.drop(columns=['fraud'])
x = sm.add_constant(x, prepend=False)
model1 = sm.Logit(y,x)
model1_fitted = model1.fit()
print(model1_fitted.summary())  #"category", "amount", and "age" have a  highly significant relationship with the probability of fraud


#train & test
data_train = data.copy().sample(frac=0.8)
data_test = data.copy().drop(data_train.index)

#response variable
y_train = data_train['fraud']
y_test = data_test['fraud']


#predictor variable
x_train = data_train.drop(columns=['fraud'])
x_test = data_test.drop(columns=['fraud'])
#constant
x_train = sm.add_constant(x_train, prepend=False)
x_test = sm.add_constant(x_test, prepend=False)


#regression model
model2 = sm.Logit(y_train, x_train)
model2_fitted = model2.fit()
print(model2_fitted.summary())  #"category", "amount", and "age" remain highly significant


#prediction
prediction = model2_fitted.predict(x_test)
print(round(prediction.head(15),3))
sorted_predictions = prediction.sort_values(ascending=False)
print(round(sorted_predictions.head(15),3))




#Youden index & ROC curve were used to determine the best threshold
fpr, tpr, thresholds = roc_curve(y_test, prediction)
roc_auc = auc(fpr, tpr)

youden_index = tpr - fpr
best_threshold_index = np.argmax(youden_index)
best_threshold = thresholds[best_threshold_index]
print(f"The best Youden index: {best_threshold}")

plt.figure()
plt.plot(fpr, tpr, color='red', lw=2, label='ROC curve (area = %0.3f)' % roc_auc)
plt.plot([0,1],[0,1], color='blue', lw=2, linestyle='--')
plt.scatter(fpr[best_threshold_index], tpr[best_threshold_index], color='green', zorder=5)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.show()



#confusion matrix
pred_1_0 = [1 if p>=0.00694 else 0 for p in prediction]
pd.crosstab(index=y_test, columns=pd.Categorical(pred_1_0), rownames=['Obs'], colnames=['Pred'])

TN = 104210  # True Negative
FP = 6492   # False Positive
FN = 127     # False Negative
TP = 315    # True Positive
#accuracy = 94.1% ((TP + TN) / (TP + TN + FP +FN))
#precision = 4.6% (TP / (TP + FP))
#recall = 71.3% (TP / (TP + FN))
#specificity = 94.1% (TN / (TN + FP))

#For the given sample (555,719 users), the model has high recall (good fraud detection capacity). However, it has low precision, meaning there are too many false positives. 
# This can be acceptable depending on the context and the resources available to investigate the fraud cases.
