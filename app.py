import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
plt.style.use('ggplot') 

df =pd.read_csv("healthcare-dataset-stroke-data.csv")
df.head()

df.dtypes

df.isna().sum()

for col in df.columns:
    if df[col].dtype=='object':
        print(col)
        print(df[col].unique())

df.drop('id' , axis = 1 , inplace = True)

df.shape

df.describe().T

df.bmi.fillna(df.bmi.mean(),inplace=True)

sns.pairplot(df,hue="stroke")

numerical_data = df[['age','avg_glucose_level','bmi']]

sns.kdeplot(data=numerical_data)

sns.countplot(x ='stroke' , data = df).set_title('Count of Stroke Patients')

sns.countplot(x='gender', hue='stroke', data=df).set_title('Stroke patients by Gender')

sns.countplot(x='ever_married', hue='stroke', data=df).set_title('Stroke patients By Marital Status')

sns.countplot(x='smoking_status',hue='stroke',data=df).set_title("Stroke Patients By Smoking Status")

sns.countplot(x='work_type',hue='stroke',data=df).set_title("Stroke Patients By Work type")

fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(16, 12))

sns.boxplot(df['age'], ax=axes[0, 0]).set_title("**BoxPlot For Age Col**")
sns.histplot(data=df, x='age', kde=True, ax=axes[0, 1]).set_title("**Distribution Of Age**")
sns.lineplot(data=df, x='age', y="stroke", ax=axes[0, 2]).set_title('**Lineplot For Age with Stroke**')

sns.boxplot(df['avg_glucose_level'], ax=axes[1, 0]).set_title("BoxPlot For Glucose")
sns.histplot(data=df, x='avg_glucose_level', kde=True, ax=axes[1, 1]).set_title("Distribution Of Glucose")
sns.lineplot(data=df, x='avg_glucose_level', y="stroke", ax=axes[1, 2]).set_title('**Lineplot For Glucose Level With Stroke')

sns.boxplot(df['bmi'], ax=axes[2, 0]).set_title("BoxPlot For Bmi Col")
sns.histplot(data=df, x='bmi', kde=True, ax=axes[2, 1]).set_title("Distribution Of Bmi")
sns.lineplot(data=df, x='bmi', y="stroke", ax=axes[2, 2]).set_title('Lineplot For Bmi With Stroke')

plt.tight_layout()
plt.show()

corr = df.select_dtypes(include='number').corr()
plt.figure(figsize=(12,6))
sns.heatmap(corr,annot=True)

for col in df.columns:
    if df[col].dtype=='object':
        print(col)
        print(df[col].unique())
    
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
for col in df.columns:
    if df[col].dtype=='object':
        df[col]=le.fit_transform(df[col]) 
    
df.dtypes

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
for col_name in df.columns:
    if df[col_name].nunique() > 5: 
        df[col_name] = scaler.fit_transform(df[[col_name]])
    
X = df.drop("stroke",axis=1)
y =df['stroke']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42,shuffle=True)

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X_train, y_train)

y_pred_lr = lr.predict(X_test)

from sklearn.metrics import accuracy_score

# Assuming y_test and y_pred_lr are already defined
accuracy_using_logistic_regression = round(accuracy_score(y_test, y_pred_lr) * 100, 2)
print("Model accuracy using Logistic Regression: ", accuracy_using_logistic_regression, "%")

import joblib

# Save the model
joblib.dump(lr, 'stroke_prediction_model.pkl')

# Load the model later when needed
loaded_model = joblib.load('stroke_prediction_model.pkl')

# Make predictions using the loaded model
y_pred_loaded = loaded_model.predict(X_test)

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Compute the confusion matrix
cm = confusion_matrix(y_test, y_pred_lr)

# Plot the heatmap
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.title("Logistic Regression")
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()