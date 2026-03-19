import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import pickle

# Load dataset
df = pd.read_csv(r"C:\Users\LENOVO\Downloads\Indian_Food_Nutrition_Processed.csv")
print(df.head())

# Data Cleaning
df['Vitamin C (mg)'] = df['Vitamin C (mg)'].fillna(df['Vitamin C (mg)'].median())
df['Folate (µg)'] = df['Folate (µg)'].fillna(df['Folate (µg)'].median())

df.fillna(df.mean(numeric_only=True), inplace=True)

# Save cleaned dataset
df.to_csv("cleaned_food_dataset.csv", index=False)

# Visualization
df['Calories (kcal)'].hist()
plt.show()

plt.figure(figsize=(10,6))
sns.heatmap(df.drop('Dish Name', axis=1).corr(), annot=True)
plt.show()

sns.boxplot(x=df['Calories (kcal)'])
plt.show()

# Features and target
X = df.drop(['Calories (kcal)', 'Dish Name', 'Carbohydrates (g)'], axis=1)

y = pd.cut(
    df['Carbohydrates (g)'],
    bins=[0,10,25,100],
    labels=['Low','Medium','High'],
    include_lowest=True
)

# Train test split
X_train,X_test,y_train,y_test = train_test_split(
    X,y,test_size=0.2,random_state=42
)

# Scaling
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Model
model = SVC(kernel='rbf')

model.fit(X_train,y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test,y_pred)

print("Accuracy:",accuracy)

print(classification_report(y_test,y_pred))

# Save model
with open("food_model.pkl","wb") as file:
    pickle.dump(model,file)
    pickle.dump(scaler, open("scaler.pkl", "wb"))