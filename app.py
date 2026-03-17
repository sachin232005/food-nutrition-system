import streamlit as st
import pandas as pd
import pickle

# Load model and scaler
model = pickle.load(open("food_model.pkl","rb"))
scaler = pickle.load(open("scaler.pkl","rb"))

# Load dataset to get correct column order
df = pd.read_csv("cleaned_food_dataset.csv")

# Get training feature columns
feature_columns = df.drop(['Calories (kcal)','Dish Name','Carbohydrates (g)'],axis=1).columns

st.title("🍲 Indian Food Nutrition Predictor")

# User Inputs
protein = st.number_input("Protein (g)",0.0)
fats = st.number_input("Fats (g)",0.0)
fibre = st.number_input("Fibre (g)",0.0)
free_sugar = st.number_input("Free Sugar (g)",0.0)
calcium = st.number_input("Calcium (mg)",0.0)
iron = st.number_input("Iron (mg)",0.0)
sodium = st.number_input("Sodium (mg)",0.0)
vitamin_c = st.number_input("Vitamin C (mg)",0.0)
folate = st.number_input("Folate (µg)",0.0)

# Create input dictionary
input_dict = {
    'Protein (g)':protein,
    'Fats (g)':fats,
    'Fibre (g)':fibre,
    'Free Sugar (g)':free_sugar,
    'Calcium (mg)':calcium,
    'Iron (mg)':iron,
    'Sodium (mg)':sodium,
    'Vitamin C (mg)':vitamin_c,
    'Folate (µg)':folate
}

# Convert to DataFrame
input_data = pd.DataFrame([input_dict])

# Reorder columns to match training data
input_data = input_data[feature_columns]

# Scale
scaled_data = scaler.transform(input_data)

# Predict
if st.button("Predict"):
    prediction = model.predict(scaled_data)
    st.success(f"Carbohydrate Level: {prediction[0]}")
