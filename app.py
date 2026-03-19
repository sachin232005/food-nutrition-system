import streamlit as st
import pandas as pd
import pickle
import psycopg2
import os

# -------------------------------
# LOAD MODEL & DATA
# -------------------------------
model = pickle.load(open("food_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

df = pd.read_csv("cleaned_food_dataset.csv")
feature_columns = df.drop(['Calories (kcal)','Dish Name','Carbohydrates (g)'], axis=1).columns

# -------------------------------
# DATABASE CONNECTION (LOCAL + CLOUD)
# -------------------------------
@st.cache_resource
def get_connection():
    try:
        # 🌐 Cloud (Render)
        DATABASE_URL = os.getenv("DATABASE_URL")
        if DATABASE_URL:
            return psycopg2.connect(DATABASE_URL)
    except:
        pass

    # 💻 Local PostgreSQL
    return psycopg2.connect(
        host="localhost",
        database="food_db",
        user="postgres",
        password="postgres",
        port="5432"
    )

try:
    conn = get_connection()
    cursor = conn.cursor()
except Exception as e:
    st.error(f"Database connection failed: {e}")

# -------------------------------
# UI
# -------------------------------
st.title("🍲 Indian Food Nutrition Predictor")

protein = st.number_input("Protein (g)", 0.0)
fats = st.number_input("Fats (g)", 0.0)
fibre = st.number_input("Fibre (g)", 0.0)
free_sugar = st.number_input("Free Sugar (g)", 0.0)
calcium = st.number_input("Calcium (mg)", 0.0)
iron = st.number_input("Iron (mg)", 0.0)
sodium = st.number_input("Sodium (mg)", 0.0)
vitamin_c = st.number_input("Vitamin C (mg)", 0.0)
folate = st.number_input("Folate (µg)", 0.0)

# -------------------------------
# PREPARE INPUT
# -------------------------------
input_dict = {
    'Protein (g)': protein,
    'Fats (g)': fats,
    'Fibre (g)': fibre,
    'Free Sugar (g)': free_sugar,
    'Calcium (mg)': calcium,
    'Iron (mg)': iron,
    'Sodium (mg)': sodium,
    'Vitamin C (mg)': vitamin_c,
    'Folate (µg)': folate
}

input_data = pd.DataFrame([input_dict])
input_data = input_data[feature_columns]

# -------------------------------
# MAPPING
# -------------------------------
prediction_mapping = {'Low': 1, 'Medium': 2, 'High': 3}
reverse_mapping = {1: 'Low', 2: 'Medium', 3: 'High'}

# -------------------------------
# PREDICT & SAVE
# -------------------------------
if st.button("Predict"):
    try:
        scaled_data = scaler.transform(input_data)
        prediction = model.predict(scaled_data)

        # Convert label → number
        prediction_numeric = prediction_mapping.get(prediction[0], 0)

        # Save to PostgreSQL
        cursor.execute(
            "INSERT INTO predictions (protein, fats, fibre, prediction) VALUES (%s, %s, %s, %s)",
            (protein, fats, fibre, prediction_numeric)
        )
        conn.commit()

        st.success(f"Carbohydrate Level: {prediction[0]}")

    except Exception as e:
        st.error(f"Error: {e}")

# -------------------------------
# SHOW HISTORY
# -------------------------------
if st.button("Show History"):
    try:
        cursor.execute("SELECT * FROM predictions")
        data = cursor.fetchall()

        df_history = pd.DataFrame(
            data,
            columns=["ID", "Protein", "Fats", "Fibre", "Prediction"]
        )

        # Convert number → label
        df_history["Prediction"] = df_history["Prediction"].map(reverse_mapping)

        st.dataframe(df_history)

    except Exception as e:
        st.error(f"Error fetching data: {e}")