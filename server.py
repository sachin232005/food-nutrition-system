from flask import Flask, request, jsonify
import pickle
import psycopg2

app = Flask(__name__)

# 🔐 API KEY
API_KEY = "mysecret123"

# Load model
model = pickle.load(open("food_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# DB connection
conn = psycopg2.connect(
    host="localhost",
    database="food_db",
    user="postgres",
    password="postgres",
    port="5432"
)
conn.autocommit = True
cursor = conn.cursor()

# -------------------------------
# PREDICT API
# -------------------------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # 🔐 Check API key
        key = request.headers.get("x-api-key")
        if key != API_KEY:
            return jsonify({"error": "Unauthorized"}), 401

        data = request.json

        input_data = [[
            data["protein"],
            data["fats"],
            data["fibre"]
        ]]

        scaled = scaler.transform(input_data)
        prediction = model.predict(scaled)[0]

        # Mapping
        mapping = {'Low': 1, 'Medium': 2, 'High': 3}
        prediction_numeric = mapping.get(prediction, 0)

        # Save to DB
        cursor.execute(
            "INSERT INTO predictions (protein, fats, fibre, prediction) VALUES (%s, %s, %s, %s)",
            (data["protein"], data["fats"], data["fibre"], prediction_numeric)
        )

        return jsonify({"prediction": prediction})

    except Exception as e:
        conn.rollback()
        return jsonify({"error": str(e)})

# -------------------------------
# HISTORY API
# -------------------------------
@app.route("/history", methods=["GET"])
def history():
    try:
        key = request.headers.get("x-api-key")
        if key != API_KEY:
            return jsonify({"error": "Unauthorized"}), 401

        cursor.execute("SELECT * FROM predictions ORDER BY id DESC")
        data = cursor.fetchall()

        reverse = {1: 'Low', 2: 'Medium', 3: 'High'}

        result = []
        for row in data:
            result.append({
                "id": row[0],
                "protein": row[1],
                "fats": row[2],
                "fibre": row[3],
                "prediction": reverse.get(row[4], "Unknown")
            })

        return jsonify(result)

    except Exception as e:
        conn.rollback()
        return jsonify({"error": str(e)})

# -------------------------------
# RUN SERVER
# -------------------------------
if __name__ == "__main__":
    app.run(debug=True)