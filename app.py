import joblib
import numpy as np
from flask_cors import CORS

from flask import Flask, request, jsonify

# Load trained model
model = joblib.load("chennai_property_model.pkl")


def investment_score(sqft, bathrooms, bhk, location=None, actual_price=None):
    input_data = np.array([[sqft, bathrooms, bhk]])
    predicted_price = model.predict(input_data)[0]  # in lakhs

    price_per_sqft = (predicted_price * 100000) / sqft

    result = {
        "location": location,
        "predicted_price_lakhs": round(predicted_price, 2),
        "predicted_price_cr": round(predicted_price / 100, 2),
        "price_per_sqft": round(price_per_sqft, 2),
    }

    if actual_price is not None:
        price_difference = predicted_price - actual_price
        percent_diff = (price_difference / actual_price) * 100

        if percent_diff > 10:
            value_tag = "Underpriced"
            recommendation = "Good Investment"
        elif percent_diff < -10:
            value_tag = "Overpriced"
            recommendation = "Not Recommended"
        else:
            value_tag = "Fairly Priced"
            recommendation = "Average Deal"

        result.update({
            "actual_price_lakhs": actual_price,
            "price_difference_lakhs": round(price_difference, 2),
            "price_difference_percent": round(percent_diff, 2),
            "value_opportunity": value_tag,
            "recommendation": recommendation
        })

    score = 50

    if sqft > 2000:
        score += 10
    elif sqft < 900:
        score -= 5

    if bhk >= 3:
        score += 10

    if bathrooms >= bhk:
        score += 5

    score = max(0, min(100, score))

    result["investment_score"] = score
    result["score_breakdown"] = {
        "base": 50,
        "size_bonus": 10 if sqft > 2000 else (-5 if sqft < 900 else 0),
        "bhk_bonus": 10 if bhk >= 3 else 0,
        "bathroom_bonus": 5 if bathrooms >= bhk else 0
    }

    return result


app = Flask(__name__)
CORS(app)



@app.route('/predict', methods=['POST'])
def predict():
    data = request.json

    sqft = data['sqft']
    bathrooms = data['bathrooms']
    bhk = data['bhk']
    location = data.get('location')
    actual_price = data.get('actual_price')

    result = investment_score(
        sqft=sqft,
        bathrooms=bathrooms,
        bhk=bhk,
        location=location,
        actual_price=actual_price
    )

    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True)
