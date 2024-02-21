from flask import Flask,jsonify,request
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

data = pd.read_csv('balanced_dataset_smote.csv')
X = data.drop('company_name', axis=1)
y = data['company_name']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

@app.route('/predict',methods=['POST'])
def predict():
    try:
        new_user_features = request.json
        new_user_df = pd.DataFrame([new_user_features])

        predicted_company = model.predict(new_user_df)
        return jsonify({'predicted_company': predicted_company[0]})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == "__main__":
    app.run(debug=True,port=8080)
