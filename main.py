from flask import Flask, render_template, request
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

# 모델과 전처리 함수를 로드
data = pd.read_csv('heart.csv')
features = [
    'Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol', 'FastingBS',
    'RestingECG', 'MaxHR', 'ExerciseAngina', 'Oldpeak', 'ST_Slope'
]
X = data[features].copy()
y = data['HeartDisease']


# 데이터 전처리 함수 정의
def preprocess_data(df):
    df.loc[:, 'Sex'] = df['Sex'].map({'M': 0, 'F': 1})
    df.loc[:, 'ChestPainType'] = df['ChestPainType'].map({
        'ASY': 0,
        'ATA': 1,
        'NAP': 2,
        'TA': 3
    })
    df.loc[:, 'RestingECG'] = df['RestingECG'].map({
        'LVH': 0,
        'Normal': 1,
        'ST': 2
    })
    df.loc[:, 'ExerciseAngina'] = df['ExerciseAngina'].map({'N': 0, 'Y': 1})
    df.loc[:, 'ST_Slope'] = df['ST_Slope'].map({'Down': 0, 'Up': 1, 'Flat': 2})
    df.loc[:, 'FastingBS'] = df['FastingBS'].map({0: 0, 1: 1})
    return df


X = preprocess_data(X)

model = RandomForestClassifier()
model.fit(X, y)


# Define the route for the home page
@app.route('/')
def home():
    return render_template('index.html')


# Define the route for the home page
@app.route('/')
def index():
    return render_template('index.html')


# Define the route for the prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Get the input values from the form
    age = int(request.form['age'])
    sex = request.form['sex']
    chest_pain = request.form['chest_pain']
    resting_bp = int(request.form['resting_bp'])
    cholesterol = int(request.form['cholesterol'])
    fasting_bs = int(request.form['fasting_bs'])
    resting_ecg = request.form['resting_ecg']
    max_hr = int(request.form['max_hr'])
    exercise_angina = request.form['exercise_angina']
    oldpeak = float(request.form['oldpeak'])
    st_slope = request.form['st_slope']

    # Map categorical values to numerical representations
    sex_mapped = 0 if sex == 'M' else 1
    chest_pain_mapped = {'ASY': 0, 'ATA': 1, 'NAP': 2, 'TA': 3}[chest_pain]
    resting_ecg_mapped = {'LVH': 0, 'Normal': 1, 'ST': 2}[resting_ecg]
    exercise_angina_mapped = {'N': 0, 'Y': 1}[exercise_angina]
    st_slope_mapped = {'Down': 0, 'Up': 1, 'Flat': 2}[st_slope]

    # Create a DataFrame with the input values
    input_data = pd.DataFrame([[
        age, sex_mapped, chest_pain_mapped, resting_bp, cholesterol,
        fasting_bs, resting_ecg_mapped, max_hr, exercise_angina_mapped,
        oldpeak, st_slope_mapped
    ]],
                              columns=features)

    # Make the prediction
    prediction = model.predict(input_data)
    proba = model.predict_proba(input_data)[0]

    if prediction[0] == 1:
        result = f"심장병이 의심됩니다! (확률: {proba[1]*100:.2f}%)"
    else:
        result = f"심장병확률은 낮지만 조심하세요! (확률: {proba[1]*100:.2f}%)"

    # Return the result as a string
    return result


if __name__ == '__main__':
    app.run('0.0.0.0', port=5000, debug=True)
