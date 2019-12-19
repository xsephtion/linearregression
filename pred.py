import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.linear_model import Ridge, Lasso, LinearRegression, LassoCV
from sklearn.metrics import mean_squared_error
from flask import Flask
from flask import jsonify

app = Flask(__name__)

@app.route("/")
def display():
    disp, colname = predict()
    a = []
    colname = int(colname) + 1
    a.append(colname)
    for i in disp:
        a.append(round(i))
    return jsonify(a)

def predict():
    df = pd.read_csv('training-data.csv')
    Y_target = df['TARGET']
    X_train = df.iloc[:, 1:19]
    colname = df.columns[19]
    model = Lasso()
    model.fit(X_train, Y_target)
    df_test = pd.read_csv('testing-data.csv')
    test_d = df_test.iloc[:, 1:19]
    pred = model.predict(test_d)
    print(colname)
    return pred, colname

if __name__ == "__main__":
    predict()
    app.run(debug=True)
