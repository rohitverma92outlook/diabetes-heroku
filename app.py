from flask import Flask, render_template, request
import pickle

model = pickle.load(open("model_diabetes.pkl", "rb"))

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/diabetes_prediction", methods = ["POST","GET"])
def diabetes_prediction():
    input_val = [[float (X) for X in request.form.values()]]
    output = model.predict(input_val)
    prob = model.predict_proba(input_val)
    return render_template("diabetes_prediction.html", prediction = output, probabilty = prob)



if __name__ == "__main__":
    app.run(debug=True)