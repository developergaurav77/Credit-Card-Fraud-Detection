
from flask import Flask, render_template, redirect, url_for,request
import numpy as np
import joblib

model = joblib.load('lr_model.pkl')
app = Flask(__name__)

@app.route('/<result>')
def predict(result):
    return f"<h1> Credit Card Status(0 for legit and 1 for fraud maybe):{result}</h1>"

@app.route('/',methods=['POST','GET'])
def home():
    if request.method == 'POST':
        msg = request.form['message']
        message = [float(x) for x in msg.split()]
        content = np.array(message).reshape(1, -1)
        prediction = model.predict(content)
        return redirect(url_for("predict",result=int(prediction)))

    else:
        return render_template('home.html')

if __name__ == '__main__':
    app.run()
