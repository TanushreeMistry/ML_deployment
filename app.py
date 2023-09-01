import pickle
from flask import Flask,request,app,jsonify,url_for,render_template
import matplotlib as plt

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler ,MinMaxScaler
from sklearn.ensemble import RandomForestRegressor


# Create a Flask app
app = Flask(__name__)#where flask starts
## load the model
# with open("pickle_cement.pkl","wb") as file:
#      pickle.dump(pipeline,file)

with open("pickle_cement.pkl","rb") as file:
     Randomforest = pickle.load(file)         


#Define a route for the homepage
@app.route('/')
def home():
    return render_template("home.html")


@app.route('/predict_api',methods=['POST'])
def predict_api():
    data=request.json["data"]
    print(data)
    print(np.array(list(data.values())).reshape(1,-1))
    output=Randomforest.predict((np.array(list(data.values())).reshape(1,-1)))
    print(output[0])
    return jsonify(output[0])

@app.route('/predict',methods=['POST'])
def predict():
    data=[float(x) for x in request.form.values()]
    num_rows=1
    num_col=8
    data=np.array(data).reshape(1,8)

    # final_input=standardize.transform(np.array(data).reshape(1,-1))
    # final_input=normalize.transform(np.array(data).reshape(1,-1))
    # print(final_input)

    output=Randomforest.predict(data)[0]
    return render_template("home.html",prediction_text="The compression strength of cement is {:.2f}".format(output))


if __name__ == '__main__':
    # Run the app
    app.run(debug=True,port=8000)