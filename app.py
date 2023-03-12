from flask import Flask, make_response, request, app, jsonify, url_for, render_template
import io
from io import StringIO
import csv
import pandas as pd
import numpy as np
import pickle
import os



app = Flask(__name__)

IMG_FOLDER=os.path.join('static','IMG')
app.config['UPLOAD_FOLDER']=IMG_FOLDER

##Load the model
model=pickle.load(open('svm_voice_model.pkl', 'rb')) #rb
scalar=pickle.load(open('scaling_voice.pkl', 'rb')) #rb

def transform(text_file_contents):
    return text_file_contents.replace("=", ",")


@app.route('/')
def home():
    return render_template("index.html")
      

@app.route('/predict', methods=['POST'])
def predict():

    f = request.files['data_file']
    if not f:
        return "No file"

    stream = io.StringIO(f.stream.read().decode("UTF8"), newline=None)
    csv_input = csv.reader(stream)
    #print("file contents: ", file_contents)
    #print(type(file_contents))
    print(csv_input)
    for row in csv_input:
        print(row)

    stream.seek(0)
    result = transform(stream.read())

    df = pd.read_csv(StringIO(result))

    #data1=[float(x) for x in request.form.values()]
    #data = np.array([data1])

    final_input=scalar.transform(np.array(df).reshape(1,-1))
    print(final_input)
    
    prediction = model.predict(final_input)
    image=str(prediction[0])+'.png'
    image=os.path.join(app.config['UPLOAD_FOLDER'],image)

    

    if prediction[0]==1:
        prediction_voice="It's a Male Voice"
    else:
        prediction_voice="It's a Female Voice"

    return render_template("index.html", prediction=prediction_voice, image=image )  


if __name__ == "__main__":
    app.run(debug=True)