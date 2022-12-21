import pandas as pd
import numpy as np 
import json
from flask import Flask ,request,app,jsonify,url_for,render_template
import pickle

app= Flask(__name__)
## Load the model 
Lr_model=pickle.load(open('LR_Model.pkl','rb'))

@app.route('/')
def home():
    return render_template('home.html')
#send a request to our app

#voir si faut ajouter transfromer au pickle

#def model_final(model=Lr_model,X=np.asarray([1,2,3,4,5,6,7,8,9,10]),threshold=0):
   # zz=model.decision_function(X)> threshold
    #return zz

#entree nom du model,donnee sous forme de tableau une ligne ,le threshold trouve 
#def prediction(model,data,threshold):
   # data=np.asarray(data).reshape(1,-1)
    #resultat=model_final(model,data,threshold)
   # if (resultat==True):
        #return 2
    #else:
        #return 1  
@app.route('/predict_api',methods=['GET', 'POST'])
def predict_api():
    data=request.json['data']
    print(data)
    print(np.array(list(data.values())).reshape(1,-1))
    new_data=np.array(list(data.values())).reshape(1,-1)#voir si c necessaire ca
    #voir a ce niveau si les donnees seront dans le bon format (transformers dessus)
    output=Lr_model.predict(new_data)
    return jsonify(int(output[0]))
    
@app.route('/predict',methods=['POST'])
def predict():
    data=[float(x) for x in request.form.values()]
    final_input=np.array(data).reshape(1,-1)
    print(final_input)
    output=Lr_model.predict(final_input)[0]
    print(output)
    if output ==2:
     return render_template("home.html",prediction_text="Vous avez une maladie cardiovasculaire")
    else:
        return render_template("home.html",prediction_text="Vous n'avez pas de  maladie cardiovasculaire")


if __name__=="__main__":
    app.run(debug=True)