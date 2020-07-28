import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import os
import xlrd
import locale
from locale import atof
locale.setlocale( locale.LC_ALL, 'en_US.UTF-8' )
regressor = pickle.load(open('model.pkl','rb'))
cv=pickle.load(open('cv.pkl','rb'))
le=pickle.load(open('le.pkl','rb'))

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')

def ashx_f(description):
    '''function determining the salary or not from the inputted text'''
    description1=[description]
    description1=cv.transform(description1)
    if le.inverse_transform(regressor.predict(description1))[0]=='True':
        return "Salary program"
    else:
        return "Not salary program"
def ashx_chap(x):

    arra=x["description"].to_numpy()
    amountt=x["amount"].to_numpy()
    pordz_test = list(map(lambda x: ashx_f(x), arra))
    dicti={"Description":arra,"Salary":pordz_test,"Amount":amountt}
    pordz_df=pd.DataFrame(dicti)
    try:
        salary="Հաճախորդի մեր բանկով ստացված աշխատավարձը կազմում է "+str(round(pordz_df.groupby('Salary')['Amount'].sum()['Salary program']/6,1))+" դրամ"
        return salary
    except:
        x="Հաճախորդը մեր բանկով աշխատավարձ չի ստանում"
        return x


@app.route('/predict',methods=['POST'])
def predict():
    file=request.files["Հաճախորդի վերջին 6 ամսվա քաղվածք"]
    file.save(os.path.join('uploads', file.filename))
    path="./uploads/"+file.filename
    wb = xlrd.open_workbook(path, logfile=open(os.devnull, 'w'))
    excel=pd.read_excel(wb)
    if excel.iloc[23,8]=='Մուտք':
        excel=excel.drop(excel.index[:24])
        excel=excel.iloc[:,[8,22]]
    else:
        excel=excel.drop(excel.index[:32])
        excel=excel.iloc[:,[7,24]]
    excel.columns=['amount','description']
    excel["amount"] = excel['amount'].str.replace('[+ ]','')
    excel["description"]=excel.description.astype(str)

    excel["amount"]=excel.amount.astype(str)
    excel["amount"]=excel.amount.apply(atof)
    output=ashx_chap(excel)
    return render_template('index.html', prediction_text='{}'.format(output))





if __name__ == "__main__":
    app.run(debug=True)
