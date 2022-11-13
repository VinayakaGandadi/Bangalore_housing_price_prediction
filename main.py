import pandas as pd
import pickle
from joblib import dump, load

from flask import Flask ,render_template,request

app=Flask(__name__)
data=pd.read_csv('cleaned.csv')
#pipe = load('linearegression.joblib')
pipe=pickle.load(open('linearegression.pkl','rb'))
@app.route('/')
def index():
    locations=sorted(data['location'].unique())
    return render_template('index.html',locations=locations)
@app.route('/predict',methods=['POST'])
def predict():
    location = request.form.get('location')
    bhk = request.form.get('bhk')
    bath = request.form.get('bathrooms')
    sqft = request.form.get('sqft')
    print(location, bhk, bath, sqft)
    input =pd.DataFrame([[location, float(sqft), float(bath), float(bhk)]],columns=['location','total_sqft','bath','bedrooms'])
    prediction=pipe.predict(input)[0]
    print(prediction)
    return str(prediction)
if __name__ == '__main__':
    app.run(debug=True, port=50001)
