# GRU Model of Prediction
import numpy as np 
import math
import pandas as pd 
from math import sqrt
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import GRU
import keras
from sklearn.preprocessing import MinMaxScaler
df = pd.read_csv(r'C:\Users\user\Desktop\bitstampUSD_1-mindata_.csv')
df['date'] = pd.to_datetime(df['Timestamp'],unit='s').dt.date
group = df.groupby('date')
Real_Price = group['Weighted Price'].mean()
df.head()
prediction_days=240
df_train= Real_Price[:len(Real_Price)-prediction_days]
df_test= Real_Price[len(Real_Price)-prediction_days:]
training_set = df_train.values
training_set = np.reshape(training_set, (len(training_set), 1))
sc = MinMaxScaler(feature_range=(0,1))
training_set = sc.fit_transform(training_set)
X_train = training_set[0:len(training_set)-1]
y_train = training_set[1:len(training_set)]
X_train = np.reshape(X_train, (len(X_train), 1, 1))
regressor = Sequential()
regressor.add(keras.layers.GRU(units = 75, activation='relu',return_sequences=True,input_shape = (None, 1)))
regressor.add(keras.layers.GRU(units=30,return_sequences=True))
regressor.add(keras.layers.GRU(units=30))
regressor.add(Dense(units = 1))
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')
regressor.fit(X_train, y_train, epochs =50, batch_size=8, verbose=2)
regressor.summary()
testpredict= regressor.predict(X_train,batch_size=8)
score= regressor.evaluate(X_train,y_train,batch_size=8,verbose=2)
MSE=score
RMSE=math.sqrt(score)
print('Test RMSE:%.3f'%RMSE)
test_set = df_test.values
inputs = np.reshape(test_set, (len(test_set), 1))
inputs = sc.transform(inputs)
inputs = np.reshape(inputs, (len(inputs), 1, 1))
predicted_BTC_price = regressor.predict(inputs)
predicted_BTC_price = sc.inverse_transform(predicted_BTC_price)
plt.figure(figsize=(10,5), dpi=80, facecolor='w', edgecolor='k')
ax = plt.gca()  
plt.plot(test_set, color = 'red', label = 'Real BTC Price')
plt.plot(predicted_BTC_price, color = 'blue', label = 'Predicted BTC Price')
plt.title('BTC Price Prediction', fontsize=40)
df_test = df_test.reset_index()
x=df_test.index
labels = df_test['date']
plt.xticks(x, labels, rotation = 'vertical')
for tick in ax.xaxis.get_major_ticks():
    tick.label1.set_fontsize(6)
for tick in ax.yaxis.get_major_ticks():
    tick.label1.set_fontsize(18)
plt.xlabel('Time', fontsize=5)
plt.ylabel('BTC Price(USD)', fontsize=20)
plt.legend(loc=2, prop={'size': 25})
plt.show()
# Saving model to disk
pickle.dump(predicted_BTC_price, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[2, 9, 6, 8]]))

app.py
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')
@app.route('/predict',methods=['POST'])
def predict():
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
     output = round(prediction[0], 2)
return render_template('index.html', prediction_text='Bitcoin price should be $ {}'.format(output))
@app.route('/predict_api',methods=['POST'])
def predict_api():
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])
    output = prediction[0]
    return jsonify(output)
if __name__ == "__main__":
    app.run(debug=True)

Request.py
import requests
url = 'http://localhost:5000/predict_api'
r = requests.post(url)
print(r.json())

Index.html
<!DOCTYPE html>
<html >
<!--From https://codepen.io/frytyler/pen/EGdtg-->
<head>
  <meta charset="UTF-8">
  <title>Bitcoin Price Prediction API</title>
  <linkhref='https://fonts.googleapis.com/css?family=Pacifico'rel='stylesheet' type='text/css'>
<linkhref='https://fonts.googleapis.com/css?family=Arimo'rel='stylesheet' type='text/css'>
<linkhref='https://fonts.googleapis.com/css?family=Hind:300'rel='stylesheet' type='text/css'>
<linkhref='https://fonts.googleapis.com/css?family=Open+Sans+Condensed:300' rel='stylesheet' type='text/css'>
<link rel="stylesheet" href="{{ url_for('static', filename='css/style.css') }}">
  </head>
<body>
 <div class="login">
	<h1>CryptoReady</h1>
  <!-- Main Input For Receiving Query to our ML -->
    <form action="{{ url_for('predict')}}"method="post">
    	<input type="text" name="Open Price" placeholder="Open Price" required="required" />
        <input type="text" name="High Price" placeholder="High Price" required="required" />
<input type="text" name="Low Price" placeholder="Low Price" required="required" />
		<input type="text" name="Close Price" placeholder="Close Price" required="required" />
<button type="submit" class="btn btn-primary btn-block btn-large">Predict</button>
    </form>
   <br>
   <br>
   {{ prediction_text }}
 </div>
</body>
</html>








