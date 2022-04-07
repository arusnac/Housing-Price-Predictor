from flask import Flask, redirect, render_template, request, redirect, session
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt2
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn import preprocessing
from sklearn.compose import ColumnTransformer
#from sklearn.linear_model import Lasso
from sklearn import svm
from sklearn import linear_model
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import seaborn as sns
import logging

default_username = 'admin'
default_password = 'c964'

app = Flask(__name__, static_url_path='/static')

def log_error(Argument):
    file = open('log.txt', 'a')
    file.write(str(Argument))
    file.write('\n')
    file.close

#Read the csv and place it into a dataframe, if the file isn't found log the error in log.txt
try:
    portland_housing = pd.read_csv('portland_housing.csv', low_memory=False)
except Exception as Argument:
    log_error(Argument)

portland_housing = portland_housing[['zipcode', 'bathrooms', 'bedrooms', 'daysOnZillow', 'homeType', 
                                     'lastSoldPrice', 'livingArea', 'lotSize', 'price', 'flooring0', 
                                     'hasGarage', 'hasView', 'schoolRating0', 'schoolRating1', 'schoolRating2', 
                                     'yearBuilt']]


#portland_housing = portland_housing.drop(portland_housing.index[5000:])
df_to_table = portland_housing.drop(portland_housing.index[31:])
portland_housing = portland_housing.dropna()

#Plot the bar chart with zip code on the x axis and price on the y axis
#Save each chart as a png to be displayed on the data.html pagae
plt.figure(figsize=(15, 10))
p = sns.barplot('zipcode', 'price', data=portland_housing, ci=False)
p.set_xticklabels(p.get_xticklabels(),rotation = 90)
plt.savefig('static/category.png', bbox_inches='tight')

#Scatter with square footage of the home as the x variable and price as the y variable
plt2.figure(figsize=(15, 12))
sns.regplot(x='livingArea', y='price', data=portland_housing)
plt2.savefig('static/scatter.png', bbox_inches='tight')
    
#Plot the categorical box chart with school rating on the x axis and price on the y axis
plt.figure(figsize=(15,10))
sns.catplot(x='schoolRating0', y='price', kind='box', data=portland_housing )
    
plt.savefig('static/bar_graph.png', bbox_inches='tight')

#Divide the data into features and the target price
X = portland_housing.drop('price', axis=1)
y = portland_housing['price']

#Convert all strings into numerical values so that they can be fit into the model
categorize = ['daysOnZillow','schoolRating0', 'schoolRating1', 'schoolRating2', 'homeType', 'flooring0', 'hasGarage', 'hasView']

one_hot = OneHotEncoder()
transformer = ColumnTransformer([('one_hot', one_hot, categorize)], remainder='passthrough')
transformed_X = transformer.fit_transform(X)

#Split the data into a test set and a training set
X_train, X_test, y_train, y_test = train_test_split(transformed_X, y, test_size=0.2)

#Use the Ridge regression model
clf = linear_model.Ridge()

clf.fit(X_train, y_train)

#Function to get the model score, predicted price and mean absolute error
def model_score():
    y_preds = clf.predict(X_test)
    score = clf.score(X_test, y_test)
    mae = mean_absolute_error(y_test, y_preds)
    return [score, mae]

#Make a prediction based on the user input
def get_input(x):
    return(clf.predict(X_test[x]))

@app.route('/', methods=['POST', 'GET'])
def index():
    return render_template('index.html')


@app.route('/data', methods =['POST', 'GET'])
def data():
    #Get the username and password from the form
    username = str(request.form.get('username'))
    password = str(request.form.get('password'))
    #Populate the dropdown box
    home0 = '0 Zipcode: ' + str(int(X_test[0, 425])) + ' Bedrooms: ' + str(int(X_test[0, 427])) + ' Bathrooms: ' + str(int(X_test[0,426])) + ' Square Footage: ' + str(int(X_test[0,429])) + ' Year Built: ' + str(int(X_test[0, 431]))
    home1 = '1 Zipcode: ' + str(int(X_test[1, 425])) + ' Bedrooms: ' + str(int(X_test[1, 427])) + ' Bathrooms: ' + str(int(X_test[1,426])) + ' Square footage: ' + str(int(X_test[1,429])) + ' Year Built: ' + str(int(X_test[1, 431]))
    home2 = '2 Zipcode: ' + str(int(X_test[2, 425])) + ' Bedrooms: ' + str(int(X_test[2, 427])) + ' Bathrooms: ' + str(int(X_test[2,426])) + ' Square footage: ' + str(int(X_test[2,429])) + ' Year Built: ' + str(int(X_test[2, 431]))                                                                                                                
    
    options = [home0, home1, home2]
    
    #Verify the user login, if incorrect print an error and add it to the log with the time
    if username == default_username and password == default_password:
        return render_template('data.html', tables=[df_to_table.to_html(classes='data')], header='true', options=options, username = username, password=password)
    else: 
        log_error('Incorrect Login at: ' + str(datetime.now()))
        return 'Incorrect login credentials'
    
@app.route('/prediction', methods=['POST', 'GET'])
def show_pred():   
    #Get the users input
    house = request.args.get('house')
    prediction = 0
    if house[0] == '0':
        prediction = get_input(0)
    elif house[0] == '1':
        prediction = get_input(1)
    elif house[0] == '2':
        prediction = get_input(2)
    print(house[0])
    #Clean the output
    prediction = str(prediction)
    prediction = prediction.replace('[','')
    prediction = prediction.replace(']','')
    prediction = int(float(prediction))
    #Get and display the model's score and mean absolute error
    score = model_score()
    return render_template('prediction.html', house = house, prediction = prediction, score = score[0], mae=score[1])


if __name__ == "__main__":
    app.run(debug=True)