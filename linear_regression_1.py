import pandas as pd
import sklearn
import numpy as np
import matplotlib.pyplot as plt
#import seaborn as sns
import psycopg2 as pg2
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

conn = pg2.connect(database = 'Stock_Project' , user = 'postgres', password = 'crump83')
engine = create_engine('postgresql://postgres:crump83@localhost:5432/Stock_Project')
cur = conn.cursor()

df = pd.read_csv("USA_Housing.csv")
#print(df.head())
X = df[["Avg. Area Income","Avg. Area House Age","Avg. Area Number of Rooms","Avg. Area Number of Bedrooms","Area Population"]]
y = df["Price"]

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = .4, random_state = 101)

lm = LinearRegression()
lm.fit(X_train, y_train)

#print(lm.intercept_)
#print(lm.coef_)
cdf = pd.DataFrame(lm.coef_, X.columns, columns = ['Coeff'])
#print(cdf)
predictions = lm.predict(X_test)

#print(predictions)

plt.scatter(y_test, predictions)
#plt.show()
metrics.mean_absolute_error(y_test, predictions)
#mean squared error
metrics.mean_squared_error(y_test, predictions)
#root mean squared error
np.sqrt(metrics.mean_squared_error(y_test, predictions))
