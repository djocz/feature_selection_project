# Default imports
import pandas as pd

data = pd.read_csv('data/house_prices_multivariate.csv')

from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier


# Your solution code here
def rf_rfe(df):
    X,y = df.iloc[:,:-1],df.iloc[:,-1]
    model = RandomForestClassifier()
    features_no = X.columns
    rfe = RFE(model,len(X.columns)/2)
    rfe.fit(X,y)
    return X.columns.values[rfe.get_support()].tolist()
