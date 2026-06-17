import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

data = pd.read_csv('AI/melb_data.csv')
y = data.Price
x = data.drop(['Price'], axis = 1)
x_train_full , x_valid_full , y_train , y_valid = train_test_split(x,y,train_size=0.8 , test_size=0.2 , random_state=0)
cols_with_missing = [col for col in x_train_full.columns if x_train_full[col].isnull().any()]
x_train_full.drop(cols_with_missing , axis=1 , inplace = True)
x_valid_full.drop(cols_with_missing , axis =1 , inplace = True)

low_cordinality_cols = [cname for cname in x_train_full.columns if x_train_full[cname].nunique()<10 and x_train_full[cname].dtype == "object"]
numerical_cols = [cname for cname in x_train_full.columns if x_train_full[cname].dtype in ['int64','float64']]
my_cols = low_cordinality_cols + numerical_cols 
x_train = x_train_full[my_cols].copy()
x_valid = x_valid_full[my_cols].copy()

x_train_encoded = pd.get_dummies(x_train , dtype=int)
x_valid_encoded = pd.get_dummies(x_valid , dtype=int)
x_train_encoded , x_valid_encoded = x_train_encoded.align(x_valid_encoded , join='left' , axis=1 ,fill_value=0)
def score_dataset(x_train , x_valid , y_train , y_valid):
    model = RandomForestRegressor(n_estimators=100 , random_state=0)
    model.fit(x_train , y_train)
    preds = model.predict(x_valid)
    return mean_absolute_error(y_valid , preds)

s = (x_train.dtypes == 'object')
object_cols = list(s[s].index)
print ("categorical variables:")
print (object_cols)

mae = score_dataset(x_train_encoded , x_valid_encoded , y_train , y_valid)
print(f'\nMean Absolute Error : ${mae:,.2f}')