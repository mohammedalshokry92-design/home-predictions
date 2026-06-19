import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    OneHotEncoder,
    OrdinalEncoder
)
data = pd.read_csv('AI/melb_data.csv')
y = data.Price
x = data.drop(['Price'],axis=1)
x_train_full , x_valid_full , y_train , y_valid = train_test_split(
   x , y , train_size=0.8 , random_state =0 
)

cols_with_missing = [
    col for col in x_train_full.columns if x_train_full[col].isnull().any()
]
x_train_full.drop(cols_with_missing , axis=1 , inplace = True)
x_valid_full.drop(cols_with_missing , axis=1 , inplace = True)
low_cordinality_cols = [
    cname
    for cname in x_train_full
    if x_train_full[cname].nunique()<10
    and x_train_full[cname].dtype == 'object'
]
numerical_cols = [
    cname
    for cname in x_train_full.columns
    if x_train_full[cname].dtype in ['int64' , 'float64']
]
my_cols = low_cordinality_cols + numerical_cols
x_train = x_train_full[my_cols].copy()
x_valid = x_valid_full[my_cols].copy()

def score_dataset(x_train , x_valid , y_train , y_valid):
    model = RandomForestRegressor(n_estimators = 300 , random_state=0)
    model.fit(x_train , y_train)
    preds = model.predict(x_valid)
    return mean_absolute_error(y_valid , preds)

s= x_train.dtypes == 'object'
object_cols = list(s[s].index)
print ('categorical varibles:')
print (object_cols)
print ('='*50)
label_x_train = x_train.copy()
label_x_valid = x_valid.copy()

ordinal_encoder = OrdinalEncoder(
    handle_unknown= 'use_encoded_value',unknown_value=-1
)
label_x_train[object_cols] = ordinal_encoder.fit_transform(
  x_train[object_cols]  
)
label_x_valid[object_cols] = ordinal_encoder.transform(x_valid[object_cols])

mae_ordinal = score_dataset(label_x_train , label_x_valid , y_train , y_valid)
print (f'MAE from Approach 2(ordinal encoding) : ${mae_ordinal: ,.2f}')
print ('-'*50)

OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(x_train[object_cols]))
OH_cols_valid = pd.DataFrame(OH_encoder.transform(x_valid[object_cols]))

OH_cols_train.index = x_train.index
OH_cols_valid.index = x_valid.index

OH_cols_train.columns = OH_cols_train.columns.astype(str)
OH_cols_valid.columns = OH_cols_valid.columns.astype(str)

num_x_train = x_train.drop(object_cols , axis = 1)
num_x_valid = x_valid.drop(object_cols , axis =1)

OH_x_train = pd.concat([num_x_train , OH_cols_train] , axis=1)
OH_x_valid = pd.concat([num_x_valid , OH_cols_valid] , axis=1)


mae_OH_encoder = score_dataset(OH_x_train , OH_x_valid , y_train , y_valid)
print(f'MAE from Approach3(One hot encoder) : ${mae_OH_encoder:,.2f}')
print ('='*50)