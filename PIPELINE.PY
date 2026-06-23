import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

data = pd.read_csv('AI/melb_data.csv')
Y = data.Price
X = data.drop(['Price'] , axis=1)

X_train_full , X_valid_full , Y_train , Y_valid = train_test_split(X , Y , train_size=0.8 , test_size = 0.2 , random_state = 0)

categorical_cols = [cname for cname in X_train_full.columns if X_train_full[cname].nunique()<10 and X_train_full[cname].dtype == "object"]
numerical_cols = [cname for cname in X_train_full.columns if X_train_full[cname].dtype in ['int64' , 'float64']]
my_cols = categorical_cols + numerical_cols

X_train = X_train_full[my_cols].copy()
X_valid = X_valid_full[my_cols].copy()
X_train.head()

numerical_transformer = SimpleImputer(strategy='constant')
categorical_transformer = Pipeline(steps = [('imputer' , SimpleImputer(strategy='most_frequent')) , ('onehot' , OneHotEncoder(handle_unknown='ignore' , sparse_output = False)) ])

preprocessor = ColumnTransformer(transformers=[("num" , numerical_transformer , numerical_cols) ,('cat' , categorical_transformer , categorical_cols) ])
print ("pipeline have been ready")
model = RandomForestRegressor(n_estimators=100 , random_state=0)

my_pipeline = Pipeline(steps=[('preprocessor',preprocessor) , ('model', model)])
my_pipeline.fit(X_train , Y_train)
preds = my_pipeline.predict(X_valid)
score = mean_absolute_error(Y_valid , preds)
print('MAE:',score)