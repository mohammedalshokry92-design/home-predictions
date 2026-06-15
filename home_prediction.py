import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

dummy_data = {
    "LotArea":[8450, 9600, 11250, 9550, 14260, 14115, 10084, 10382, 6120, 7420],
    "YearBuilt":[2003, 1976, 2001, 1915, 2000, 1993, 2004, 1973, 1931, 1939],
    "1stFlrSF":[856, 1262, 920, 961, 1145, 796, 1694, 1107, 1022, 1040],
    "2ndFlrSF":[854, 0, 866, 756, 1053, 566, 0, 983, 0, 0],
    "BedroomAbvGr":[3, 3, 3, 3, 4, 1, 3, 3, 2, 2],
    "SalePrice":[205800,181500,223500,140000,250000,215000,135000,155500,260000,195000]
}

df = pd.DataFrame(dummy_data)
df.to_csv("train.csv", index=False)
print("data file have been done")
home_data= pd.read_csv("train.csv")
y = home_data.SalePrice

features = [
    "LotArea",
    "YearBuilt",
    "1stFlrSF",
    "2ndFlrSF",
    "BedroomAbvGr",
]
x = home_data[features]
train_x , val_x , train_y , val_y = train_test_split(x,y,test_size=0.3 , random_state=1)

rf_model = RandomForestRegressor(random_state=1)
rf_model.fit(train_x , train_y)

rf_val_predictions = rf_model.predict(val_x)
rf_val_mae = mean_absolute_error(rf_val_predictions , val_y)

print ("programe succesfully worked")
print(f"average of home prediction :${rf_val_mae:.2f}")
print("prediction of 5 houses\n")
print(rf_val_predictions[:5])