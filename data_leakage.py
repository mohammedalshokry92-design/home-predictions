import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score


data = pd.read_csv('AI/AER_credit_card_data.csv' , true_values=['yes'] , false_values=['no'])
y = data.card
X= data.drop(['card'] , axis=1)
print ("Number of rows in dataset:", X.shape[0])
X.head()
my_pipeline = make_pipeline(RandomForestClassifier(n_estimators=100))
cv_scores = cross_val_score(my_pipeline , X , y , cv=5 , scoring='accuracy')
print ("cross_validation accuracy: %f" % cv_scores.mean())
expenditures_cardholders = X.expenditure[y]
expenditures_noncardholders = X.expenditure[~y]
print ('Fraction of those who did not receive a card and had no expenditures: %.2f' \
      %((expenditures_noncardholders == 0).mean()))
print('Fraction of those who did not receive a card and had no expenditures: %.2f' \
      %((expenditures_cardholders == 0).mean()))
potential_leaks = ['expenditure' , 'share' , 'active' , 'majorcards']
X2 = X.drop(potential_leaks , axis=1)
cv_scores = cross_val_score(my_pipeline , X2, y , cv=5 , scoring= 'accuracy')
print ("cross-val accuracy:%f" % cv_scores.mean())