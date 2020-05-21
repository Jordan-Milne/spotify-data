import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn_pandas import DataFrameMapper
from sklearn.preprocessing import StandardScaler, LabelEncoder, LabelBinarizer
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv('playlists.csv')

target = 'person'
y = df[target]
X = df.drop(target, axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
df.head()

# DataFrame Mapper
mapper = DataFrameMapper([
     (['danceability'], StandardScaler()),
     ('explicit', LabelEncoder()),
     (['energy'], [StandardScaler()]),
     (['key'], [StandardScaler()]),
     (['loudness'],  [StandardScaler()]),
     (['mode'], StandardScaler()),
     (['speechiness'], StandardScaler()),
     (['instrumentalness'], StandardScaler()),
     (['liveness'],  StandardScaler()),
     (['valence'], StandardScaler()),
     (['tempo'],  StandardScaler()),
     (['duration_ms'],  StandardScaler()),
     ], df_out=True)

Z_train = mapper.fit_transform(X_train)
Z_test = mapper.transform(X_test)


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV


model = RandomForestClassifier().fit(Z_train, y_train)
print(f'Training score {round(model.score(Z_train, y_train),3)}')
print(f'Training score {round(model.score(Z_test, y_test),3)}')

param_grid = {'n_estimators': [1, 10, 25, 50, 100],
              'min_samples_split':[2, 4, 6, 8, 10],
              'min_samples_leaf': [1, 2, 3, 4],
              'max_depth': [1, 10, 25, 50]}

grid = GridSearchCV(RandomForestClassifier(), param_grid, cv=3, verbose=1, n_jobs=-1)
grid.fit(Z_train, y_train)



print(grid.best_score_)
print(grid.best_params_)
rf_best = grid.best_estimator_


import catboost as cb

model = cb.CatBoostClassifier(
    iterations=1000,
    early_stopping_rounds=10,
    custom_loss=['AUC', 'Accuracy']
)

model.fit(
    Z_train,
    y_train,
    eval_set=(Z_train, y_train),
    verbose=False,
    plot=True)
model.tree_count_

model_with_early_stop = cb.CatBoostClassifier(
    iterations=200,
    random_seed=63,
    learning_rate=0.5,
    early_stopping_rounds=20
)
model_with_early_stop.fit(
    X_train, y_train,
    cat_features=cat_features,
    eval_set=(X_validation, y_validation),
    verbose=False,
    plot=True
)













# Base Model
model = RandomForestClassifier()
# model = LogisticRegression()
model.fit(Z_train,y_train)
model.score(Z_train,y_train)
model.score(Z_test,y_test)

# # personal testing
# yhat = model.predict(Z_test)
#
#
# pd.DataFrame({
#     'y_true': y_test,
#     'y_hat': yhat
# })
