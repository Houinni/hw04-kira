import numpy as np
import pandas as pd
from matplotlib.pyplot import subplots
import statsmodels.api as sm
import sklearn.model_selection as skm
import sklearn.linear_model as skl
from sklearn.preprocessing import StandardScaler
from ISLP import load_data
from ISLP.models import ModelSpec as MS
from functools import partial


from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from ISLP.models import \
(Stepwise,
sklearn_selected,
sklearn_selection_path,sklearn_sm)
from l0bnb import fit_path

def nCp(sigma2, estimator, X, Y):
    "Negative Cp statistic"
    n, p = X.shape
    Yhat = estimator.predict(X)
    RSS = np.sum((Y - Yhat)**2)
    return -(RSS + 2 * p * sigma2) / n

# 1. Split 
College = load_data('College')
College = College.dropna()
College_train, College_valid = skm.train_test_split(College,
                                        test_size=512,
                                        random_state=1)

# allvars = College.columns.drop(['Apps'])
# design = MS(allvars)

# # 2. Linear model
# X_train = design.fit_transform(College_train)
# y_train = College_train['Apps']
# model = sm.OLS(y_train, X_train)
# results = model.fit()

# X_valid = design.transform(College_valid)
# y_valid = College_valid['Apps']
# valid_pred = results.predict(X_valid)
# print(np.mean((y_valid - valid_pred)**2))

# Cross validation

# hp_model = sklearn_sm(sm.OLS, MS(allvars))
# X, Y = College.drop(columns=['Apps']), College['Apps']
# cv_results = skm.cross_validate(hp_model,X,Y,cv=College.shape[0])
# cv_err = np.mean(cv_results['test_score'])
# print(cv_err)

# Ridge regression model

design = MS(College.columns.drop('Apps')).fit(College)
Y = np.array(College['Apps'])

D = design.fit_transform(College)
D = D.drop('intercept', axis=1)
X = np.asarray(D)

lambdas = 10**np.linspace(8, -2, 100) / Y.std()

outer_valid = skm.ShuffleSplit(n_splits=1,
                                test_size=512,
                                random_state=1)


inner_cv = skm.KFold(n_splits=5,
                    shuffle=True,
                    random_state=2)

scaler = StandardScaler(with_mean=True, with_std=True)

ridgeCV = skl.ElasticNetCV(alphas=lambdas,
                            l1_ratio=0,
                            cv=inner_cv)

pipeCV = Pipeline(steps=[('scaler', scaler),
                            ('ridge', ridgeCV)])

results = skm.cross_validate(pipeCV,
                            X,
                            Y,
                            cv=outer_valid,
                            scoring='neg_mean_squared_error')

print(results['test_score'])