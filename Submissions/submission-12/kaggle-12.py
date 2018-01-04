# -*- coding: utf-8 -*-

# This code is initially based on the Kaggle kernel from Sergei Neviadomski, which can be found in the following link
# https://www.kaggle.com/neviadomski/how-to-get-to-top-25-with-simple-model-sklearn/notebook
# and the Kaggle kernel from Pedro Marcelino, which can be found in the link below
# https://www.kaggle.com/pmarcelino/comprehensive-data-exploration-with-python/notebook
# Also, part of the preprocessing and modelling has been inspired by this kernel from Serigne
# https://www.kaggle.com/serigne/stacked-regressions-top-4-on-leaderboard
# And this kernel from juliencs has been pretty helpful too!
# https://www.kaggle.com/juliencs/a-study-on-regression-applied-to-the-ames-dataset

# Adding needed libraries and reading data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import ensemble, tree, linear_model, preprocessing
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.linear_model import ElasticNet, Lasso
from sklearn.kernel_ridge import KernelRidge
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.utils import shuffle
from scipy import stats
from scipy.stats import norm, skew, boxcox
from scipy.special import boxcox1p
import xgboost as xgb
import lightgbm as lgb

import warnings
warnings.filterwarnings('ignore')

# Class AveragingModels
# This class is Serigne's simplest way of stacking the prediction models, by
# averaging them. We are going to use it as it represents the same that we have
# been using in the late submissions, but this applies perfectly to rmsle_cv function.
class AveragingModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, models):
        self.models = models

    # we define clones of the original models to fit the data in
    def fit(self, X, y):
        self.models_ = [clone(x) for x in self.models]

        # Train cloned base models
        for model in self.models_:
            model.fit(X, y)

        return self

    #Now we do the predictions for cloned models and average them
    def predict(self, X):
        predictions = np.column_stack([
            model.predict(X) for model in self.models_
        ])
        return np.mean(predictions, axis=1)


train = pd.read_csv("../../train.csv")
test = pd.read_csv("../../test.csv")


#Save the 'Id' column
train_ID = train['Id']
test_ID = test['Id']
train.drop('Id', axis=1, inplace=True)
test.drop('Id', axis=1, inplace=True)

# Visualizing outliers
fig, ax = plt.subplots()
ax.scatter(x = train['GrLivArea'], y = train['SalePrice'])
plt.ylabel('SalePrice', fontsize=13)
plt.xlabel('GrLivArea', fontsize=13)
#plt.show()

# Now the outliers can be deleted
train = train.drop(train[(train['GrLivArea'] > 4000) & (train['SalePrice'] < 300000)].index)

#Check the graphic again, making sure there are no outliers left
fig, ax = plt.subplots()
ax.scatter(train['GrLivArea'], train['SalePrice'])
plt.ylabel('SalePrice', fontsize=13)
plt.xlabel('GrLivArea', fontsize=13)
#plt.show()

#We use the numpy fuction log1p which applies log(1+x) to all elements of the column
train["SalePrice"] = np.log1p(train["SalePrice"])

#Check the new distribution
sns.distplot(train['SalePrice'] , fit=norm);

# Get the fitted parameters used by the function
(mu, sigma) = norm.fit(train['SalePrice'])
print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

#Now plot the distribution
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
            loc='best')
plt.ylabel('Frequency')
plt.title('SalePrice distribution')

#Get also the QQ-plot
fig = plt.figure()
res = stats.probplot(train['SalePrice'], plot=plt)
#plt.show()

# Splitting to features and labels
train_labels = train.pop('SalePrice')

# Test set does not even have a 'SalePrice' column, so both sets can be concatenated
features = pd.concat([train, test], keys=['train', 'test'])

# Checking for missing data, showing every variable with at least one missing value in train set
total_missing_data = features.isnull().sum().sort_values(ascending=False)
missing_data_percent = (features.isnull().sum()/features.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total_missing_data, missing_data_percent], axis=1, keys=['Total', 'Percent'])
print(missing_data[missing_data['Percent']> 0])

# Deleting non-interesting variables for this case study
features.drop(['Utilities'], axis=1, inplace=True)

# Imputing missing values and transforming certain columns

# Converting OverallCond to str
features.OverallCond = features.OverallCond.astype(str)

# MSSubClass as str
features['MSSubClass'] = features['MSSubClass'].astype(str)

# MSZoning NA in pred. filling with most popular values
features['MSZoning'] = features['MSZoning'].fillna(features['MSZoning'].mode()[0])

# LotFrontage NA filling with median according to its OverallQual value
median = features.groupby('OverallQual')['LotFrontage'].transform('median')
features['LotFrontage'] = features['LotFrontage'].fillna(median)

# Alley NA in all. NA means no access
features['Alley'] = features['Alley'].fillna('NoAccess')

# MasVnrArea NA filling with median according to its OverallQual value
median = features.groupby('OverallQual')['MasVnrArea'].transform('median')
features['MasVnrArea'] = features['MasVnrArea'].fillna(median)

# MasVnrType NA in all. filling with most popular values
features['MasVnrType'] = features['MasVnrType'].fillna(features['MasVnrType'].mode()[0])

# BsmtQual, BsmtCond, BsmtExposure, BsmtFinType1, BsmtFinType2
# NA in all. NA means No basement
for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
    features[col] = features[col].fillna('NoBsmt')

# TotalBsmtSF  NA in pred. I suppose NA means 0
features['TotalBsmtSF'] = features['TotalBsmtSF'].fillna(0)

# Electrical NA in pred. filling with most popular values
features['Electrical'] = features['Electrical'].fillna(features['Electrical'].mode()[0])

# KitchenAbvGr to categorical
features['KitchenAbvGr'] = features['KitchenAbvGr'].astype(str)

# KitchenQual NA in pred. filling with most popular values
features['KitchenQual'] = features['KitchenQual'].fillna(features['KitchenQual'].mode()[0])

# FireplaceQu  NA in all. NA means No Fireplace
features['FireplaceQu'] = features['FireplaceQu'].fillna('NoFp')

# Garage-like features NA in all. NA means No Garage
for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageYrBlt', 'GarageCond'):
    features[col] = features[col].fillna('NoGrg')

# GarageCars and GarageArea NA in pred. I suppose NA means 0
for col in ('GarageCars', 'GarageArea'):
    features[col] = features[col].fillna(0.0)

# SaleType NA in pred. filling with most popular values
features['SaleType'] = features['SaleType'].fillna(features['SaleType'].mode()[0])

# PoolQC NA in all. NA means No Pool
features['PoolQC'] = features['PoolQC'].fillna('NoPool')

# MiscFeature NA in all. NA means None
features['MiscFeature'] = features['MiscFeature'].fillna('None')

# Fence NA in all. NA means no fence
features['Fence'] = features['Fence'].fillna('NoFence')

# BsmtHalfBath and BsmtFullBath NA means 0
for col in ('BsmtHalfBath', 'BsmtFullBath'):
    features[col] = features[col].fillna(0)

# Functional NA means Typ
features['Functional'] = features['Functional'].fillna('Typ')

# NA in Bsmt SF variables means not that type of basement, 0 square feet
for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF'):
    features[col] = features[col].fillna(0)

# NA in Exterior1st filled with the most common value
features['Exterior1st'] = features['Exterior1st'].fillna(features['Exterior1st'].mode()[0])

# NA in Exterior2nd means No 2nd material
features['Exterior2nd'] = features['Exterior2nd'].fillna('NoExt2nd')

# Year and Month to categorical
features['YrSold'] = features['YrSold'].astype(str)
features['MoSold'] = features['MoSold'].astype(str)

# Adding total sqfootage feature and removing Basement, 1st and 2nd floor features
features['TotalSF'] = features['TotalBsmtSF'] + features['1stFlrSF'] + features['2ndFlrSF']
#features.drop(['TotalBsmtSF', '1stFlrSF', '2ndFlrSF'], axis=1, inplace=True)


###################################################################################################
# Let's rank those categorical features that can be understood to have an order
# Criterion: give higher ranking to better feature values
features = features.replace({'Street' : {'Grvl':1, 'Pave':2},
                             'Alley' : {'NoAccess':0, 'Grvl':1, 'Pave':2},
                             'LotShape' : {'I33':1, 'IR2':2, 'IR1':3, 'Reg':4},
                             'LandContour' : {'Low':1, 'HLS':2, 'Bnk':3, 'Lvl':4},
                             'LotConfig' : {'FR3':1, 'FR2':2, 'CulDSac':3, 'Corner':4, 'Inside':5},
                             'LandSlope' : {'Gtl':1, 'Mod':2, 'Sev':3},
                             'HouseStyle' : {'1Story':1, '1.5Fin':2, '1.5Unf':3, '2Story':4, '2.5Fin':5, '2.5Unf':6, 'SFoyer':7, 'SLvl':8},
                             'ExterQual' : {'Po':1, 'Fa':2, 'TA':3, 'Gd':4, 'Ex':5},
                             'ExterCond' : {'Po':1, 'Fa':2, 'TA':3, 'Gd':4, 'Ex':5},
                             'BsmtQual' : {'NoBsmt':0, 'Po':1, 'Fa':2, 'TA':3, 'Gd':4, 'Ex':5},
                             'BsmtCond' : {'NoBsmt':0, 'Po':1, 'Fa':2, 'TA':3, 'Gd':4, 'Ex':5},
                             'BsmtExposure' : {'NoBsmt':0, 'No':1, 'Mn':2, 'Av':3, 'Gd':4},
                             'BsmtFinType1' : {'NoBsmt':0, 'Unf':1, 'LwQ':2, 'Rec':3, 'BLQ':4, 'ALQ':5, 'GLQ':6},
                             'BsmtFinType2' : {'NoBsmt':0, 'Unf':1, 'LwQ':2, 'Rec':3, 'BLQ':4, 'ALQ':5, 'GLQ':6},
                             'HeatingQC' : {'Po':1, 'Fa':2, 'TA':3, 'Gd':4, 'Ex':5},
                             'CentralAir' : {'N':0, 'Y':1},
                             'KitchenQual' : {'Po':1, 'Fa':2, 'TA':3, 'Gd':4, 'Ex':5},
                             'Functional' : {'Sal':0, 'Sev':1, 'Maj2':2, 'Maj1':3, 'Mod':4, 'Min2':5, 'Min1':6, 'Typ':7},
                             'FireplaceQu' : {'NoFp':0, 'Po':1, 'Fa':2, 'TA':3, 'Gd':4, 'Ex':5},
                             'GarageType' : {'NoGrg':0, 'Detchd':1, 'CarPort':2, 'BuiltIn':3, 'Basment':4, 'Attchd':5, '2Types':6},
                             'GarageFinish' : {'NoGrg':0, 'Unf':1, 'RFn':2, 'Fin':3},
                             'GarageQual' : {'NoGrg':0, 'Po':1, 'Fa':2, 'TA':3, 'Gd':4, 'Ex':5},
                             'GarageCond' : {'NoGrg':0, 'Po':1, 'Fa':2, 'TA':3, 'Gd':4, 'Ex':5},
                             'PavedDrive' : {'N':0, 'P':1, 'Y':2},
                             'PoolQC' : {'NoPool':0, 'Fa':1, 'TA':2, 'Gd':3, 'Ex':4},
                             'Fence' : {'NoFence':0, 'MnWw':1, 'GdWo':2, 'MnPrv':3, 'GdPrv':4}
                             })
###################################################################################################

# Box-cox transformation to most skewed features
numeric_feats = features.dtypes[features.dtypes != 'object'].index

# Check the skew of all numerical features
skewed_feats = features[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
print("\nSkew in numerical features:")
skewness = pd.DataFrame({'Skew' :skewed_feats})
skewness.head(10)

# Box-cox
skewness = skewness[abs(skewness) > 0.75]
print("There are {} skewed numerical features to Box Cox transform\n".format(skewness.shape[0]))

from scipy.special import boxcox1p
skewed_features = skewness.index
lam = 0.15
for feat in skewed_features:
    features[feat] = boxcox1p(features[feat], lam)


# Label encoding to some categorical features
categorical_features = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond',
        'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1',
        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',
        'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond',
        'YrSold', 'MoSold')
lbl = LabelEncoder()

for col in categorical_features:
    lbl.fit(list(features[col].values))
    features[col] = lbl.transform(list(features[col].values))

# Getting Dummies
features = pd.get_dummies(features)

# Splitting features
train_features = features.loc['train'].select_dtypes(include=[np.number]).values
test_features = features.loc['test'].select_dtypes(include=[np.number]).values

# Validation function
n_folds = 5

def rmsle_cv(model):
    kf = KFold(n_folds, shuffle=True, random_state=101010).get_n_splits(train_features)
    rmse= np.sqrt(-cross_val_score(model, train_features, train_labels, scoring="neg_mean_squared_error", cv = kf))
    return(rmse)

# Modelling
enet_model = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=101010))

print(enet_model)
#score = rmsle_cv(enet_model)
#print("\nRMSLE: {:.4f} (+/- {:.4f})\n".format(score.mean(), score.std()))

gb_model = ensemble.GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10,
                                   loss='huber', random_state =101010)

print(gb_model)
#score = rmsle_cv(gb_model)
#print("\nRMSLE: {:.4f} (+/- {:.4f})\n".format(score.mean(), score.std()))

xgb_model = xgb.XGBRegressor(colsample_bytree=0.2, gamma=0.0,
                             learning_rate=0.05, max_depth=3,
                             min_child_weight=1.7, n_estimators=2200,
                             reg_alpha=0.9, reg_lambda=0.6,
                             subsample=0.5, silent=1, seed=101010)

print(xgb_model)
#score = rmsle_cv(xgb_model)
#print("\nRMSLE: {:.4f} (+/- {:.4f})\n".format(score.mean(), score.std()))

lasso_model = make_pipeline(RobustScaler(), Lasso(alpha=0.0005, random_state=101010))

print(lasso_model)
#score = rmsle_cv(lasso_model)
#print("\nRMSLE: {:.4f} (+/- {:.4f})\n".format(score.mean(), score.std()))

krr_model = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)

print(krr_model)
#score = rmsle_cv(krr_model)
#print("\nRMSLE: {:.4f} (+/- {:.4f})\n".format(score.mean(), score.std()))


# Now let's check how do the averaged models work
averaged_models = AveragingModels(models = (gb_model, xgb_model, enet_model, lasso_model, krr_model))
print("AVERAGED MODELS")
score = rmsle_cv(averaged_models)
print("\nRMSLE: {:.4f} (+/- {:.4f})\n".format(score.mean(), score.std()))

# Getting our SalePrice estimation
averaged_models.fit(train_features, train_labels)
final_labels = np.exp(averaged_models.predict(test_features))

# Saving to CSV
pd.DataFrame({'Id': test_ID, 'SalePrice': final_labels}).to_csv('submission-12.csv', index =False)
