#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import pandas as pd
import numpy as np


# In[2]:


data = pd.read_csv('housing.csv')
data.describe()


# In[3]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
data.hist(bins=20, figsize=(20, 15))
plt.show()


# In[4]:


def split_train_test(data, test_radio):
    np.random.seed(3)
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_radio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]


# In[5]:


data['income_cat'] = np.ceil(data['median_income'] / 1.5)
data['income_cat'].where(data['income_cat']<5, 5.0, inplace=True)


# In[6]:


from sklearn.model_selection import StratifiedShuffleSplit


# In[7]:


split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=1)
for train_index, test_index in split.split(data, data['income_cat']):
    strat_train_set = data.iloc[train_index]
    strat_test_set = data.iloc[test_index]


# In[8]:


for strat_set in (strat_train_set, strat_test_set):
    strat_set.drop(['income_cat'], axis=1, inplace=True)
data.drop(['income_cat'], axis=1, inplace=True)


# In[9]:


data_bak = data.copy()


# In[10]:


data.plot(kind='scatter', x='longitude', y = 'latitude', alpha=0.3,
         s=data['population']/100, label='population', c='median_house_value',
         cmap=plt.get_cmap('jet'), colorbar=True)
plt.legend()


# In[11]:


corr_matrix = data.corr()
corr_matrix['median_house_value'].sort_values(ascending=False)


# In[12]:


from pandas.tools.plotting import scatter_matrix
attributes = ['median_house_value', 'median_income', 'total_rooms',
             'housing_median_age']
scatter_matrix(data[attributes], figsize=(12,8))


# In[13]:


data.plot(kind='scatter', x='median_income', y='median_house_value', alpha=0.1)


# In[14]:


data['rooms_per_household'] = data['total_rooms']/data['households']
data['bedrooms_per_room'] = data['total_bedrooms']/data['total_rooms']
data['population_per_household'] = data['population']/data['households']
corr_matrix = data.corr()
corr_matrix['median_house_value'].sort_values(ascending=False)


# In[291]:


housing = strat_train_set.drop('median_house_value', axis=1)
housing_label = strat_train_set['median_house_value'].copy()


# In[102]:


from sklearn.preprocessing import Imputer
imputer = Imputer(strategy='median')

housing_num = housing.drop('ocean_proximity', axis=1)

imputer.fit(housing_num)
X = imputer.transform(housing_num)
housing_tr = pd.DataFrame(X, columns=housing_num.columns)


# In[103]:


from sklearn.preprocessing import OneHotEncoder


# In[104]:


encoder = OneHotEncoder()


# In[105]:


housing_cat = housing['ocean_proximity']


# In[227]:


class encode():
    #def __init__(self):
    def fit(self, X, y=None):
        encoder.fit(X.reshape(-1,1))
        return encoder
    def transform(self, X, y=None):
        housing_cat_encoded = encoder.transform(X.reshape(-1,1)).toarray()
        return housing_cat_encoded
enc = encode()


# In[228]:


rooms_ix, bedrooms_ix, population_ix, household_ix = 3, 4, 5, 6
class CombineAttributesAdder():
    def __init__(self, add_bedrooms_per_room = True):
        self.add_bedrooms_per_room = add_bedrooms_per_room
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        rooms_per_household = X[:, rooms_ix] / X[:, household_ix]
        population_per_household = X[:, population_ix] / X[:, household_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]
    
attribs_adder = CombineAttributesAdder(add_bedrooms_per_room=False)
housing_extra_attribs = attribs_adder.transform(housing.values)


# In[229]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# In[230]:


num_pipeline1 = Pipeline([
    ('imputer', Imputer(strategy='median')),
    ('attribs_adder', CombineAttributesAdder()),
    ('std_scaler', StandardScaler()),
])


# In[231]:


housing_num_tr = num_pipeline1.fit_transform(housing_num)


# In[232]:


class DataFrameSelector():
    def __init__(self, attribute_name):
        self.attribute_name = attribute_name
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        return X[self.attribute_name].values


# In[282]:


from sklearn.pipeline import FeatureUnion


# In[283]:


num_attribs = list(housing_num.columns)


# In[284]:


cat_attribs = ['ocean_proximity']


# In[285]:


num_pipeline0 = Pipeline([
    ('selector', DataFrameSelector(num_attribs)),
])
#num_pipeline = FeatureUnion(transformer_list=[
num_pipeline = Pipeline([
    ('selector', DataFrameSelector(num_attribs)),
    ('imputer', Imputer(strategy='median')),
    ('attribs_adder', CombineAttributesAdder()),
    ('std_scaler', StandardScaler()),
    #('num_pipeline0', num_pipeline0),
    #('num_pipeline1', num_pipeline1),
])


# In[286]:


cat_pipeline = Pipeline([
    ('selector', DataFrameSelector(cat_attribs)),
    ('onehotEncode', encode()),
])


# In[292]:


full_pipeline = FeatureUnion(transformer_list=[
    ('num_pipeline', num_pipeline),
    ('cat_pipeline', cat_pipeline),
])
housing.shape


# In[293]:


housing_prepared = full_pipeline.fit_transform(housing)
housing_prepared.shape


# In[240]:


from sklearn.linear_model import LinearRegression


# In[241]:


lin_reg = LinearRegression()


# In[242]:


lin_reg.fit(housing_prepared, housing_label)


# In[294]:


some_data = housing.iloc[:5]
some_labels = housing_label.iloc[:5]
some_data_prepared = full_pipeline.transform(some_data)
some_data_prepared.shape


# In[295]:


lin_reg.predict(some_data_prepared)


# In[248]:


some_labels.values


# In[249]:


from sklearn.metrics import mean_squared_error
housing_predictions = lin_reg.predict(housing_prepared)
lin_mse = mean_squared_error(housing_label, housing_predictions)
lin_rmse = np.sqrt(lin_mse)
lin_rmse


# In[250]:


from sklearn.tree import DecisionTreeRegressor
tree_reg = DecisionTreeRegressor()
tree_reg.fit(housing_prepared, housing_label)


# In[253]:


housing_predictions = tree_reg.predict(housing_prepared)
tree_mse = mean_squared_error(housing_label, housing_predictions)
tree_rmse = np.sqrt(tree_mse)
tree_rmse


# In[269]:


from sklearn.model_selection import cross_val_score
scores = cross_val_score(tree_reg, housing_prepared, housing_label,
                        scoring='neg_mean_squared_error', cv=10)
rmse_scores = np.sqrt(-scores)


# In[270]:


def display_scores(scores):
    print('Scores:', scores)
    print('Mean:', scores.mean())
    print('Standard deviation:', scores.std())


# In[279]:


display_scores(rmse_scores)


# In[272]:


lin_scores = cross_val_score(lin_reg, housing_prepared, housing_label,
                            scoring='neg_mean_squared_error', cv=10)
lin_rmse_scores = np.sqrt(-lin_scores)


# In[273]:


display_scores(lin_rmse_scores)


# In[276]:


from sklearn.ensemble import RandomForestRegressor
forest_reg = RandomForestRegressor()
forest_reg.fit(housing_prepared, housing_label)
housing_predictions = forest_reg.predict(housing_prepared)
rand_mse = mean_squared_error(housing_label, housing_predictions)
rand_rmse = np.sqrt(rand_mse)
rand_rmse


# In[277]:


rand_scores = cross_val_score(forest_reg, housing_prepared, housing_label,
                             scoring='neg_mean_squared_error', cv=10)
rand_rmse_scores = np.sqrt(-rand_scores)


# In[278]:


display_scores(rand_rmse_scores)


# In[297]:


test = strat_test_set.drop('median_house_value', axis=1)
test_label = strat_test_set['median_house_value'].copy()


# In[298]:


test_prepared = full_pipeline.transform(test)
predictions = lin_reg.predict(test_prepared)
mse = mean_squared_error(test_label, predictions)
rmse = np.sqrt(mse)
rmse


# In[ ]:





# In[ ]:




