#!/usr/bin/env python
# coding: utf-8

# # Dragon Real Estate - Price Predictor
# 

# In[1]:


import pandas as pd


# In[2]:


housing = pd.read_csv("housing.csv")


# In[3]:


housing.head()


# In[4]:


#RM= rooms per dwelling
#LSTAT=lower status of population
#PTRATIO=pupil teacher ration by town
#MEDV= median value of owner occupied homes in $1000'S


# In[5]:


housing.info()


# In[6]:


housing['chas'].value_counts()


# In[7]:


housing.describe()


# In[8]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[9]:


import matplotlib as plt


# In[10]:


housing.hist(bins=50,figsize=(20,15))


# 
# # Train Test Splitting

# In[11]:


import numpy as np

def split_train_test(data,test_ratio):
    np.random.seed(42)
    shuffled = np.random.permutation(len(data))
    print(shuffled)
    test_set_size=int(len(data) *test_ratio)
    test_indices = shuffled[:test_set_size]
    train_indices= shuffled[test_set_size:]
    return data.iloc[train_indices],data.iloc[test_indices]


# In[12]:


train_set,test_set=split_train_test(housing,0.2)


# In[13]:


#print(f"Rows in train set: {len(train_set)}\nRows in test set: {len(test_set)}\n")   #fstring


# In[14]:


from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(housing,test_size=0.2 , random_state=42)
print(f"Rows in train set: {len(train_set)}\nRows in test set: {len(test_set)}\n")   #fstring


# In[15]:


from sklearn.model_selection import StratifiedShuffleSplit
split=StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)
for train_index, test_index in split.split(housing,housing['chas']):
    
    strat_train_set = housing.loc[train_index]
    strat_test_set=housing.loc[test_index]
    
    


# In[16]:


strat_test_set['chas'].value_counts()


# In[17]:


housing=strat_train_set.copy()


# # Looking for Correlations
# 

# In[18]:


corr_matrix=housing.corr()


# In[19]:


corr_matrix['medv'].sort_values(ascending=False) 
#result shown below is 1 which means strong positive correlation


# In[20]:


from pandas.plotting import scatter_matrix
attributes=["rm","zn","indus","age"]
scatter_matrix(housing[attributes],figsize =(12,8))


# In[21]:


housing.plot(kind='scatter',x='rm',y='medv',alpha=0.8)


# # Atrribute Combinations

# In[22]:


housing['taxrm']=housing['tax']/housing['rm']


# In[23]:


housing.head()


# In[24]:


corr_matrix=housing.corr()
corr_matrix['medv'].sort_values(ascending=False) 


# In[25]:


housing.plot(kind='scatter',x='taxrm',y='medv',alpha=0.5) 


# In[26]:


housing = strat_train_set.drop("medv",axis=1)
housing_labels = strat_train_set['medv'].copy()


# # Missing Attributes

# In[27]:


#To take care of missing attributes , you have three options:
#1. get rid of missing data points
#2. get rid of whole attribute
#3. set the value to some value(0,mean or median)


# In[28]:


a=housing.dropna(subset=["rm"]) #option1
a.shape


# In[29]:


housing.drop("rm",axis=1).shape #option2
#Note that there is no rm column also note that original housing dataframr will remain unchanged 


# In[30]:


median=housing["rm"].median() # compute median for option3


# In[31]:


housing["rm"].fillna(median) #option 3


# In[32]:


housing.shape


# In[33]:


housing.describe()


# In[34]:


from sklearn.impute import SimpleImputer
imputer=SimpleImputer(strategy="mean")
imputer.fit(housing)


# In[35]:


imputer.statistics_


# In[36]:


imputer.statistics_.shape


# In[37]:


X=imputer.transform(housing)


# In[38]:


housing_tr=pd.DataFrame(X,columns=housing.columns)


# In[39]:


housing_tr.describe()


#  # Scikit-learn Design

# Primarily three types of objects 
# 1. Estimators-It estimates some parameter based on dataset ex=imputer
# It has a fit method and transform method.
# Fit method - Fits the dataset and calculates internal parameters
# 
# 2. Transformers- transform method takes input and returns output based on learning from fit().It also has a convinience function called fit_transform() which fits and then transforms
# 
# 3. Predictors-LinearRegression is an example of predictor. fit() & predict() are two common functions. It also gives score() function which will evaluate predictions.
# 

# # Feature Scaling 
# 

# Primarily , two types of feature scaling methods:
# 1. Min-max scaling (Normalization)
#     (value-min)/(max-min)
#     Sklearn provides a class called MinMax scalar for this 
#     
# 2. Standardization
#     (value-mean)/std
#     sklearn provides a class called Standard Scaler for this

# # Creating a Pipeline

# In[40]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
my_pipeline = Pipeline([
    ('imputer',SimpleImputer(strategy="mean")),
    ('std_scaler',StandardScaler()),
])


# In[41]:


housing_num_tr=my_pipeline.fit_transform(housing_tr)


# In[42]:


housing_num_tr


# ## Selecting a desired model for Dragon Real Estates 

# In[43]:


from sklearn.linear_model import LinearRegression 
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
#model = LinearRegression()
#model=DecisionTreeRegressor()
model = RandomForestRegressor()
model.fit(housing_num_tr,housing_labels)


# In[44]:


some_data=housing.iloc[:5]


# In[45]:


some_labels= housing_labels.iloc[:5]


# In[46]:


prepared_data=my_pipeline.transform(some_data)


# In[47]:


model.predict(prepared_data)


# In[48]:


list(some_labels)


# In[49]:


##evaluating the model 


# In[50]:


from sklearn.metrics import mean_squared_error
housing_predictions=model.predict(housing_num_tr)
mse=mean_squared_error(housing_labels,housing_predictions)
rmse=np.sqrt(mse)


# In[51]:


rmse #overfitting happened 


# # Better Technique - Cross Validation

# In[52]:


from sklearn.model_selection import cross_val_score
scores=cross_val_score(model,housing_num_tr,housing_labels,scoring="neg_mean_squared_error",cv=10)
rmse_scores=np.sqrt(-scores)


# In[53]:


rmse_scores


# In[54]:


#for finding out mean and std deviation
def print_scores(scores):
    print("Scores:",scores)
    print("Mean: ",scores.mean())
    print("Mean: ",scores.mean())
    print("Standard deviation: ",scores.std())


# In[55]:


print_scores(rmse_scores)


# In[56]:


#Decision tree output                              #Linear Regression 
#Mean:  4.207225187158817                           #Mean:  5.030437102767305  
#Standard deviation:  0.8745567547159991           #Standard deviation:  1.0607661158294834

#RandomForestRegressor
#Mean:  3.3009631251857217
#Standard deviation:  0.7076841067486248


# ## Saving the Model
# 
# 

# In[57]:


#launch
from joblib import dump, load
dump(model,'Dragon.joblib')


# ## Testing the model

# In[62]:


X_test =strat_test_set.drop("medv",axis=1)
Y_test=strat_test_set["medv"].copy()
X_test_prepared=my_pipeline.transform(X_test)
final_predictions=model.predict(X_test_prepared)
final_mse=mean_squared_error(Y_test, final_predictions)
final_rmse=np.sqrt(final_mse)
print(final_predictions,list(Y_test))


# In[61]:


final_rmse

