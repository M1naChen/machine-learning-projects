#!/usr/bin/env python
# coding: utf-8

# # Price prediction for Airbnb new listings--NYC Area
# 
# ---
# 
# ## Table of Contents
# [Background & Motivation](#background_motivation)  
# [A Discussion about Questions](#discussion_about_questions)  
# [Methods & Results](#method_results)  
# [Communication of Results, and Advice to a Non-expert](#communication_of_results)   
# [Exploratory Data Analysis](#Exploratory)  
# 
# [Baseline Model](#baseline_model)  
# [SVR Model](#SVR)  
# [Knn Model](#Knn)  
# [Regression Model](#Regression)  
# [Example of the final model application](#example)
# 
# 
# [Reference](#reference)  
# [Appendixes](#Appendixes)  
# 
# 
# ---

# 
# # Authors
# 1. Mina Chen
# 2. Cheryl Cui
# 3. Perla Gala  
# 
# Project Completion Date:May 26, 2021
# 
# <a id="background_motivation"></a>
# 
# # Background & Motivation
# 
# Airbnb disrupted the hospitality business in 2008 when it was funded by Brian Chesky, Nathan Blecharczyk, Joe Gebbia. According to one of the latest reports New Airbnb Hosts Have Earned $1 Billion During the Pandemic (Airbnb, Feb 17, 2021). After more than one year in lock-down people are looking for safe ways to reconnect with friends and families,  and Airbnb hosts have the kind of properties that work fine for handling groups. Back in 2018 Airbnb already demonstrated to be a better fit for groups, according to the website concepta, 88% of reservations are for groups of 2-4 people, while the hotel industry’s average was 1.2 guests per room. Therefore, even within the pandemic challenges, Airbnb is still a growing opportunity for hosts to earn profit and guests to find an affordable location to reconnect with friends and family,
# 
# Back in 2018, while describing the major benefits Airbnb has from predictive analytics, the website  concepta, stated 80% of Airbnb hosts population have only one listing, and the majority have full time jobs, family, and they use Airbnb as a source of extra income. Majority of the hosts don't have experience on property management or market research on rental pricings. Airbnb developed Smart Pricing for the hosts that already belong to their platform. This tool allows the host to easily set a price and update based on the current data regarding available inventory in the area, seasonality, local events, etc. Airbnb and its hosts heavily rely on predictive analytics to adjust prices and allow profitability to the host and the company. The number of beds, bathrooms, location and past successful events are some of the features that Smart Pricing considers to suggest prices, saving time and effort for hosts to set prices, and maximizing profit for both hosts and Airbnb. 
# 
# But what about people that are not part of the Airbnb yet? The motivation for study is to provide information to potential hosts in New York (or any city if the parameters are adjusted and source data is available) a scientific, non-cost alternative to estimate the prices they could set for their properties. Data is the most valuable asset, and thankfully there is a lot of openness to share data; Inside Airbnb is an independent organization willing to provide publicly available data to get insights about a city's Airbnb's listings. For this specific study we used data available in kaggle. This study bases the results on analyzing available data for listing in the NY area including categorical variables such as neighbourhood and room type and quantitative data such as minimum nights, latitude, longitude, calculated host listings and availability in the upcoming 365 days. We processed this data by applying three different machine learning (ML) models, two of them analogy based: k Nearest Neighbors (kNN) algorithm and  Support Vector Machines (SVMs) with RBF Kernel, the third method is Linear Regression model that different from the other ones is not a classification algorithm but a statistical method to predict continuous variables.
# 
# <a id="discussion_about_questions"></a>
# # A Discussion about Questions
# **How does a non-experienced Airbnb host calculate a fair night price for his/her property?** 
# The business question here is relevant to non-experienced individuals who want to explore how much they could price their properties if they decide to step up into the Airbnb business. The study is approaching people who do not have access to sophisticated tools such as Smart Pricing from Airbnb developed for their current host, and/or do not have a lot of experience in property pricing methods and market research.
# 
# **Statistical question:**
# 
# | Business Objective | Statistical Question |
# | :--- | :--- |
# | Decide if its is worth to step in Airbnb business | <ul>$Y$ = Estimate night price <br> $X$ = <ul><li>Numeric features: latitude, longitude, minimum_nights, calculated_host_listings_count, availability_365. <li> Categorical variables: neighbourhood_group, neighbourhood, room_type.<ul></ul>
# 
# 
# The added value from this statistical question is based on the direct application it has, based on some parameters that any person can find available on the internet, it becomes easy and practical the process to set a price for a property that anyone would like to list in Airbnb. 
# The limitations come with the lack of data available. While this study focus on features such as location and number of nights the host would like to offer the property, the model could grow robust by adding other components from time series analysis, specific calendar events or larger datasets. 
# 
# 
# <a id="method_results"></a>
# # Method & Results
# 
# With the goal to find a method to estimate the night price for an Airbnb we evaluated three different methods: Support Vector Machines (SVMs) with RBF Kernel, k Nearest Neighbors (kNN) and Linear Regression.
# 
# Note that the scope of this study is limited to the public available data of Airbnb listings in New York City from 2011 to 2019. 
# 
# Feature Preprocessing
# For the three methods discussed, pre-processing of the data included
# 
# **Numeric Data**<br>
# * Imputer: Median
# * StandardScaler
# 
# **Categorical data**<br>
# * OneHotEncoder
# * Imputer: constant
# * Fill missing values: constant 
# * Handle_unknown: ignore 
# 
# Additionally, for the final selected, we evaluated the alternative to apply ordinal encode for the variable room_type to assign priority to the different values for this variable:
# 1. Shared room
# 2. Private room
# 3. Entire home/apt
# 
# ### Results
# 
# After fitting the models, and run random search to find the best parameters, we came up with the test scores for each model:
# 
# |Model|Test Score|neighbourhood_group|neighbourhood|lat|long|room_type|price|minimum_nights|Availabiliy|
# |-----|----------|-------------------|-------------|---|----|---------|-----|--------------|-----------|
# |Dummy Classifier|0.042|√|√|√|√|√|√|√|√|
# |SVR|0.566|√|√|√|√|√|√|√|√|
# |*k*-nn|0.552|√|√|√|√|√|√|√|√|
# |Regression|0.535|X|√|√|√|√|√|√|√|
# 
# The best score value is from the *k*-nn model, however, we don't want to use this method nor SVR given the type of data we are trying to estimate. Regression model is more relaiable to analyze each one of the features in the observation we have and predict better the best value for the quantitative variable ths study is solving for.
# 
# Improvements to the model could include provide more data even about the host. [Concepta](https://www.conceptatech.com/blog/price-optimization-how-dynamic-pricing-helps-airbnb-hosts-earn-big) websites states that *guests seem to prefer renting from women over 50* and *lder hosts make about $8350 mostrly renting partial properties and guest houses* (concepta, 2018). Having access to demographics among the host can make this model more robust.  
# 
# <a id="communication_of_results"></a>
# # Communication of Results, and Advice to a Non-expert
# 
# A good way to explain the result is by looking at an hypothetical example. We want to calculate the price for a for a private room in my property located in Brooklyn in the neighborhood of Kensington (latitude: 40.56735, longitude: -74.0245, this information can be obtained by plugging your addres in most of the maps applications), the minimum number of nights a guest can book the room is one night, and the owner of this property is willing to host a customer for 300 nights of a year. 
# 
# The data that will be provided is:
# 
# |Neighborhood group|Neighborhood|latitude|longitude|Room type|Host listings|Minimum nights|Availability 365|
# |------------------|------------|--------|---------|---------|-------------|--------------|----------------|
# |Brooklyn|Kensington|40.56735|-74.0245|Private Room|1|1|300|
# 
# With the models we evaluated the room can be priced at:
# 
# |Option|Model|Price per night ($)|
# |------|-----|-------------------|
# |Option 1|SVR|124.81|
# |Option 2|K-NN |77.97|
# |Option 3|Linear regression|87.09|
# 
# Even when a potential host can be attracted to the most expensive price, it is important to highlight that the most accurate is option 3, the recommended price is actually pricing the room at $87.09. 
# Option one is a good method that estimates the price based on similarities of this room against the ones that are listed in the data we have available to build the models. This option has the advantage to *remember* the most representative examples among all the data we have. 
# The second option suggests a price only based on some of the most similar observations (in this case we only set 11)  without adding any particular priority at some other listings that may have more importance in the data. 
# Finally, the best option is listing the room at **87.09**, this is the best alternative since the price is suggested based on how important is each individual feature in our property, given the data we have available.  This alternative is considering the positive or negative impact that each feature has and weighs how important is each feature in comparison with the others. 
# 
# 

# # Setup
# 
# ## Library import
# We import all the required Python libraries

# In[1]:


# Importing our libraries
import pandas as pd
import altair as alt
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_validate, train_test_split
from sklearn import datasets
from sklearn.linear_model import Ridge
from sklearn.svm import SVC,SVR

import sys
sys.path.append('code/')
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Preprocessing and pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

import scipy
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer


# <a id="Exploratory"></a>
# # Exploratory Data Analysis
# We set all relevant parameters for our notebook. By convention, parameters are uppercase, while all the 
# other variables follow Python's guidelines.

# 
# ## Data import

# In[2]:


ab_nyc = pd.read_csv('AB_NYC_2019.csv')
#, encoding= "windows-1252")
ab_nyc.head(3)


# ## Data processing

# ### Understand the data

# In[3]:


ab_nyc.shape


# In[4]:


ab_nyc.dtypes


# In[5]:


ab_nyc.isnull().sum()


# Recall our goal is price prediction, the null values in the dataset will not affect our analysis (they are not part of the features we will use), but we are replacing the null values for completeness of the data cleaning process.

# ### Data imputation

# In[6]:


ab_nyc.fillna({'last_review':"NoReviews"}, inplace=True)
ab_nyc.fillna({'reviews_per_month':0}, inplace=True)
ab_nyc.fillna({'name':"NoName"}, inplace=True)
ab_nyc.fillna({'host_name':"NoName"}, inplace=True)


# In[7]:


#check the dataset again after imputation--no null values
ab_nyc.isnull().sum()


# In[8]:


ab_nyc.describe(include="all")


# In[9]:


len(ab_nyc)


# In[10]:


len(ab_nyc[ab_nyc['price']==0])


# We notice that minimum price is $0, which does't make sense. This is probably due to data collection error. There are only 11/48895 rows having this case, and we will remove these rows.

# In[11]:


ab_nyc1=ab_nyc[ab_nyc['price']!=0]
ab_nyc1.head()


# ### Drop unnecessary columns

# Once again, our goal is to help new Airbnb hosts to set up reasonable pricing for their listings; therefore, columns related to reviews will be dropped to make the model appplicable to **new listings**.

# Columns containing the listing id/name/host_id/host_name will be excluded from the modelling as well, because they simply provide the basic information of the listing, and this information such as id should not have effects on the pricing.

# In[12]:


ab_nyc1=ab_nyc1.drop(['number_of_reviews','last_review','reviews_per_month','name','id','host_id','host_name'], axis=1)


# We will now use ab_nyc1 for the further analysis.

# <a id="Visualization"></a>
# ## Visualization of the dataset

# In[13]:


numeric_features = ['latitude','longitude','minimum_nights','calculated_host_listings_count','availability_365']
categorical_features =['neighbourhood_group','neighbourhood','room_type']


# ### Splitting the data and use only the training set for Visualization

# In[14]:


train_df, test_df = train_test_split(ab_nyc1, test_size = 0.2, random_state = 123)

X_train = train_df.drop(columns=["price"])
y_train = train_df["price"]
X_test = test_df.drop(columns=["price"])
y_test = test_df["price"]


# ### 1. Price distribution 

# In[15]:


#the rows in the dataset is large so we need to disable the max_rows in Altair
alt.data_transformers.disable_max_rows()


# In[16]:


alt.Chart(train_df).mark_bar().encode(
    alt.X("price:Q",
          bin=alt.BinParams()),
    y='count(*):Q').properties(title= ('Target:' + 'price')
)


# The histogram is right skewed. We noticed that most listing prices fall within [1,1000]. We will zooom in and plot the histogram separately for under \\$1000 and over $1000 listings. 

# In[17]:


alt.Chart(train_df[["price"]][train_df["price"]<1000]).mark_bar().encode(
    alt.X("price:Q",
          bin=alt.BinParams()),
    y='count(*):Q').properties(title= ('Target:' + ' Zoom in price <$1000')
)#under $1000


# In[18]:


alt.Chart(train_df[["price"]][train_df["price"]>1000]).mark_bar().encode(
    alt.X("price:Q",
          bin=alt.BinParams()),
    y='count(*):Q').properties(title= ('Target:' + 'Zoom in price >$1000')
)#over $1000


# ### 2. log(price) distribution

# Since the original data for price does not follow the bell curve, we can log-transform this data to make it as normally distributed as possible. We try to use log transformation to reduce the skewness of the “price”.

# In[19]:


ab_nyc1['log_price']=np.log(ab_nyc1.price)


# In[20]:


train_df, test_df = train_test_split(ab_nyc1, test_size = 0.2, random_state = 123)

X_train = train_df.drop(columns=["price","log_price"])
y_train = train_df["log_price"]

X_test = test_df.drop(columns=["price","log_price"])
y_test = test_df["log_price"]

train_df.head(3)


# In[21]:


alt.Chart(train_df[["log_price"]]).mark_bar().encode(
    alt.X("log_price:Q",
          bin=alt.BinParams()),
    y='count(*):Q').properties(title= ('Target:' + 'log_price')
)


# The histogram looks more normally distributed.

# ### 3. Numeric Feature Distribution

# In[22]:


def plot_histogram(df,feature):
    """
    plots a histogram of the distribution of features

    Parameters
    ----------
    feature: str
        the feature name
    Returns
    -------
    altair.vegalite.v3.api.Chart
        an Altair histogram 
    """


   ## Creates a visualization named histogram that plots the feature on the x-axis
   ##  and the frequency/count on the y-axis and colouring by the income column
    histogram = alt.Chart(df).mark_bar(
        opacity=0.7).encode(
        alt.X(str(feature) + str(':O'), bin=alt.Bin(maxbins=30)),
        alt.Y('count()', stack=None)
        ).properties(title= ('Numeric Feature:' + feature)
    )
    return histogram


# In[23]:


figure_dict = dict()
for feature in numeric_features:
    #ab_nyc_1000 = ab_nyc_1000.sort_values('price')
    figure_dict.update({feature:plot_histogram(train_df,feature)})
figure_panel = alt.vconcat(*figure_dict.values())
figure_panel


# **Minimum_nights**<br>
# 
# Majority of the listing have a minimum booking of less than 10 days, with the most common policy being 1-2 days as the minimum of booking. From the correlation matrix, it is yet hard to see a strong relationship between this feature and price.<br>
# 
# **Calculated_host_listings_count**<br>
# 
# It is most common for a host to have only 1 Airbnb listings, but there are some extreme cases where one host has more than 20 listings.<br>
# 
# **Avalability_365**<br>
# 
# This feature represents s a number the hosts input when they register for the Airbnb listings. Most of the listings have less than 20 days a year as their availability. Besides that, it is yet hard to predict if more availability leads to higher price or the opposite. It is worth keeping this feature to test further in the model.<br>
# 

# ### 3. Categorical Feature Distribution

# In[24]:


alt.Chart(train_df).mark_bar().encode(
    x='neighbourhood_group',
    y='count()').properties(title= ('Categorical Feature:' + 'neighbourhood_group')
)


# In[25]:


alt.Chart(train_df).mark_bar().encode(
    x='neighbourhood',
    y='count()').properties(title= ('Categorical Feature:' + 'neighbourhood')
)


# Out of all neighborhoods, “Bedford-Stuyvesant” and “Bushwick” have the highest amount of Airbnb listings, both located in the centre of Brooklyn. Based on business sense, we know that the features of “neighborhood” and “neighborhood_group” have certain overlapping information, because neighborhood_group (similar district) includes multiple neighborhoods. We will test models with both features and only one of the features to identify if it is worth keeping both in the later part of this report.

# In[26]:


alt.Chart(train_df).mark_bar().encode(
    x='room_type',
    y='count()').properties(title= ('Categorical Feature:' + 'room_type')
)


# **Room Type**
# 
# It is observed that most listings are entire home/entire apartment or private room. Shared room listings exist but are much less compared with the first two room types. The average price of an entire home/apt is the highest (USD\\$212/night), followed by private room (USD\\$90/night) and shared room (USD\\$70/night).

# In[27]:


plt.figure(figsize=(10,10))
sns.scatterplot(x='longitude', y='latitude', hue='neighbourhood_group',s=20, data=ab_nyc1)


# In[28]:


ab_nyc1.groupby(['neighbourhood_group']).mean()


# **Longitude and latitude & neighbourhood_group**<br>
# 
# By drawing the map out on “longitude” and “latitude” data and group the Airbnb listings by “neighbourhood_group”, we observed that there are more listings in Brooklyn, Manhattan and Queens (more crowded), and it is less busy in neighbourhood groups such as Bronx and Staten Island.<br>
# 
# With this information, when we look at the average pricing of the different neighbourhood groups, we found that as expected Brooklyn and Manhattan have a higher listing price due to their central location and higher Airbnb competition.<br>
# 
# However, it is interesting that Staten Island also has a high average listing price, but the listings there are much more). Due to the above observations, we retentively decide to keep neighborhood_group as a feature in our model.
# 

# ## Data transformation

# We already performed data imputation. Now we want to do some transformation respectively for the numeric and categorical features.

# In[29]:


numeric_transformer = make_pipeline(SimpleImputer(strategy="median"),
                                    StandardScaler())

categorical_transformer = make_pipeline(
                SimpleImputer(strategy="constant", fill_value="missing"),
                OneHotEncoder(handle_unknown="ignore"))
                
preprocessor = make_column_transformer(
               (numeric_transformer, numeric_features), 
               (categorical_transformer, categorical_features))


# ## Corelation Matrix & Multicollinearity

# We want to see the correlation matrix and check multicollinearity.

# In[30]:


corr=ab_nyc1.corr()
corr


# In[31]:


plt.figure( figsize = (10,10))
sns.heatmap(ab_nyc1.corr(), annot = True)


# In[32]:


multicollinearity, V=np.linalg.eig(corr)
multicollinearity


# Based on the correlation matrix plot, it seems that there is no high correlation between the features that could cause potential multicollinearity issues. Thus, we will keep all these features to the next step – modeling: “Latitude”, “longitude”, “minimum_nights”, “calculated_host_listings_count”, “availability_365”.

# <a id="baseline_model"></a>
# 
# # Method & Results
# ## Baseline model

# This dummy classifier model uses the "most_frequent" strategy to decide a property's listing price. We want to see the scoring of this baseline model and further see how well our other models do.

# In[33]:


X=ab_nyc1.drop(columns=['price','log_price'])
y=ab_nyc1[['price']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)


# In[34]:


dummy_clf = DummyClassifier(strategy="most_frequent")


# In[35]:


dummy_clf.fit(X_train, y_train)
dummy_clf.score(X_train, y_train)#train score


# In[36]:


dummy_clf.score(X_test, y_test)#test score


# Now we want to try similarity-based algorithm and see if it is good to get the price prediction for the listings simply based on how similar they are.

# ## SVR model
# We developed a Support Vector Regression (SVR) model since it is a type of regression method, but different from SVM that is used for classification, [SVR](https://www.kdnuggets.com/2017/03/building-regression-models-support-vector-regression.html) can be used to predict numeric values. Different from the K-nn model, the SVR model remembers the key examples which makes it more efficient. We first defined a main pipeline that transformed all features and used default hyperparameters. Then, we did hyperparameter tuning to choose the best gamma and C for the model. We decided to use gamma value = 0.1 and C value = 100 for the final model, because they gave the highest validation score. The SVR model is underfitting since both training and test score are quite low (0.126 and 0.105 respectively). It also took a great amount of training time since the dataset has many features. Thus, we move on to try out other models to improve on the computing time and the model accuracy.

# In[37]:


#Define a main pipeline that transforms all the different features and uses an SVR model with default hyperparameters.        
svr_pipe = make_pipeline(preprocessor, SVR())


# In[38]:


X=ab_nyc1.drop(columns=['price','log_price'])
y=ab_nyc1['log_price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)


# In[40]:


#Hyperparameter tuning
param_grid = {
    "svr__gamma": [0.1, 1.0, 10, 100],
    "svr__C": [0.1, 1.0, 10, 100]
}

random_search = RandomizedSearchCV(svr_pipe, param_grid, cv=5, verbose=3, n_jobs=-1, n_iter=10, random_state = 42, return_train_score=True)
random_search.fit(X_train, y_train)


# In[41]:


pd.DataFrame(random_search.cv_results_)[["params",
                                         "mean_test_score", 
                                         "mean_train_score",
                                         "rank_test_score"]]


# In[42]:


best_score = random_search.best_score_
print("Best SVC model's validation score:", best_score)#0.5658486809334746 


# In[43]:


best_gamma=random_search.best_params_['svr__gamma']
best_c=random_search.best_params_['svr__C']
print('Best gamma:', best_gamma, 'Best C:', best_c)#Best gamma: 0.1 Best C: 10 


# We chose gamma value = 1.0 and C value = 10 for the final model, because they gave the higest validation score. The gap between the validation score and the test score is also very small. The scoring is better than the baseline model, but still not ideal.

# ## *K*-nn model

# We want to explore the k-NN model to find out if given the similarities in the features of the observations, we can predict a correct fair price for the Airbnb listings. It is important to highlight that one of the weaknesses of this approach is the Curse of Dimensionality. We want to be very specific with the features we will evaluate to keep this k-NN model meaningful. 
# The features that will be included in the model. For the k-nn model numeric features will include:
# 
# **Numeric features**
# * latitude
# * longitude
# * last_review
# * minimum_nights
# * calculated_host_listings_count
# * availability_365
# 
# **Categorical variables**
# * neighbourhood_group
# * neighbourhood
# * room_type
# 
# All this information is data that any new host can provide about their property, since it will include the location and the number of nights that the host is willing to offer the place in the upcoming 356 days.  
# Next step for this model is preparing the pipeline to handle numerical data by scaling values, and setting a median strategy; while categorical values will take a constant values for the missing values and will be processes wint One-Hot Encode (add valued per each value of the category), unknown values will be ignored. 

# In[44]:


knn_pipe = make_pipeline(preprocessor, KNeighborsRegressor())


# Next step is finding the best value for the parameter `n_neighbors` in the KNeighborsClassifier method, we will perform a random search with values from 11 to 21 to find the best parameter value for the model.

# In[45]:


k_range = list(range(11, 21))
knn_grid_params = {'kneighborsregressor__n_neighbors':k_range} 

knn_random_search = RandomizedSearchCV(knn_pipe, knn_grid_params, cv=5, verbose=4, n_jobs=-1, n_iter=5,random_state=123)
knn_random_search.fit(X_train, y_train)


# According to the random search the best parameters are:

# In[46]:


r = pd.DataFrame(knn_random_search.cv_results_)
r[r['rank_test_score']==1][['param_kneighborsregressor__n_neighbors','mean_test_score']]#check the output


# ## Regression model

# Linear regression models can help us to predict the pricing based on specific features, and the coefficients of the regression models can explain what features contribute positively or negatively to the listing. The hosts can use the results to consider which neighbourhood they want to invest because the area will bring more revenue.  
# In this section, we will examine the scores using price and log(price) as y-varibles separately to get an ideal model. 

# ### Use log(price) as target value
# We explored 4 regression modes in total to finalize our final model, which has the highest test and train scores, as well as the lowest error metrics.   
# Please see the **Appendixes** for the other 3 models we tried to address the 2 questions:      
# 1) Whether to include both neighbourhood and neighbourhood group, or one of the categorical variable is sufficient ?  
# 2) We want to try to apply ordinal encoding for the room_type variable and see if it improves the model fit; as it is reasonable to think that the general pricing trend will be: entire room > private room > shared room.
# 

# ###  Neighbourhood only--Our final model

# In[47]:


numeric_features = ['latitude','longitude','minimum_nights','calculated_host_listings_count','availability_365']
categorical_features_n =['neighbourhood','room_type']#our categorical features only include neighbourhood

preprocessor_n = make_column_transformer(
               (numeric_transformer, numeric_features), 
               (categorical_transformer, categorical_features_n))


# In[48]:


ab_nyc3=ab_nyc1.drop(columns=['neighbourhood_group'])
X=ab_nyc3.drop(columns=['price','log_price'])
y=ab_nyc3['log_price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)


# In[49]:


linreg_pipe_n= make_pipeline(preprocessor_n,LinearRegression())
linreg_pipe_n.fit(X_train,y_train)


# In[55]:


scoring={
    "r2": "r2",
    "neg_rmse": "neg_root_mean_squared_error",    
    "neg_mse": "neg_mean_squared_error",    
}

scores = cross_validate(linreg_pipe_n, X_train, y_train, cv=10, return_train_score=True, scoring =scoring)
scores_df = pd.DataFrame(scores)
scores_df


# In[56]:


scores_df.mean()


# In[57]:


print('R-squared train score: {:.3f}'.format(linreg_pipe_n.score(X_train, y_train)))
print('R-squared test score: {:.3f}'.format(linreg_pipe_n.score(X_test, y_test)))


# ### MAPE for the best model (Neighbourhood only)

# In[58]:


y_pred=linreg_pipe_n.predict(X_test)


# In[59]:


def mape(true, pred):
    return 100.*(np.abs((pred - true)/true)).mean()
mape(y_pred,y_test)


# On average, the final model has around 7.24% error in our predicted log(price), which is pretty satisfactory.
# 
# 

# ### Backtransform log(price) to price
# Because we used log(price) as our y variable, this is to show how to backtransform the y vairiable back to the original price in the original unit USD$.

# In[60]:


y_pred=linreg_pipe_n.predict(X_test)
y_pred_price=np.exp(y_pred)#transform the prediction back to the original units
y_pred_price


# ### Visualizing the predictions of log(price) and the real log(price) in the test set

# In[61]:


df = pd.DataFrame(y_test).assign(predicted = y_pred).rename(columns = {'log_price': 'true'})
plt.scatter(y_test, y_pred, alpha=0.3, s = 5)
grid = np.linspace(y_test.min(), y_test.max(), 1000)
plt.plot(grid, grid, '--k');
plt.xticks(fontsize= 12);
plt.yticks(fontsize= 12);
plt.xlabel("true log(price)", fontsize=14);
plt.ylabel("predicted log(price)", fontsize=14);


# This plot indicates that the predicted log(price) and the true log(price) are very similar--there are only a few outliers.

# ### Understand the coefficients of the best model (Neighbourhood only)

# In[62]:


onehot_columns = list(linreg_pipe_n.named_steps['columntransformer'].named_transformers_['pipeline-2'].named_steps['onehotencoder'].get_feature_names(input_features=categorical_features_n))#get the list of categorical features after transformation

numeric_features_list = list(numeric_features)#get the list of numeric features after transformation
numeric_features_list.extend(onehot_columns)#combine the list


# In[63]:


feats_coef = linreg_pipe_n.named_steps['linearregression'].coef_ #get the coefficients


# ### Top 5 neighbourhoods that the hosts can consider investing

# In[64]:


pd.DataFrame(data=feats_coef, index=numeric_features_list, columns=['Coefficients']).sort_values('Coefficients', ascending=False)[0:5]


# ### Top 5 neighbouhoods that the hosts should avoid investing

# In[65]:


pd.DataFrame(data=feats_coef, index=numeric_features_list, columns=['Coefficients']).sort_values('Coefficients', ascending=True)[0:5]


# ### Coefficients for numeric features

# In[66]:


pd.DataFrame(data=feats_coef, index=numeric_features_list, columns=['Coefficients']).loc[['latitude',
 'longitude',
 'minimum_nights',
 'calculated_host_listings_count',
 'availability_365']].sort_values('Coefficients', ascending=False)


# The more days of availability the host can offer, the higher the price of the listings. This flexibility can attract more customers even if they may need to pay a bit higher for the listing. 

# ### Coefficients for room type

# In[67]:


pd.DataFrame(data=feats_coef, index=numeric_features_list, columns=['Coefficients']).loc[['room_type_Private room','room_type_Shared room','room_type_Entire home/apt']].sort_values('Coefficients', ascending=False)


# Unsurprisingly, entire home/apt room type contributes a lot to the pricing. Hosts can expect higher price from this kind of properties.

# <a id="example"></a>
# ## Example of the model application

# A client comes to us and wants to get a reasonable price of her listing on Airbnb. The following is the information of her listing:  

# In[68]:


data = [['Brooklyn','Kensington',40.56735,-74.0245,'Private room',1,1,300]]
X_listing = pd.DataFrame(data, columns = ['neighbourhood_group','neighbourhood','latitude','longitude','room_type','minimum_nights','calculated_host_listings_count','availability_365'] )
X_listing


# ### 1) SVR model application

# In[69]:


svr_best_model = random_search.best_estimator_
svr_y=svr_best_model.predict(X_listing)
np.exp(svr_y)#124.81400468


#  ### 2)*k*-nn model application

# In[70]:


knn_best_model = knn_random_search.best_estimator_
knn_y=knn_best_model.predict(X_listing)
np.exp(knn_y)#check the output


# ### 3) Regression model application

# In[74]:


linreg_data = [['Kensington',40.56735,-74.0245,'Private room',1,1,300]]#remove neighbourhood group  
linreg_X = pd.DataFrame(linreg_data, columns = ['neighbourhood','latitude','longitude','room_type','minimum_nights','calculated_host_listings_count','availability_365'] )


# In[76]:


linreg_y=linreg_pipe_n.predict(linreg_X)
np.exp(linreg_y)#backtransform log(price) to price


# According to our final model, we will advise the owner to set her listing at USD\\$87/night compared with the market price.
# We are confident that this final model we developed could be of help for future Airbnb hosts to evaluate their properties and decide on a price that is reasonable for the market. SVR model tends to give a higher price (USD\\$125/night), while *k*-nn model gives a lower price of USD\\$78/night.   
# A great advantage to the regression model is its scalability in the sense that it can be adapted to almost any city in the USA, or even any region in the world, as long as the dataset has those features. As discussed at the beginning of the report, more Airbnb listings data can be obtained at [Inside Airbnb](http://insideairbnb.com/get-the-data.html).

# <a id="reference"></a>
# 
# # References
# 
# * Airbnb. (February 17, 2021). Report: New Airbnb Hosts Have Earned $1 Billion During the Pandemic.
# * Concepta. (September, 18, 2018). Price Optimization: How Dynamic Pricing Helps Airbnb Hosts Earn Big. https://www.conceptatech.com/blog/price-optimization-how-dynamic-pricing-helps-airbnb-hosts-earn-big.
# * Kdnuggets. (March, 2017). Building Regression Models Support Vector Regression. https://www.kdnuggets.com/2017/03/building-regression-models-support-vector-regression.html
# * Inside Airbnb. http://insideairbnb.com/get-the-data.html

# <a id="appendixes"></a>
# 
# # Appendixes
# 

# Besides the final linear regression model we decided, we also tried a few other models using linear regression as following:
# ### 1. Use Price as target value

# In[77]:


X=ab_nyc1.drop(columns=['price','log_price'])
y=ab_nyc1['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)


# In[78]:


linreg_pipe= make_pipeline(preprocessor,LinearRegression())
linreg_pipe.fit(X_train,y_train)


# In[79]:


y_pred=linreg_pipe.predict(X_test)#get the predictions for the test set


# In[80]:


print('R-squared train score: {:.3f}'.format(linreg_pipe.score(X_train, y_train)))
print('R-squared test score: {:.3f}'.format(linreg_pipe.score(X_test, y_test)))


# In[81]:


scoring={
    "r2": "r2",
    "neg_rmse": "neg_root_mean_squared_error",    
    "neg_mse": "neg_mean_squared_error",    
}


# In[82]:


scores = cross_validate(linreg_pipe, X_train, y_train, cv=10, return_train_score=True, scoring =scoring)
scores_df = pd.DataFrame(scores)
scores_df


# In[83]:


scores_df.mean()


# The scoring using price as the y variable in the linear regression is quite low and underfitting the data. We notice the erros are quite large, indicating the forecasts are inaccurate. Next we will try using log(price) as the y variable.

# ### 2. Use log(price) as target value
# ### a) Neighbourhood group only

# In[84]:


numeric_features = ['latitude','longitude','minimum_nights','calculated_host_listings_count','availability_365']
categorical_features_ng =['neighbourhood_group','room_type']#our categorical features contains neighbourhood group only


# In[85]:


preprocessor_ng = make_column_transformer(
               (numeric_transformer, numeric_features), 
               (categorical_transformer, categorical_features_ng))


# In[86]:


ab_nyc2=ab_nyc1.drop(columns=['neighbourhood'])


# In[87]:


X=ab_nyc2.drop(columns=['price','log_price'])
y=ab_nyc2['log_price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)


# In[88]:


linreg_pipe_ng= make_pipeline(preprocessor_ng,LinearRegression())
linreg_pipe_ng.fit(X_train,y_train)


# In[89]:


scores = cross_validate(linreg_pipe_ng, X_train, y_train, cv=10, return_train_score=True, scoring =scoring)
scores_df = pd.DataFrame(scores)
scores_df


# In[90]:


scores_df.mean()


# In[91]:


print('R-squared train score: {:.3f}'.format(linreg_pipe_ng.score(X_train, y_train)))
print('R-squared test score: {:.3f}'.format(linreg_pipe_ng.score(X_test, y_test)))


# ### b) Including both Neighbourhood and Neighbourhood group

# In[92]:


numeric_features = ['latitude','longitude','minimum_nights','calculated_host_listings_count','availability_365']
categorical_features_nng =['neighbourhood_group','neighbourhood','room_type']#categorical features include both neighbourhood and neighbourhood group

preprocessor_nng = make_column_transformer(
               (numeric_transformer, numeric_features), 
               (categorical_transformer, categorical_features_nng))


# In[93]:


X=ab_nyc1.drop(columns=['price','log_price'])
y=ab_nyc1['log_price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)


# In[94]:


linreg_pipe_nng= make_pipeline(preprocessor_nng,LinearRegression())
linreg_pipe_nng.fit(X_train,y_train)


# In[95]:


scores = cross_validate(linreg_pipe_nng, X_train, y_train, cv=10, return_train_score=True, scoring =scoring)
scores_df = pd.DataFrame(scores)
scores_df


# In[96]:


scores_df.mean()


# In[97]:


print('R-squared train score: {:.3f}'.format(linreg_pipe_nng.score(X_train, y_train)))
print('R-squared test score: {:.3f}'.format(linreg_pipe_nng.score(X_test, y_test)))


# Comparing the validation, training and test scores and the RMSE, MSE for the 3 linear regression models, we found that neighbourhood plays an important role in the modelling. It provides more granular categories, which leads to more accurate forecasts. Including both neighbourhood and neighbourhood group does not help improve the scores; therefore,our final model **includes neighbourhood only**.

# ### c) Convert "room_type" using ordinal encoding

# As mentioned before, the general price levels are: entire home/apt >prive room > shared room. So we want to try ordinal encoding for the room_type.

# In[98]:


ab_nyc1.groupby(['room_type']).mean()


# In[99]:


# encoding "room type" differently
categorical_features =['neighbourhood_group',
                       'neighbourhood']
categorical_features_ord = ['room_type']

numeric_transformer = Pipeline(
                     [("scaler",StandardScaler())])
categorical_transformer = Pipeline(
                      [("onehot", OneHotEncoder(dtype=int,handle_unknown="ignore"))])
categorical_features_ord_transformer = Pipeline(
                      [("ordinal", OrdinalEncoder(categories=[['Shared room', 'Private room', 'Entire home/apt']]))])

preprocessor = ColumnTransformer(
    transformers=[("numeric", numeric_transformer, numeric_features),
         ("categorical", categorical_transformer, categorical_features),
         ("categorical_ord", categorical_features_ord_transformer, categorical_features_ord)        
                 ], 
    remainder='drop'    
)


# In[100]:


# regression model with "room type" encoded as ordinal encoding
X=ab_nyc1.drop(columns=['price','log_price'])
y=ab_nyc1['log_price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
ord_linreg_pipe= make_pipeline(preprocessor,LinearRegression())
ord_linreg_pipe.fit(X_train,y_train)


# In[101]:


scores = cross_validate(linreg_pipe, X_train, y_train, cv=10, return_train_score=True, scoring =scoring)
scores_df = pd.DataFrame(scores)
scores_df
scores_df.mean()


# In[102]:


print('R-squared train score: {:.3f}'.format(ord_linreg_pipe.score(X_train, y_train)))
print('R-squared test score: {:.3f}'.format(ord_linreg_pipe.score(X_test, y_test)))


# It turns out although in business sense, room type could potentially be converted using ordinal encoding, the scoring and error metrics are not as good as the linear regression model b). 

# ### Let's see what the other 3 log(price) models do in the prediction of this listing.

# #### Neighbourhood group model

# In[103]:


data = [['Brooklyn',40.56735,-74.0245,'Private room',1,1,300]] 
X_listing = pd.DataFrame(data, columns = ['neighbourhood_group','latitude','longitude','room_type','minimum_nights','calculated_host_listings_count','availability_365'] ) 
X_listing

y_log=linreg_pipe_ng.predict(X_listing)
np.exp(y_log)


# This model will advise a higher price than our final model, which could harm the client's competency-- her listing will be less attractive to the customers.

# #### Neighbourhood and Neighbourhood group model

# In[104]:


data = [['Brooklyn','Kensington',40.56735,-74.0245,'Private room',1,1,300]]
X_listing = pd.DataFrame(data, columns = ['neighbourhood_group','neighbourhood','latitude','longitude','room_type','minimum_nights','calculated_host_listings_count','availability_365'] )
  
X_listing
y_log=linreg_pipe_nng.predict(X_listing)
np.exp(y_log)


# As expected, this model gives very similar price prediction as our final model.  

# #### Convert "room_type" using ordinal encoding

# In[105]:


y_ord=ord_linreg_pipe.predict(X_listing)
np.exp(y_ord)


# *Recall the SVR and KNN models*, they gave us USD\\$125/night and USD\\$78/night for this same property. Setting the price too much higher or lower is not ideal for maximizing the client's competency and profit.
