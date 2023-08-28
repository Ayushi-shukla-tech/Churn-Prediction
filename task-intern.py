#!/usr/bin/env python
# coding: utf-8

# In[149]:


get_ipython().system('pip install numpy')
import numpy as np


# In[150]:


pip install seaborn


# In[151]:


pip install -U scikit-learn


# In[152]:


pip install matplotlib


# In[153]:


pip install plotly


# In[154]:


pip install statsmodels


# In[155]:


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


# In[156]:


get_ipython().system('pip install pandas')
import pandas as pd


# In[157]:


df = pd.read_csv('customer_churn_large_dataset-Copy1.csv')


# In[158]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
pd.set_option('display.max_columns', None)

import plotly.express as px #for visualization
import matplotlib.pyplot as plt #for visualization 

def dataoveriew(df, message):
    print(f'{message}:n')
    print('Number of rows: ', df.shape[0])
    print("nNumber of features:", df.shape[1])
    print("nData Features:")
    print(df.columns.tolist())
    print("nMissing values:", df.isnull().sum().values.sum())
    print("nUnique values:")
    print(df.nunique())

dataoveriew(df, 'Overview of the dataset')


# In[159]:


df.head()


# In[160]:


df.min()


# In[161]:


df.max()


# In[162]:


target_instance = df["Churn"].value_counts().to_frame()
target_instance = target_instance.reset_index()
target_instance = target_instance.rename(columns={'index': 'Category'})
fig = px.pie(target_instance, values='Churn', names='Category', color_discrete_sequence=["green", "red"],
             title='Distribution of Churn')
fig.show()


# In[163]:


#Defining bar chart function
def bar(feature, df=df ):
    #Groupby the categorical feature
    temp_df = df.groupby([feature, 'Churn']).size().reset_index()
    temp_df = temp_df.rename(columns={0:'Count'})
    #Calculate the value counts of each distribution and it's corresponding Percentages
    value_counts_df = df[feature].value_counts().to_frame().reset_index()
    categories = [cat[1][0] for cat in value_counts_df.iterrows()]
    #Calculate the value counts of each distribution and it's corresponding Percentages
    num_list = [num[1][1] for num in value_counts_df.iterrows()]
    div_list = [element / sum(num_list) for element in num_list]
    percentage = [round(element * 100,1) for element in div_list]
    #Defining string formatting for graph annotation
    #Numeric section
    def num_format(list_instance):
        formatted_str = ''
        for index,num in enumerate(list_instance):
            if index < len(list_instance)-2:
                formatted_str=formatted_str+f'{num}%, ' #append to empty string(formatted_str)
            elif index == len(list_instance)-2:
                formatted_str=formatted_str+f'{num}% & '
            else:
                formatted_str=formatted_str+f'{num}%'
        return formatted_str
    #Categorical section
    def str_format(list_instance):
        formatted_str = ''
        for index, cat in enumerate(list_instance):
            if index < len(list_instance)-2:
                formatted_str=formatted_str+f'{cat}, '
            elif index == len(list_instance)-2:
                formatted_str=formatted_str+f'{cat} & '
            else:
                formatted_str=formatted_str+f'{cat}'
        return formatted_str
    #Running the formatting functions
    num_str = num_format(percentage)
    cat_str = str_format(categories)

    #Setting graph framework
    fig = px.bar(temp_df, x=feature, y='Count', color='Churn', title=f'Churn rate by {feature}', barmode="group", color_discrete_sequence=["green", "red"])
    fig.add_annotation(
                text=f'Value count of distribution of {cat_str} are<br>{num_str} percentage respectively.',
                align='left',
                showarrow=False,
                xref='paper',
                yref='paper',
                x=1.4,
                y=1.3,
                bordercolor='black',
                borderwidth=1)
    fig.update_layout(
        # margin space for the annotations on the right
        margin=dict(r=400),
    )

    return fig.show()


# In[164]:


bar('Gender')
bar('Location')
bar('Age')


# In[165]:


bar('Total_Usage_GB')
bar('Subscription_Length_Months')
bar('Monthly_Bill')


# In[166]:


df.dtypes


# In[167]:


def hist(feature):
    group_df = df.groupby([feature, 'Churn']).size().reset_index()
    group_df = group_df.rename(columns={0: 'Count'})
    fig = px.histogram(group_df, x=feature, y='Count', color='Churn', marginal='box', title=f'Churn rate frequency to {feature} distribution', color_discrete_sequence=["green", "red"])
    fig.show()


# In[168]:


hist('Subscription_Length_Months')
hist('Monthly_Bill')
hist('Total_Usage_GB')


# In[169]:


#Create an empty dataframe
bin_df = pd.DataFrame()

#Update the binning dataframe
bin_df['tenure_bins'] =  pd.qcut(df['Subscription_Length_Months'], q=3, labels= ['low', 'medium', 'high'])
bin_df['MonthlyCharges_bins'] =  pd.qcut(df['Monthly_Bill'], q=3, labels= ['low', 'medium', 'high'])
bin_df['TotalCharges_bins'] =  pd.qcut(df['Total_Usage_GB'], q=3, labels= ['low', 'medium', 'high'])
bin_df['Churn'] = df['Churn']

#Plot the bar chart of the binned variables
bar('tenure_bins', bin_df)
bar('MonthlyCharges_bins', bin_df)
bar('TotalCharges_bins', bin_df)


# In[170]:


df.drop(["Name"],axis=1,inplace = True)


# In[171]:


import matplotlib.pyplot as plt

# Replace the placeholder values with the actual counts for each location
Los_Angeles = (df[df.Location == 'Los Angeles']).shape[0]
New_York = (df[df.Location == 'New York']).shape[0]
Miami = (df[df.Location == 'Miami']).shape[0]
Chicago = (df[df.Location == 'Chicago']).shape[0]
Houston = (df[df.Location == 'Houston']).shape[0]

plt.figure(figsize=(8, 5))

labels = ['Los Angeles', 'New York', 'Miami', 'Chicago', 'Houston']
colors = ['#abcdef', '#aabbcc', '#11A5AA', '#1CA53B', '#FF5733']

plt.pie([Los_Angeles, New_York, Miami, Chicago, Houston], labels=labels, colors=colors, autopct='%.2f%%')

plt.title('Different Locations of Clients')

plt.show()


# In[172]:


plt.figure(figsize=(10, 6))
sns.barplot(x='Total_Usage_GB', y='Churn', data=df, ci=None)
plt.xticks(rotation=45)
plt.show()


# In[173]:


import pandas as pd

# Calculate ratio of Monthly_Bill to Subscription_Length_Months
#df['Bill_Per_Month'] = df['Monthly_Bill'] / df['Subscription_Length_Months']

# Calculate ratio of Total_Usage_GB to Subscription_Length_Months
df['Usage_Per_Month'] = df['Total_Usage_GB'] / df['Subscription_Length_Months']

# Defining Senior Citizen
df['Senior_Citizen'] = df['Age'] > 60
df['Senior_Citizen'] = df['Senior_Citizen'].astype(int)

# Total BIll
df['Total_Charges'] = df['Monthly_Bill'] * df['Subscription_Length_Months']

# Charges per GB
df['Cost_Per_GB'] = df['Total_Usage_GB'] / df['Total_Charges']

# Monthly Usage Trend
#df['Monthly_Usage_Trend'] = df.groupby('CustomerID')['Total_Usage_GB'].diff()

# Long Term Customer
df['Long_Term_Customer'] = (df['Subscription_Length_Months'] > 12).astype(int)

# Usage Behaviour
usage_bins = [0, 50, 150, float('inf')]
usage_labels = ['Light', 'Moderate', 'Heavy']
df['Usage_Behavior'] = pd.cut(df['Total_Usage_GB'], bins=usage_bins, labels=usage_labels)

# Average Monthly Usage per Location
df['Avg_Monthly_Usage_Per_Location'] = df.groupby('Location')['Total_Usage_GB'].transform('mean')

# Usage Change Percentage:
df['Usage_Change_Percentage'] = df.groupby('CustomerID')['Total_Usage_GB'].pct_change() * 100

# Average Monthly Usage per Location
df['Avg_Monthly_Usage_Per_Location'] = df.groupby('Location')['Total_Usage_GB'].transform('mean')

# Usage Variation
df['Usage_Variation'] = df.groupby('CustomerID')['Total_Usage_GB'].transform('std')

# Price Sensitivity:
df['Price_Sensitivity'] = df['Monthly_Bill'] / df['Subscription_Length_Months']





# Print the first few rows of the updated dat
df.head()


# In[174]:


df.isnull().sum()


# In[175]:


df.drop(["CustomerID"],axis=1,inplace = True)
df.drop(["Usage_Change_Percentage"],axis=1,inplace = True)
df.drop(["Usage_Variation"],axis=1,inplace = True)


# In[176]:


df.drop(["Usage_Per_Month"],axis=1,inplace = True)
df.drop(["Cost_Per_GB"],axis=1,inplace = True)
df.drop(["Long_Term_Customer"],axis=1,inplace = True)
df.drop(["Avg_Monthly_Usage_Per_Location"],axis=1,inplace = True)
df.drop(["Price_Sensitivity"],axis=1,inplace = True)
df.drop(["Usage_Behavior"],axis=1,inplace = True)


# In[177]:


df.head()


# In[178]:


# Encode categorical features

#Defining the map function
def binary_map(feature):
    return feature.map({'Yes':1, 'No':0})

## Encoding target feature
#df['Churn'] = df[['Churn']].apply(binary_map)

# Encoding gender category
df['Gender'] = df['Gender'].map({'Male':1, 'Female':0})

#Encoding other binary category
#binary_list = ['Location']
#df[binary_list] = df[binary_list].apply(binary_map)

#Encoding the other categoric features with more than two categories
df = pd.get_dummies(df, drop_first=True)


# In[179]:


df.head()


# In[180]:


corr = df.corr()

fig = px.imshow(corr,width=1000, height=1000)
fig.show()


# In[181]:


import statsmodels.api as sm
import statsmodels.formula.api as smf

#Change variable name separators to '_'
all_columns = [column.replace(" ", "_").replace("(", "_").replace(")", "_").replace("-", "_") for column in df.columns]

#Effect the change to the dataframe column names
df.columns = all_columns

#Prepare it for the GLM formula
glm_columns = [e for e in all_columns if e not in ['CustomerID', 'Churn']]
glm_columns = ' + '.join(map(str, glm_columns))

#Fiting it to the Generalized Linear Model
glm_model = smf.glm(formula=f'Churn ~ {glm_columns}', data=df, family=sm.families.Binomial())
res = glm_model.fit()
print(res.summary())


# In[182]:


np.exp(res.params)


# In[183]:


df_scaled = df.copy()


# In[186]:


#feature scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
df['Subscription_Length_Months'] = sc.fit_transform(df[['Subscription_Length_Months']])
df['Monthly_Bill'] = sc.fit_transform(df[['Monthly_Bill']])
df['Total_Usage_GB'] = sc.fit_transform(df[['Total_Usage_GB']])


# In[201]:


df.columns


# In[187]:


df.head()


# In[188]:


df.head()


# In[189]:


# Import Machine learning algorithms
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

#Import metric for performance evaluation
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

#Split data into train and test sets
from sklearn.model_selection import train_test_split
X = df.drop('Churn', axis=1)
y = df['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=50)

#Defining the modelling function
def modeling(alg, alg_name, params={}):
    model = alg(**params) #Instantiating the algorithm class and unpacking parameters if any
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    #Performance evaluation
    def print_scores(alg, y_true, y_pred):
        print(alg_name)
        acc_score = accuracy_score(y_true, y_pred)
        print("accuracy: ",acc_score)
        pre_score = precision_score(y_true, y_pred)
        print("precision: ",pre_score)
        rec_score = recall_score(y_true, y_pred)
        print("recall: ",rec_score)
        f_score = f1_score(y_true, y_pred, average='weighted')
        print("f1_score: ",f_score)

    print_scores(alg, y_test, y_pred)
    return model

# Running logistic regression model
log_model = modeling(LogisticRegression, 'Logistic Regression')


# In[190]:


from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold
log = LogisticRegression()
rfecv = RFECV(estimator=log, cv=StratifiedKFold(10, random_state=50, shuffle=True), scoring="accuracy")
rfecv.fit(X, y)


# In[191]:


plt.figure(figsize=(8, 6))
plt.plot(range(1, len(rfecv.cv_results_['mean_test_score']) + 1), rfecv.cv_results_['mean_test_score'])
plt.grid()
plt.xticks(range(1, X.shape[1] + 1))
plt.xlabel("Number of Selected Features")
plt.ylabel("CV Score")
plt.title("Recursive Feature Elimination (RFE)")
plt.show()

print("The optimal number of features: {}".format(rfecv.n_features_))


# In[192]:


X_rfe = X.iloc[:, rfecv.support_]

# Overview of the optimal features in comparison with the initial dataframe
print("X dimension: {}".format(X.shape))
print("X column list:", X.columns.tolist())
print("X_rfe dimension: {}".format(X_rfe.shape))
print("X_rfe column list:", X_rfe.columns.tolist())


# In[193]:


# Running logistic regression model
log_model = modeling(LogisticRegression, 'Logistic Regression Classification')


# In[194]:


#svc_model = modeling(SVC, 'SVC Classification')


# In[195]:


rf_model = modeling(RandomForestClassifier, "Random Forest Classification")


# In[196]:


dt_model = modeling(DecisionTreeClassifier, "Decision Tree Classification")


# In[197]:


nb_model = modeling(GaussianNB, "Naive Bayes Classification")


# # Model Optimization

# In[202]:


# Convert the column names to a list
feature_names_list = list(X_train.columns)

# Visualize the Decision Tree
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

plt.figure(figsize=(80, 20))
plot_tree(dt_model, feature_names=feature_names_list, max_depth=2, filled=True)


# In[204]:


for max_d in range(1,26):
  model = DecisionTreeClassifier(max_depth=max_d, random_state=42)
  model.fit(X_train, y_train)
  print('The Accuracy for max_depth {} is:'.format(max_d), model.score(X_train, y_train))
  print('')


# In[205]:


#Saving best model 
import joblib
#Sava the model to disk
filename = 'model.sav'
joblib.dump(model, filename)


# In[206]:


import pickle
pickle_out = open("model.pkl","wb")
pickle.dump(model, pickle_out)
pickle_out.close()


# In[207]:


import joblib

# Save model
joblib.dump(model, "model.pkl")


# In[208]:


df.dtypes


# In[ ]:




