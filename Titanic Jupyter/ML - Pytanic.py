#!/usr/bin/env python
# coding: utf-8

# In[64]:


import pandas as pd

df = pd.read_csv("train.csv")
#df_test = pd.read_csv("train.csv")

#print(df.shape)
#print(df.size)


# In[65]:


#df.head(5)

#df.columns


# In[67]:


#print(df.info())
#print(df.describe())


# In[68]:


#df.isnull().sum()


# In[69]:


def get_title(name):
    if '.' in name:
        return name.split(',')[1].split('.')[0].strip()
    else:
        return 'Unknown'

# A list with the all the different titles
#titles = sorted(set([x for x in df.Name.map(lambda x: get_title(x))]))
#print(len(titles), ":", titles)
#print()


# In[70]:


# Normalize the titles
def replace_titles(x):
    title = x['Title']
    if title in ['Capt', 'Col', 'Major']:
        return 'Officer'
    elif title in ["Jonkheer","Don",'the Countess', 'Dona', 'Lady',"Sir"]:
        return 'Royalty'
    elif title in ['the Countess', 'Mme', 'Lady']:
        return 'Mrs'
    elif title in ['Mlle', 'Ms']:
        return 'Miss'
    else:
        return title


# In[71]:


# Lets create a new column for the titles
df['Title'] = df['Name'].map(lambda x: get_title(x))
# train.Title.value_counts()
# train.Title.value_counts().plot(kind='bar')

# And replace the titles, so the are normalized to 'Mr', 'Miss' and 'Mrs'
df['Title'] = df.apply(replace_titles, axis=1)
#print(df.Title.value_counts())


# In[72]:


df['Age'].fillna(df['Age'].median(), inplace=True)
df['Fare'].fillna(df['Fare'].median(), inplace=True)
df['Embarked'].fillna("S", inplace=True)
df.drop("Cabin", axis=1, inplace=True)
df.drop("Ticket", axis=1, inplace=True)
df.drop("Name", axis=1, inplace=True)
df.Sex.replace(('male','female'), (0,1), inplace = True)
df.Embarked.replace(('S','C','Q'), (0,1,2), inplace = True)
df.Title.replace(('Mr','Miss','Mrs','Master','Dr','Rev','Officer','Royalty'), (0,1,2,3,4,5,6,7), inplace = True)


# In[73]:


#print(df.isnull().sum())
#print(df['Sex'].sample(5))
#print(df['Embarked'].sample(5))
#print(df.columns)


# In[74]:


#corr = df.corr()
#corr.Survived.sort_values(ascending=False)


# In[75]:


from sklearn.model_selection import train_test_split

x = df.drop(["Survived", "PassengerId"], axis=1)
y = df["Survived"]

x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.1) 


# In[76]:


import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

randomforest = RandomForestClassifier()
randomforest.fit(x_train, y_train)
#y_pred = randomforest.predict(x_val)
#acc_randomforest = round(accuracy_score(y_pred, y_val) * 100, 2)
#print("Accuracy: {}".format(acc_randomforest))

pickle.dump(randomforest, open('titanic_model.sav', 'wb'))


# In[80]:

'''
df_test = pd.read_csv("test.csv")
# Lets create a new column for the titles
df_test['Title'] = df_test['Name'].map(lambda x: get_title(x))
# train.Title.value_counts()
# train.Title.value_counts().plot(kind='bar')

# And replace the titles, so the are normalized to 'Mr', 'Miss' and 'Mrs'
df_test['Title'] = df_test.apply(replace_titles, axis=1)
ids = df_test['PassengerId']

df_test['Age'].fillna(df_test['Age'].median(), inplace=True)
df_test['Fare'].fillna(df_test['Fare'].median(), inplace=True)
df_test['Embarked'].fillna("S", inplace=True)
df_test.drop("Cabin", axis=1, inplace=True)
df_test.drop("Ticket", axis=1, inplace=True)
df_test.drop("Name", axis=1, inplace=True)
df_test.drop("PassengerId", axis=1, inplace=True)
df_test.Sex.replace(('male','female'), (0,1), inplace = True)
df_test.Embarked.replace(('S','C','Q'), (0,1,2), inplace = True)
df_test.Title.replace(('Mr','Miss','Mrs','Master','Dr','Rev','Officer','Royalty'), (0,1,2,3,4,5,6,7), inplace = True)

'''
# In[83]:


#predictions = randomforest.predict(df_test)
#output = pd.DataFrame({'PassengerId': ids, "Survived": predictions})
#output.to_csv('submission.csv', index=False)


# In[ ]:

def prediction_model(pclass, sex, age, sibsp, parch, fare, embarked, title):
    import pickle

    x = [[pclass, sex, age, sibsp, parch, fare, embarked, title]]
    
    randomforest = pickle.load(open("titanic_model.sav", "rb"))
    predictions = randomforest.predict(x)
    print(predictions)


