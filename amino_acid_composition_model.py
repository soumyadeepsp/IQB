#!/usr/bin/env python
# coding: utf-8

# In[9]:


import numpy as np
import csv
file = './train.csv'

fields = [] 
rows = [] 
  
# reading csv file 
with open(file, 'r') as csvfile: 
    # creating a csv reader object 
    csvreader = csv.reader(csvfile) 
      
    # extracting field names through first row 
    # fields = csvreader.next() 
  
    # extracting each data row one by one 
    for row in csvreader: 
        rows.append(row) 
    
rows = rows[1:]
print(rows)

# Any results you write to the current directory are saved as output.


# In[10]:


acids = ['A','R','N','D','C','Q','E','G','H','I','L','K','M','F','P','S','T','W','Y','V']

features = []
labels = []

for r in rows:
    try:
        labels.append(int(r[1]))
        cur_seq = r[2]

        cur_feature = []
        for acid in acids:
            cur_feature.append(cur_seq.count(acid))
        features.append(cur_feature)
    except:
        print("error")

features = np.array(features)
labels = np.array(labels)
print(features)


# In[4]:


# # Number of trees in random forest
# n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# # Number of features to consider at every split
# max_features = ['auto', 'sqrt']
# # Maximum number of levels in tree
# max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
# max_depth.append(None)
# # Minimum number of samples required to split a node
# min_samples_split = [2, 5, 10]
# # Minimum number of samples required at each leaf node
# min_samples_leaf = [1, 2, 4]
# # Method of selecting samples for training each tree
# bootstrap = [True, False]
# # Create the random grid
# random_grid = {'n_estimators': n_estimators,
#                'max_features': max_features,
#                'max_depth': max_depth,
#                'min_samples_split': min_samples_split,
#                'min_samples_leaf': min_samples_leaf,
#                'bootstrap': bootstrap}

# print(random_grid)
# from sklearn.model_selection import RandomizedSearchCV
# from sklearn.ensemble import RandomForestClassifier
# clf = RandomForestClassifier()
# rf_random = RandomizedSearchCV(estimator = clf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
# rf_random.fit(features, labels)


# # In[5]:


# print(rf_random.best_params_)


# In[11]:


from sklearn.svm import SVC # "Support Vector Classifier" 
clf = RandomForestClassifier(n_estimators=1800, min_samples_split=2, min_samples_leaf=1, max_features = 'auto', max_depth=20, bootstrap=False)
  
# fitting x samples and y classes 
clf.fit(features, labels)


# In[12]:


features_test = []

file_test = './test.csv'

rows_test = []


with open(file_test, 'r') as csvfile: 
    # creating a csv reader object 
    csvreader = csv.reader(csvfile) 
      
    # extracting field names through first row 
    # fields = csvreader.next() 
  
    # extracting each data row one by one 
    for row in csvreader: 
        rows_test.append(row) 
    
rows_test = rows_test[1:]

ids_test = []
for r in rows_test:
    try:
        ids_test.append(r[0])
        cur_seq = r[1]

        cur_feature = []
        for acid in acids:
            cur_feature.append(cur_seq.count(acid))
        features_test.append(cur_feature)
    except:
        print("error")

features_test = np.array(features_test)
print(ids_test)
print(features_test)


# In[14]:


labels_test = clf.predict(features_test)

fields = ['ID', 'Label']
results = [list(a) for a in zip(ids_test, labels_test)]

with open('amino_predictions.csv', 'w') as csvfile: 
    # creating a csv writer object 
    csvwriter = csv.writer(csvfile) 
      
    # writing the fields 
    csvwriter.writerow(fields) 
      
    # writing the data rows 
    csvwriter.writerows(results)


# In[ ]:




