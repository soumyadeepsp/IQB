#!/usr/bin/env python
# coding: utf-8

# In[13]:


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


# In[14]:


acids = ['A','R','N','D','C','Q','E','G','H','I','L','K','M','F','P','S','T','W','Y','V']

features = []
labels = []

for r in rows:
    try:
        labels.append(int(r[1]))
        cur_seq = r[2]

        cur_feature = []
        for acid1 in acids:
            for acid2 in acids:
                cur_feature.append(cur_seq.count(acid1+acid2)/400*100)
        features.append(cur_feature)
    except:
        print("error")

features = np.array(features)
labels = np.array(labels)
print(features)


# In[8]:


# from sklearn.model_selection import GridSearchCV
# from sklearn import svm

# Cs = [0.001, 0.01, 0.1, 1, 10]
# gammas = [0.001, 0.01, 0.1, 1]
# param_grid = {'C': Cs, 'gamma' : gammas}
# grid_search = GridSearchCV(svm.SVC(kernel='rbf'), param_grid, cv=3)
# grid_search.fit(features, labels)
# print(grid_search.best_params_)


# In[15]:


from sklearn.svm import SVC # "Support Vector Classifier" 
clf = SVC(kernel='rbf',  decision_function_shape='ovo')

# # from sklearn.ensemble import RandomForestClassifier
# clf = RandomForestClassifier(n_estimators=1600, min_samples_split=5, min_samples_leaf=1, max_features = 'sqrt', max_depth=70, bootstrap=False)

# from sklearn.neural_network import MLPClassifier
# clf = MLPClassifier(hidden_layer_sizes=(150,100,50), max_iter=300)
  
# fitting x samples and y classes 
clf.fit(features, labels)


# In[16]:


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
        
#         if (len(r[1]) > 10):
#             cur_seq = r[1][:10]

        cur_feature = []
        for acid1 in acids:
            for acid2 in acids:
                cur_feature.append(cur_seq.count(acid1+acid2)/400*100)
        features_test.append(cur_feature)
    except:
        print("error")

features_test = np.array(features_test)
print(ids_test)
print(features_test)


# In[17]:


labels_test = clf.predict(features_test)

fields = ['ID', 'Label']
results = [list(a) for a in zip(ids_test, labels_test)]

with open('dipeptide_predictions.csv', 'w') as csvfile: 
    # creating a csv writer object 
    csvwriter = csv.writer(csvfile) 
      
    # writing the fields 
    csvwriter.writerow(fields) 
      
    # writing the data rows 
    csvwriter.writerows(results)


# 
