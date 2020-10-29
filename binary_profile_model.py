#!/usr/bin/env python
# coding: utf-8

# In[8]:


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


# In[9]:


acids = ['A','R','N','D','C','Q','E','G','H','I','L','K','M','F','P','S','T','W','Y','V']

features = []
labels = []

for r in rows:
    try:
        labels.append(int(r[1]))
        n_seq = ''
        c_seq = ''
        if (len(r[2]) < 10):
            n_seq = r[2] + '1'*(10-len(r[2]))
            c_seq = '1'*(10-len(r[2])) + r[2][::-1]
        else:
            n_seq = r[2][:10]
            c_seq = r[2][len(r[2]) - 10:]
        cur_feature = []
        for acid in acids:
            cur_elem = []
            for c in n_seq:
                if (c == acid):
                    cur_elem.append(1)
                else:
                    cur_elem.append(0)
            cur_feature.append(cur_elem)
            cur_elem = []
            for c in c_seq:
                if (c == acid):
                    cur_elem.append(1)
                else:
                    cur_elem.append(0)
            cur_feature.append(cur_elem)
        features.append(list(np.array(cur_feature).ravel()))
    except:
        print("error")

features = np.array(features)
labels = np.array(labels)
print(features)


# In[4]:


# from sklearn.model_selection import GridSearchCV
# from sklearn import svm

# Cs = [0.001, 0.01, 0.1, 1, 10]
# gammas = [0.001, 0.01, 0.1, 1]
# param_grid = {'C': Cs, 'gamma' : gammas}
# grid_search = GridSearchCV(svm.SVC(kernel='rbf'), param_grid, cv=3)
# grid_search.fit(features, labels)
# print(grid_search.best_params_)


# In[10]:


from sklearn.svm import SVC # "Support Vector Classifier" 
clf = SVC(kernel='rbf')  
  
# fitting x samples and y classes 
clf.fit(features, labels)


# In[11]:


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
        n_seq = ''
        c_seq = ''
        if (len(r[1]) < 10):
            n_seq = r[1] + '1'*(10-len(r[1]))
            c_seq = '1'*(10-len(r[1])) + r[1][::-1]
        else:
            n_seq = r[1][:10]
            c_seq = r[1][len(r[1]) - 10:]
        cur_feature = []
        for acid in acids:
            cur_elem = []
            for c in n_seq:
                if (c == acid):
                    cur_elem.append(1)
                else:
                    cur_elem.append(0)
            cur_feature.append(cur_elem)
            cur_elem = []
            for c in c_seq:
                if (c == acid):
                    cur_elem.append(1)
                else:
                    cur_elem.append(0)
            cur_feature.append(cur_elem)
        features_test.append(list(np.array(cur_feature).ravel()))
    except:
        print("error")

features_test = np.array(features_test)
print(ids_test)
print(features_test)


# In[12]:


labels_test = clf.predict(features_test)

fields = ['ID', 'Label']
results = [list(a) for a in zip(ids_test, labels_test)]

with open('binary_predictions.csv', 'w') as csvfile: 
    # creating a csv writer object 
    csvwriter = csv.writer(csvfile) 
      
    # writing the fields 
    csvwriter.writerow(fields) 
      
    # writing the data rows 
    csvwriter.writerows(results)


# In[ ]:




