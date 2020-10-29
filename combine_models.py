#!/usr/bin/env python
# coding: utf-8

# In[5]:


import csv

fields = ['ID', 'Label']

amino = []
dipeptide = []
binary = []

results = []

file_amino = './amino_predictions.csv'
file_dipeptide = './dipeptide_predictions.csv'
file_binary = './binary_predictions.csv'

with open(file_amino, 'r') as csvfile:
    csvreader = csv.reader(csvfile)
    for row in csvreader: 
        amino.append(row)

amino = amino[1:]
print(len(amino))


with open(file_dipeptide, 'r') as csvfile:
    csvreader = csv.reader(csvfile)
    for row in csvreader: 
        dipeptide.append(row)

dipeptide = dipeptide[1:]
print(len(dipeptide))


with open(file_binary, 'r') as csvfile:
    csvreader = csv.reader(csvfile)
    for row in csvreader: 
        binary.append(row)

binary = binary[1:]
print(len(binary))


# In[6]:


size = len(binary)

for i in range(size):
    cur = str(amino[i][1]) + str(binary[i][1]) + str(dipeptide[i][1])
    
    cur_res = [amino[i][0]]
    if (cur.count('-1') > 1):
        cur_res.append(-1)
    else:
        cur_res.append(1)
    
    results.append(cur_res)
    
print(len(results))
with open('combine_predictions.csv', 'w') as csvfile: 
    # creating a csv writer object 
    csvwriter = csv.writer(csvfile) 
      
    # writing the fields 
    csvwriter.writerow(fields) 
      
    # writing the data rows 
    csvwriter.writerows(results)


# In[ ]:




