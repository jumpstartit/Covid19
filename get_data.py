# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 00:45:23 2020

@author: tl074051
"""
import requests
import pandas as pd
from os import listdir
from os.path import isfile, join
import h2o
from h2o.automl import H2OAutoML
url_content =[]
for i in range(1,8):
    req = requests.get("http://api.covid19india.org/csv/latest/raw_data"+str(i)+".csv")
    url_content=req.content
    write_disk=open("csv/raw_data"+str(i)+".csv","wb")
    write_disk.write(url_content)
    write_disk.close()
    
extension = 'csv'
all_filenames = [f for f in listdir("csv/") if isfile(join("csv/", f))]
combine_csv=pd.concat([pd.read_csv("csv/"+f) for f in all_filenames])
clean=combine_csv.filter(['Index','Date Announced','Detected State','Num Cases'], axis=1)
clean['Num Cases']=clean["Num Cases"].fillna(0)
# clean['Date Announced'] = pd.to_datetime(clean['Date Announced'])
aggregation_functions = {'Num Cases': 'sum'}

new=clean.groupby(['Date Announced','Detected State']).aggregate(aggregation_functions).reset_index()


new.to_csv("combined_csv.csv")

data_f=new[['Date Announced','Detected State','Num Cases']].copy()
h2o.init()
data=h2o.H2OFrame(data_f)

parts = data.split_frame(ratios = [.8])
train = parts[0]
test = parts[1]
x=data.columns
print(x)
aml =  H2OAutoML(max_models=20, seed=1,max_runtime_secs = 120)
y="Num Cases"
aml.train(x,y, training_frame=train)

lb = aml.leaderboard
preds = aml.leader.predict(test)
