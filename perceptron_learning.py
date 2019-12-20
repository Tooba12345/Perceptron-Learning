import numpy as np
import pandas as pd
from pandas import DataFrame

nfl_data = pd.read_csv("dataset2_pre_Processed.csv")
nfl_data.head(100)


##for pre-Processing

rows,columns=nfl_data.shape
for i in range(rows):
    if (nfl_data.loc[i,"Actual"]=="R"):
        nfl_data.loc[i,"Actual"]=1
    else:
        nfl_data.loc[i,"Actual"]=0
nfl_data.tail(100)


mydataset=[]
for i in range(0,len(nfl_data)):
    mylist1=nfl_data.iloc[i].tolist()
    mydataset.append(mylist1)
    
#print(mydataset)
alpha = 0.9
epoch = 800
Predicted_List=[]
def Predict_function(row,weights):
    threshold=weights[0]
    for i in range(len(row)-1):
        
        threshold += weights[i + 1] * row[i]
        #print(Activation)
    if threshold >=1:
        return 1
    else:
        return 0
    print(Predicted_List)
def Updated_weights(mydataset,alpha,epoch):
   # print("what is length",len(mydataset[0]))
    weights=[0.0 for i in range(len(mydataset[0]))]
    #print("weights in loop",weights)
    for epoch in range(epoch):
        error_sum=0
        for row in mydataset:
            prediction=Predict_function(row,weights)
            error=row[-1]-prediction
            #error_percentage=100*error/row[-1]
            #print("Percentage of an error after one iteration",error_percentage)
            error_sum =error_sum + error**2
            #print(error_sum)
            weights[0] = weights[0] + alpha * error
            for i in range(len(row)-1):
                weights[i + 1] = weights[i + 1] + alpha * error * row[i]
        print('epoch=%d, alpha=%.3f, error=%.3f,  weight_0=%.3f,  weight_1=%.3f,  weight_2=%.3f' % (epoch, alpha, error_sum,weights[0],weights[1],weights[2]))
    return weights
weights = Updated_weights(mydataset, alpha, epoch)
print("The final weights are",weights)
actual_List=[]
for row in mydataset:
    ac=row[-1]
    actual_List.append(ac)
weights=[0.0 for i in range(0,3)]
