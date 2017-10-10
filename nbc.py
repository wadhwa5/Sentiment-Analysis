import pandas as pd
import numpy as np
import os
import sys
from collections import Counter
os.chdir("/Users/ankurwadhwa/Desktop/Spring17/CS 573-data mining/Homework/homework 2")

hw1=pd.read_csv("yelp_data.csv",engine="python",sep="\t|\n",names=("reviewID","classLabel","reviewText"))

def cleanreview(reviewtxt):
    #upper to lower
    lowercase = reviewtxt.astype(str).str.lower()
    #Strip all punctuation
    nopun = lowercase.astype(str).str.replace('.!:?-_','')
    nopun = nopun.astype(str).str.replace('[^\w\s]','')
    nopun = nopun.astype(str).str.replace('  ','')
    #splitting
    sp=nopun.astype(str).str.split()
    return(sp)

hw1["reviewText"]=cleanreview(hw1["reviewText"])
#raw_input=("Hello enter your file")
##user input of train and test data
#itrain = sys.argv[1]
#itest = sys.argv[2]
#uctrain=pd.read_csv(itrain,engine="python",sep="\t|\n",names=("reviewID","classLabel","reviewText"))
#uctest=pd.read_csv(itest,engine="python",sep="\t|\n",names=("reviewID","classLabel","reviewText"))

#cleaning process

#uctrain["reviewText"]=cleanreview(uctrain["reviewText"])
#uctest["reviewText"]=cleanreview(uctest["reviewText"])

#train=uctrain
#test=uctest

#splitting data into train-test set
train=hw1.sample(frac=0.90)
test=hw1.drop(train.index)



bag=list()
s=list(train["reviewText"])
for i in range(1,len(s)):
    s[0]=np.concatenate((s[0],s[i]),axis=0)
s=(s[0])
bag=list(s)
allcount=Counter(bag)
top=Counter(bag).most_common(600)

#Feature words 
featwords=np.array(top[100:600])
for i in range(0,10):
    print ('Word ',i+1,featwords[i,0])

#Feature matrix
tt=np.array(train)
featmat=np.zeros((len(tt[:,1]),len(featwords[:,1])),dtype=np.int)
for i in range(0,len(tt[:,1])):
    for j in range(0,len(featwords[:,1])):
        if featwords[j,0] in tt[i,2]:
            featmat[i,j]=1

#count number and probability of XY 11 10 01 00
count_table=np.zeros((len(featmat[1,:]),4),dtype=np.int)
for i in range(0,len(featmat[1,:])):
    for j in range(0,len(tt[:,1])):
            if featmat[j,i]==1 and tt[j,1]==1:
                count_table[i,0]=count_table[i,0]+1
            if featmat[j,i]==1 and tt[j,1]==0:
                count_table[i,1]=count_table[i,1]+1
            if featmat[j,i]==0 and tt[j,1]==1:
                count_table[i,2]=count_table[i,2]+1
            if featmat[j,i]==0 and tt[j,1]==0:
                count_table[i,3]=count_table[i,3]+1
                

prob_table=np.zeros((len(featmat[1,:]),4),dtype=np.float)
for i in range(0,len(featmat[1,:])):
    for j in range(0,len(count_table[1,:])):
        if j==0 :
            prob_table[i,j]=count_table[i,j]/(sum(tt[:,1]))
        if j==1 :
            prob_table[i,j]=count_table[i,j]/(len(tt[:,1])-sum(tt[:,1]))
        if j==2 :
            prob_table[i,j]=count_table[i,j]/(sum(tt[:,1]))
        if j==3 :
            prob_table[i,j]=count_table[i,j]/(len(tt[:,1])-sum(tt[:,1]))
        
        
        
Probclass_Y1=sum(tt[:,1])/len(tt[:,1])
Probclass_Y0=(len(tt[:,1])-sum(tt[:,1]))/len(tt[:,1])



#laplace smoothing function
for i in range(0,len(featmat[1,:])):
    for j in range(0,len(count_table[1,:])):
        if prob_table[i,j]==0:
            if j==0 :
               prob_table[i,j]=(count_table[i,j]+1)/(sum(tt[:,1])+len(featwords[:,1]))
            if j==1 :
               prob_table[i,j]=(count_table[i,j]+1)/(len(tt[:,1])-sum(tt[:,1])+len(featwords[:,1]))
            if j==2 :
               prob_table[i,j]=(count_table[i,j]+1)/(sum(tt[:,1])+len(featwords[:,1]))
            if j==3 :
               prob_table[i,j]=(count_table[i,j]+1)/(len(tt[:,1])-sum(tt[:,1])+len(featwords[:,1]))

#marginal prob
M_prob=(prob_table[:,0]+prob_table[:,1])  

#---------------------------------#----------------------------------#---------------------------#
#testing phase
testdata=np.array(test)
testfeatmat=np.zeros((len(testdata[:,1]),len(featwords[:,1])),dtype=np.int)
#testing feature matrix
for i in range(0,len(testdata[:,1])):
    for j in range(0,len(featwords[:,1])):
        if featwords[j,0] in testdata[i,2]:
            testfeatmat[i,j]=1
                       
testprobmatY1=np.array(testfeatmat,dtype=float)
testprobmatY0=np.array(testfeatmat,dtype=float)

# predicting probability of P(Y=1|X)
for i in range(0,len(testdata[:,1])):
    for j in range(0,len(testfeatmat[1,:])):
        if testfeatmat[i,j]==0:
            testprobmatY1[i,j]=(prob_table[j,2]) 
        if testfeatmat[i,j]==1:
            testprobmatY1[i,j]=(prob_table[j,0])

# predicting probability of P(Y=0|X)
for i in range(0,len(testdata[:,1])):
    for j in range(0,len(testfeatmat[1,:])):
        if testfeatmat[i,j]==0:
            testprobmatY0[i,j]=(prob_table[j,3]) 
        if testfeatmat[i,j]==1:
            testprobmatY0[i,j]=(prob_table[j,1])

#Calculation of arg max P(Y=1|X) or  P(Y=0|X)        
final_pred_probY1=(np.prod(testprobmatY1,axis=1))*(Probclass_Y1)
final_pred_probY0=(np.prod(testprobmatY0,axis=1))*(Probclass_Y0)

prediction=np.zeros((len(testdata[:,1]),1),dtype=np.int)
for i in range(0,len(testdata[:,1])):
    if final_pred_probY1[i]>final_pred_probY0[i]:
        prediction[i]=1
  
#calculating zero one loss
error=0
for i in range(0,len(testdata[:,1])):
    if prediction[i]!=testdata[i,1]:
        error=error+1
zoe=error/len(testdata[:,1]) 
print("ZERO-ONE-LOSS",zoe) 
     