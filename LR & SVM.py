import pandas as pd
import numpy as np
import string
import os
from collections import Counter
import collections
import math
import sys

# making functions 

def zoe(prediction,testdata):
    error=0
    for i in range(0,len(testdata[:,1])):
        if prediction[i]!=testdata[i,1]:
            error=error+1
    return error/len(testdata[:,1]) 

def remove_punctuation(s):
    s = ''.join([i for i in s if i not in frozenset(string.punctuation)])
    return s
#
#itrain = sys.argv[1]
#itest = sys.argv[2]
#inputm=np.int(sys.argv[3])
#train=pd.read_csv(itrain,engine="python",sep="\t|\n",names=("reviewID","classLabel","reviewText"))
#test=pd.read_csv(itest,engine="python",sep="\t|\n",names=("reviewID","classLabel","reviewText"))

os.chdir("/Users/ankurwadhwa/Desktop/Spring17/CS 573-data mining/Homework/homework 2")

hw1=pd.read_csv("yelp_data.csv",engine="python",sep="\t|\n",names=("reviewID","classLabel","reviewText"))
np.random.seed(200)
train=hw1.sample(frac=0.1)
test=hw1.drop(train.index)
##test=test1.sample(n=200)
#

train['reviewText'] = train['reviewText'].astype(str)
train['reviewText'] = train['reviewText'].str.lower()
train['reviewText'] = train['reviewText'].apply(remove_punctuation)
train['reviewText'] = train['reviewText'].str.split()

test['reviewText'] = test['reviewText'].astype(str)
test['reviewText'] = test['reviewText'].str.lower()
test['reviewText'] = test['reviewText'].apply(remove_punctuation)
test['reviewText'] = test['reviewText'].str.split()

tt=np.array(train)
bagofwords = []
for row in tt[:,2] :
                unique = set(row)
                bagofwords.extend(unique)
bags = collections.Counter(bagofwords)     
bags = bags.most_common()

#number of words W 10 50 250 500 1000 4000
W=4000         
bags1 = np.array(bags[100:(100+W)])
featwords = bags1[:,0].tolist()

#Feature matrix
featmat=np.zeros((len(tt[:,1]),len(featwords)),dtype=np.int)
for i in range(0,len(tt[:,1])):
    for j in range(0,len(featwords)):
        if featwords[j] in tt[i,2]:
            featmat[i,j]=1




#Feature matrix for test set
testdata=np.array(test)
testfeatmat=np.zeros((len(testdata[:,1]),len(featwords)),dtype=np.int)

for i in range(0,len(testdata[:,1])):
    for j in range(0,len(featwords)):
        if featwords[j] in testdata[i,2]:
            testfeatmat[i,j]=1

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
                
#------------------------# Global conditions   #------------------------------#

tol= 0.000001

#------------------------# Logistic Regression  #-----------------------------# 
if inputm==1:    
    itr=0
    steplr=0.01
    lamblr=0.01
    x0=np.full((len(featmat[:,0]),1),1,dtype=int)
    newfeat=np.matrix(np.append(x0, featmat, axis=1))
    wlr=np.matrix(np.full((len(featmat[0,:]+1)+1,1),0,dtype=float))
    y=np.matrix(np.array(tt[:,1]))
    ypred=np.matrix(np.zeros((len(featmat[:,0]),1)))
    
    wnewlr=np.matrix(np.full((len(featmat[0,:])+1,1),0,dtype=float))
    
    while itr<100 or  abs(wnewlr-wlr).all() <= tol:
        ypred=1/(1+np.exp(-1*((np.matrix((newfeat)*wlr)).astype(float))))
        dif=(np.transpose((y-np.transpose(ypred))*newfeat))-lamblr*wlr
        wnewlr=wlr
        wlr=wlr+steplr*dif
        itr=itr+1
    
    xt0=np.full((len(testfeatmat[:,0]),1),1,dtype=int)    
    testfeat=np.append(xt0,testfeatmat, axis=1)
    prediction=np.zeros((len(testfeat[:,0]),1))
    
    prediction=1/(1+np.exp(-1*((np.matrix((testfeat)*wlr)).astype(float))))
    for i in range(0,len(test)): 
        if prediction[i]>=0.5:
               prediction[i]=1
        else:
               prediction[i]=0
    print("ZERO-ONE-LOSS-LR",zoe(prediction,testdata)) 


#------------------------# Support Vector Machine  #--------------------------# 

if inputm==2:
        
        itr=0
        step_svm=0.5
        lamda_svm=0.01
        x0=np.full((len(featmat[:,0]),1),1,dtype=int)
        newfeat_svm=np.append(x0, featmat, axis=1)
        
        w_svm=np.zeros([len(featmat[0,:])+1,1])
        w_svm_new=np.zeros([len(featmat[0,:])+1,1])        
        def svm10(data):
            for i in range(len(data)):
                if data[i,1]==0:
                    data[i,1]=-1
            
            return data
       
        train_svm=svm10(tt)
        test_svm=svm10(testdata)
        y_svm = np.zeros([np.size(train,axis=0),1])
        y_svm[:,0] = train_svm[:,1]

        while itr<100 or  abs(w_svm_new-w_svm).all() <= tol:
            ypred_svm =  np.array(np.matrix(newfeat_svm)*np.matrix(w_svm))
            ypred_svm[ypred_svm[:,0]>=0]=1
            ypred_svm[ypred_svm[:,0]<0]=-1
            for j in range(len(featwords)+1):
               ulta_delta = 0 
               delta=np.matmul(np.transpose(y_svm[y_svm[:,0]*ypred_svm[:,0] < 1,0]),newfeat_svm[y_svm[:,0]*ypred_svm[:,0] < 1,j])
               ulta_delta = np.size(tt,axis=0)*lamda_svm*w_svm[j,0]- delta
               w_svm_new[j,0] = w_svm[j,0]  
               w_svm[j,0] = w_svm[j,0]-(step_svm*ulta_delta/np.size(train,axis=0))
               
            itr=itr+1 
        xt0=np.full((len(testfeatmat[:,0]),1),1,dtype=int)
        testfeat=np.append(xt0,testfeatmat, axis=1)
        prediction_svm=np.array(np.transpose(np.transpose(np.matrix(w_svm))*np.transpose(np.matrix(testfeat))))
        for i in range(0,len(testdata)): 
            if prediction_svm[i]>=0:
                   prediction_svm[i]=1
            else:
                   prediction_svm[i]=-1
        mismatch=0
        for i in range(0,len(testdata)): 
            if prediction_svm[i]!=testdata[i,1]:
                   mismatch=mismatch+1
        zoe_svm=mismatch/(len(testdata[:,1]))
        print("ZERO-ONE-LOSS-SVM",zoe_svm)
        
        
#-------------------------------Analysis Codes -------------------------------#
#--------------------Without changing feature matrix--------------------------
#import pandas as pd
#import numpy as np
#import string
#import os
#from collections import Counter
#import collections
#import math
#import sys
#
##functions 
#def zoe(prediction,testdata):
#    error=0
#    for i in range(0,len(testdata[:,1])):
#        if prediction[i]!=testdata[i,1]:
#            error=error+1
#    return error/len(testdata[:,1]) 
#
#def remove_punctuation(s):
#    s = ''.join([i for i in s if i not in frozenset(string.punctuation)])
#    return s
#
##reading file from computer
#os.chdir("/Users/ankurwadhwa/Desktop/Spring17/CS 573-data mining/Homework/homework 2")
#
#hw=pd.read_csv("yelp_data.csv",engine="python",sep="\t|\n",names=("reviewID","classLabel","reviewText"))
#
#
#hw['reviewText'] = hw['reviewText'].astype(str)
#hw['reviewText'] = hw['reviewText'].str.lower()
#hw['reviewText'] = hw['reviewText'].apply(remove_punctuation)
#hw['reviewText'] = hw['reviewText'].str.split()
#
#
##making empty list which are used in program
#baseline=[]
#cross_nbc=[]
#cross_lor=[]
#cross_svm=[]
#
#avg_baseline=[]
#avg_nbc=[]
#avg_lor=[]
#avg_svm=[]
#
##making data set 
#S=np.array(hw)
#np.random.seed(101)
#np.random.shuffle(S)
#hwlen=len(hw)
#S1,S2,S3,S4,S5,S6,S7,S8,S9,S10=np.split(S,10)
#all=[S1,S2,S3,S4,S5,S6,S7,S8,S9,S10]
#D=[0.01,0.03,0.05,0.08,0.1,0.15]
#TSS=(np.array(D)*len(hw)).astype(int)
#
##for k in range(0,5):
#k=2
#print("datasetsize",TSS[k])
#for m in range(len(all)):
#    print(m)
#    test=all[m]
#    np.random.seed(101)
#    testdf=pd.DataFrame(test)
#    leftdf=hw.drop(testdf.index)
#    lefta=np.array(leftdf)
#    traindf=leftdf.sample(n=TSS[k])
#    train=np.array(traindf)
#    
#    #making feature matrix 
#    bagofwords = []
#    for row in train[:,2] :
#        unique = set(row)
#        bagofwords.extend(unique)
#    bags = collections.Counter(bagofwords)     
#    bags = bags.most_common()
#    
#    W=4000         
#    bags1 = np.array(bags[100:(100+W)])
#    featwords = bags1[:,0].tolist()
#    
#    #Feature matrix for train
#    featmat_train=np.zeros((len(train[:,1]),len(featwords)),dtype=np.int)
#    for a in range(0,len(train[:,1])):
#        for b in range(0,len(featwords)):
#            if featwords[b] in train[a,2]:
#                featmat_train[a,b]=1
#    
#    #Feature matrix for test
#    featmat_test=np.zeros((len(test[:,1]),len(featwords)),dtype=np.int)
#    for c in range(0,len(test[:,1])):
#        for d in range(0,len(featwords)):
#            if featwords[d] in test[c,2]:
#                featmat_test[c,d]=1
#    
#    #---------------------------NBC Model---------------------------------#
#    #count number and probability of XY 11 10 01 00
#    count_table=np.zeros((len(featmat_train[1,:]),4),dtype=np.int)
#    for i in range(0,len(featmat_train[1,:])):
#        for j in range(0,len(train[:,1])):
#                if featmat_train[j,i]==1 and train[j,1]==1:
#                    count_table[i,0]=count_table[i,0]+1
#                if featmat_train[j,i]==1 and train[j,1]==0:
#                    count_table[i,1]=count_table[i,1]+1
#                if featmat_train[j,i]==0 and train[j,1]==1:
#                    count_table[i,2]=count_table[i,2]+1
#                if featmat_train[j,i]==0 and train[j,1]==0:
#                    count_table[i,3]=count_table[i,3]+1
#    
#    prob_table=np.zeros((len(featmat_train[1,:]),4),dtype=np.float)
#    for i in range(0,len(featmat_train[1,:])):
#        for j in range(0,len(count_table[1,:])):
#            if j==0 :
#                prob_table[i,j]=(count_table[i,j]+1)/(sum(train[:,1])+2)
#            if j==1 :
#                prob_table[i,j]=(count_table[i,j]+1)/(len(train[:,1])-sum(train[:,1])+2)
#            if j==2 :
#                prob_table[i,j]=(count_table[i,j]+1)/(sum(train[:,1])+2)
#            if j==3 :
#                prob_table[i,j]=(count_table[i,j]+1)/(len(train[:,1])-sum(train[:,1])+2)
#           
#    
#    #Base Line 
#    Probclass_Y1=sum(train[:,1])/len(train[:,1])
#    Probclass_Y0=(len(train[:,1])-sum(train[:,1]))/len(train[:,1])
#    if Probclass_Y1>Probclass_Y0:
#        base=Probclass_Y1
#    else:
#        base=Probclass_Y0
#    baseline.append(base)
#    
#    testprobmatY1=np.array(featmat_test,dtype=float)
#    testprobmatY0=np.array(featmat_test,dtype=float)
#    
#    # predicting probability of P(Y=1|X)
#    for i in range(0,len(test[:,1])):
#        for j in range(0,len(featmat_test[1,:])):
#            if featmat_test[i,j]==0:
#                testprobmatY1[i,j]=(prob_table[j,2]) 
#            if featmat_test[i,j]==1:
#                testprobmatY1[i,j]=(prob_table[j,0])
#    
#    # predicting probability of P(Y=0|X)
#    for i in range(0,len(test[:,1])):
#        for j in range(0,len(featmat_test[1,:])):
#            if featmat_test[i,j]==0:
#                testprobmatY0[i,j]=(prob_table[j,3]) 
#            if featmat_test[i,j]==1:
#                testprobmatY0[i,j]=(prob_table[j,1])
#    #Calculation of arg max P(Y=1|X) or  P(Y=0|X)        
#    final_pred_probY1=(np.prod(testprobmatY1,axis=1))*(Probclass_Y1)
#    final_pred_probY0=(np.prod(testprobmatY0,axis=1))*(Probclass_Y0)
#    
#    prediction=np.zeros((len(test[:,1]),1),dtype=np.int)
#    for i in range(0,len(test[:,1])):
#        if final_pred_probY1[i]>final_pred_probY0[i]:
#            prediction[i]=1
#    Probtest_Y1=sum(test[:,1])/len(test[:,1])  
#    Probtest_Y0=(len(test[:,1])-sum(test[:,1]))/len(test[:,1]) 
#    #calculating zero one loss
#    error=0
#    for i in range(0,len(test[:,1])):
#        if prediction[i]!=test[i,1]:
#            error=error+1
#    zoenbc=error/len(test[:,1]) 
#    print("ZERO-ONE-LOSS-NBC",zoenbc)
#    cross_nbc.append(zoenbc)
#    
#    
#    
#    #--------------------------Logistic Regression------------------------#
#    tol=tol= 0.000001
#    itr=0
#    step_lr=0.01
#    lambda_lr=0.01
#    
#    x0=np.full((len(featmat_train[:,0]),1),1,dtype=int)
#    
#    newfeat=np.matrix(np.append(x0, featmat_train, axis=1))
#    
#    wlr=np.matrix(np.full((len(featmat_train[0,:]+1)+1,1),0,dtype=float))
#    
#    y=np.matrix(np.array(train[:,1]))
#    
#    ypred=np.matrix(np.zeros((len(featmat_train[:,0]),1)))
#    
#    wnewlr=np.matrix(np.full((len(featmat_train[0,:])+1,1),0,dtype=float))
#    
#    dif=np.matrix(np.zeros((len(featmat_train[:,0]),1)))
#    
#    while itr<100 or  np.sqrt(np.sum(np.square(wnewlr-wlr))) <= tol:
#        ypred=1/(1+np.exp(-1*((np.matrix((newfeat)*wlr)).astype(float))))
#        dif=(np.transpose((y-np.transpose(ypred))*newfeat))-lambda_lr*wlr
#        wnewlr=wlr
#        wlr=wlr+step_lr*dif
#        itr=itr+1
#    
#    xt0=np.full((len(featmat_test[:,0]),1),1,dtype=int)    
#    testfeat=np.append(xt0,featmat_test, axis=1)
#    
#    prediction_lr=np.zeros((len(testfeat[:,0]),1))
#    
#    prediction_lr=1/(1+np.exp(-1*((np.matrix((testfeat)*wlr)).astype(float))))
#    for i in range(0,len(test)): 
#        if prediction_lr[i]>=0.5:
#               prediction_lr[i]=1
#        else:
#               prediction_lr[i]=0
#    
#    print("ZERO-ONE-LOSS-LR",zoe(prediction_lr,test))
#    cross_lor.append(zoe(prediction_lr,test))
#    
#    #---------------------- Support Vector Machine -----------------------#
#    itr=0
#    step_svm=0.5
#    lamda_svm=0.01
#    newfeat_svm=np.append(x0, featmat_train, axis=1)
#    
#    w_svm=np.zeros([len(featmat_train[0,:])+1,1])
#    w_svm_new=np.zeros([len(featmat_train[0,:])+1,1])
#    
#    def svm10(data):
#        for i in range(len(data)):
#            if data[i,1]==0:
#                data[i,1]=-1
#        
#        return data
#    train_svm=svm10(train)
#    test_svm=svm10(test)
#    y_svm = np.zeros([np.size(train,axis=0),1])
#    y_svm[:,0] = train_svm[:,1]
#
#    while itr<100 or  abs(w_svm_new-w_svm).all() <= tol:
#        ypred_svm =  np.array(np.matrix(newfeat_svm)*np.matrix(w_svm))
##        ypred_svm[ypred_svm[:,0]>=0]=1
##        ypred_svm[ypred_svm[:,0]<0]=-1
#        for j in range(len(featwords)+1):
#           ulta_delta = 0
#           delta=0
#           delta=np.matmul(np.transpose(y_svm[y_svm[:,0]*ypred_svm[:,0] < 1,0]),newfeat_svm[y_svm[:,0]*ypred_svm[:,0] < 1,j])
#           ulta_delta = np.size(train,axis=0)*lamda_svm*w_svm[j,0]- delta
#           w_svm_new[j,0] = w_svm[j,0]  
#           w_svm[j,0] = w_svm[j,0]-(step_svm*ulta_delta/np.size(train,axis=0))
#        itr=itr+1 
#
#    prediction_svm=np.array(np.transpose(np.transpose(np.matrix(w_svm))*np.transpose(np.matrix(testfeat))))
#    for i in range(0,len(test)): 
#        if prediction_svm[i]>=0:
#               prediction_svm[i]=1
#        else:
#               prediction_svm[i]=-1
#    
#    zoe_svm=sum(abs(test_svm[:,1]-np.transpose(prediction_svm)))/(2*len(test_svm[:,1]))
#    print("ZERO-ONE-LOSS-SVM",zoe_svm)
#    
#
#    cross_svm.append(zoe_svm)
#    
#    
#    
#    #--------------Averaging result from cross validation-----------------#
#avg_baseline.append(np.mean(baseline))
#avg_nbc.append(np.mean(cross_nbc))
#avg_lor.append(np.mean(cross_lor))
#avg_svm.append(np.mean(cross_svm))
###    
#combine=np.array([baseline,cross_nbc,cross_lor,cross_svm]).astype(float)
#std=np.std(combine,axis=1)
#mn=np.mean(combine,axis=1)
#(mn)
#(std)



#------------------------changed feature matrix---------------
#import pandas as pd
#import numpy as np
#import string
#import os
#from collections import Counter
#import collections
#import math
#import sys
#
##functions 
#def zoe(prediction,testdata):
#    error=0
#    for i in range(0,len(testdata[:,1])):
#        if prediction[i]!=testdata[i,1]:
#            error=error+1
#    return error/len(testdata[:,1]) 
#
#def remove_punctuation(s):
#    s = ''.join([i for i in s if i not in frozenset(string.punctuation)])
#    return s
#
##reading file from computer
#os.chdir("/Users/ankurwadhwa/Desktop/Spring17/CS 573-data mining/Homework/homework 2")
#
#hw=pd.read_csv("yelp_data.csv",engine="python",sep="\t|\n",names=("reviewID","classLabel","reviewText"))
#
#
#hw['reviewText'] = hw['reviewText'].astype(str)
#hw['reviewText'] = hw['reviewText'].str.lower()
#hw['reviewText'] = hw['reviewText'].apply(remove_punctuation)
#hw['reviewText'] = hw['reviewText'].str.split()
#
#
##making empty list which are used in program
#baseline=[]
#cross_nbc=[]
#cross_lor=[]
#cross_svm=[]
#
#avg_baseline=[]
#avg_nbc=[]
#avg_lor=[]
#avg_svm=[]
#
##making data set 
#S=np.array(hw)
#np.random.seed(101)
#np.random.shuffle(S)
#hwlen=len(hw)
#S1,S2,S3,S4,S5,S6,S7,S8,S9,S10=np.split(S,10)
#all=[S1,S2,S3,S4,S5,S6,S7,S8,S9,S10]
#D=[0.01,0.03,0.05,0.08,0.1,0.15]
#TSS=(np.array(D)*len(hw)).astype(int)
#
#k=0
#print("datasetsize",TSS[k])
#for m in range(len(all)):
#    print(m)
#    test=all[m]
#    np.random.seed(101)
#    testdf=pd.DataFrame(test)
#    leftdf=hw.drop(testdf.index)
#    lefta=np.array(leftdf)
#    traindf=leftdf.sample(n=TSS[k])
#    train=np.array(traindf)
#    
#    #making feature matrix 
#    bagofwords = []
#    for row in train[:,2] :
#        unique = set(row)
#        bagofwords.extend(unique)
#    bags = collections.Counter(bagofwords)     
#    bags = bags.most_common()
#    
#    W=4000         
#    bags1 = np.array(bags[100:(100+W)])
#    featwords = bags1[:,0].tolist()
#    
#    #Feature matrix for train
#    featmat_train=np.zeros((len(train[:,1]),len(featwords)),dtype=np.int)
#    for a in range(0,len(train[:,1])):
#        for b in range(0,len(featwords)):
#            if featwords[b] in train[a,2]:
#                if train[a,2].count(featwords[b])>1:
#                   featmat_train[a,b]=2
#                else:
#                   featmat_train[a,b]=1
#
#    #Feature matrix for test
#    featmat_test=np.zeros((len(test[:,1]),len(featwords)),dtype=np.int)
#    for c in range(0,len(test[:,1])):
#        for d in range(0,len(featwords)):
#            if featwords[d] in test[c,2]:
#                if test[c,2].count(featwords[d])>1:
#                   featmat_test[c,d]=2
#                else:
#                   featmat_test[c,d]=1
#                   
#    
#    #---------------------------NBC Model---------------------------------#
#    #count number and probability of XY 21 20 11 10 01 00
#    count_table=np.zeros((len(featmat_train[1,:]),6),dtype=np.int)
#    for i in range(0,len(featmat_train[1,:])):
#        for j in range(0,len(train[:,1])):
#                if featmat_train[j,i]==2 and train[j,1]==1:
#                    count_table[i,0]=count_table[i,0]+1
#                if featmat_train[j,i]==2 and train[j,1]==0:
#                    count_table[i,1]=count_table[i,1]+1
#                if featmat_train[j,i]==1 and train[j,1]==1:
#                    count_table[i,2]=count_table[i,2]+1
#                if featmat_train[j,i]==1 and train[j,1]==0:
#                    count_table[i,3]=count_table[i,3]+1
#                if featmat_train[j,i]==0 and train[j,1]==1:
#                    count_table[i,4]=count_table[i,4]+1
#                if featmat_train[j,i]==0 and train[j,1]==0:
#                    count_table[i,5]=count_table[i,5]+1
#    
#    prob_table=np.zeros((len(featmat_train[1,:]),6),dtype=np.float)
#    for i in range(0,len(featmat_train[1,:])):
#        for j in range(0,len(count_table[1,:])):
#            if j==0 :
#                prob_table[i,j]=(count_table[i,j]+1)/(sum(train[:,1])+3)
#            if j==1 :
#                prob_table[i,j]=(count_table[i,j]+1)/(len(train[:,1])-sum(train[:,1])+3)
#            if j==2 :
#                prob_table[i,j]=(count_table[i,j]+1)/(sum(train[:,1])+3)
#            if j==3 :
#                prob_table[i,j]=(count_table[i,j]+1)/(len(train[:,1])-sum(train[:,1])+3)
#            if j==4 :
#                prob_table[i,j]=(count_table[i,j]+1)/(sum(train[:,1])+3)
#            if j==5 :
#                prob_table[i,j]=(count_table[i,j]+1)/(len(train[:,1])-sum(train[:,1])+3)
#    
#    #Base Line 
#    Probclass_Y1=sum(train[:,1])/len(train[:,1])
#    Probclass_Y0=(len(train[:,1])-sum(train[:,1]))/len(train[:,1])
#    if Probclass_Y1>Probclass_Y0:
#        base=Probclass_Y1
#    else:
#        base=Probclass_Y0
#    baseline.append(base)
#    
#    testprobmatY1=np.array(featmat_test,dtype=float)
#    testprobmatY0=np.array(featmat_test,dtype=float)
#    
#    # predicting probability of P(Y=1|X)
#    for i in range(0,len(test[:,1])):
#        for j in range(0,len(featmat_test[1,:])):
#            if featmat_test[i,j]==2:
#                testprobmatY1[i,j]=(prob_table[j,0]) 
#            if featmat_test[i,j]==1:
#                testprobmatY1[i,j]=(prob_table[j,2]) 
#            if featmat_test[i,j]==0:
#                testprobmatY1[i,j]=(prob_table[j,4])
#    
#    # predicting probability of P(Y=0|X)
#    for i in range(0,len(test[:,1])):
#        for j in range(0,len(featmat_test[1,:])):
#            if featmat_test[i,j]==2:
#                testprobmatY0[i,j]=(prob_table[j,1])
#            if featmat_test[i,j]==1:
#                testprobmatY0[i,j]=(prob_table[j,3]) 
#            if featmat_test[i,j]==0:
#                testprobmatY0[i,j]=(prob_table[j,5])
#    #Calculation of arg max P(Y=1|X) or  P(Y=0|X)        
#    final_pred_probY1=(np.prod(testprobmatY1,axis=1))*(Probclass_Y1)
#    final_pred_probY0=(np.prod(testprobmatY0,axis=1))*(Probclass_Y0)
#    
#    prediction=np.zeros((len(test[:,1]),1),dtype=np.int)
#    for i in range(0,len(test[:,1])):
#        if final_pred_probY1[i]>final_pred_probY0[i]:
#            prediction[i]=1
#    Probtest_Y1=sum(test[:,1])/len(test[:,1])  
#    Probtest_Y0=(len(test[:,1])-sum(test[:,1]))/len(test[:,1]) 
#    #calculating zero one loss
#    error=0
#    for i in range(0,len(test[:,1])):
#        if prediction[i]!=test[i,1]:
#            error=error+1
#    zoenbc=error/len(test[:,1]) 
#    print("ZERO-ONE-LOSS-NBC",zoenbc)
#    cross_nbc.append(zoenbc)
#    
#    
#    
#    #--------------------------Logistic Regression------------------------#
#    tol=tol= 0.000001
#    itr=0
#    step_lr=0.01
#    lambda_lr=0.01
#    
#    x0=np.full((len(featmat_train[:,0]),1),1,dtype=int)
#    
#    newfeat=np.matrix(np.append(x0, featmat_train, axis=1))
#    
#    wlr=np.matrix(np.full((len(featmat_train[0,:]+1)+1,1),0,dtype=float))
#    
#    y=np.matrix(np.array(train[:,1]))
#    
#    ypred=np.matrix(np.zeros((len(featmat_train[:,0]),1)))
#    
#    wnewlr=np.matrix(np.full((len(featmat_train[0,:])+1,1),0,dtype=float))
#    
#    dif=np.matrix(np.zeros((len(featmat_train[:,0]),1)))
#    
#    while itr<100 or  np.sqrt(np.sum(np.square(wnewlr-wlr))) <= tol:
#        ypred=1/(1+np.exp(-1*((np.matrix((newfeat)*wlr)).astype(float))))
#        dif=(np.transpose((y-np.transpose(ypred))*newfeat))-lambda_lr*wlr
#        wnewlr=wlr
#        wlr=wlr+step_lr*dif
#        itr=itr+1
#    
#    xt0=np.full((len(featmat_test[:,0]),1),1,dtype=int)    
#    testfeat=np.append(xt0,featmat_test, axis=1)
#    
#    prediction_lr=np.zeros((len(testfeat[:,0]),1))
#    
#    prediction_lr=1/(1+np.exp(-1*((np.matrix((testfeat)*wlr)).astype(float))))
#    for i in range(0,len(test)): 
#        if prediction_lr[i]>=0.5:
#               prediction_lr[i]=1
#        else:
#               prediction_lr[i]=0
#    
#    print("ZERO-ONE-LOSS-LR",zoe(prediction_lr,test))
#    cross_lor.append(zoe(prediction_lr,test))
#    
#    #---------------------- Support Vector Machine -----------------------#
#    itr=0
#    step_svm=0.5
#    lamda_svm=0.01
#    newfeat_svm=np.append(x0, featmat_train, axis=1)
#    
#    w_svm=np.zeros([len(featmat_train[0,:])+1,1])
#    w_svm_new=np.zeros([len(featmat_train[0,:])+1,1])
#    
#    def svm10(data):
#        for i in range(len(data)):
#            if data[i,1]==0:
#                data[i,1]=-1
#        
#        return data
#    train_svm=svm10(train)
#    test_svm=svm10(test)
#    y_svm = np.zeros([np.size(train,axis=0),1])
#    y_svm[:,0] = train_svm[:,1]
#
#    while itr<100 or  abs(w_svm_new-w_svm).all() <= tol:
#        ypred_svm =  np.array(np.matrix(newfeat_svm)*np.matrix(w_svm))
#        ypred_svm[ypred_svm[:,0]>=0]=1
#        ypred_svm[ypred_svm[:,0]<0]=-1
#        for j in range(len(featwords)+1):
#           ulta_delta = 0
#           delta=0
#           delta=np.matmul(np.transpose(y_svm[y_svm[:,0]*ypred_svm[:,0] < 1,0]),newfeat_svm[y_svm[:,0]*ypred_svm[:,0] < 1,j])
#           ulta_delta = np.size(train,axis=0)*lamda_svm*w_svm[j,0]- delta
#           w_svm_new[j,0] = w_svm[j,0]  
#           w_svm[j,0] = w_svm[j,0]-(step_svm*ulta_delta/np.size(train,axis=0))
#        itr=itr+1 
#
#    prediction_svm=np.array(np.transpose(np.transpose(np.matrix(w_svm))*np.transpose(np.matrix(testfeat))))
#    for i in range(0,len(test)): 
#        if prediction_svm[i]>=0:
#               prediction_svm[i]=1
#        else:
#               prediction_svm[i]=-1
#    
#    zoe_svm=sum(abs(test_svm[:,1]-np.transpose(prediction_svm)))/(2*len(test_svm[:,1]))
#    print("ZERO-ONE-LOSS-SVM",zoe_svm)
#    
#
#    cross_svm.append(zoe_svm)
#    
#    
#    
#    #--------------Averaging result from cross validation-----------------#
#avg_baseline.append(np.mean(baseline))
#avg_nbc.append(np.mean(cross_nbc))
#avg_lor.append(np.mean(cross_lor))
#avg_svm.append(np.mean(cross_svm))
###    
#combine=np.array([baseline,cross_nbc,cross_lor,cross_svm]).astype(float)
#std=np.std(combine,axis=1)
#mn=np.mean(combine,axis=1)
#(mn)
#(std)            
        
