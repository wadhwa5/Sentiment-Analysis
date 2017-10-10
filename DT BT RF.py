import string
from collections import Counter
import numpy as np
import os
from random import randrange,shuffle,sample

os.chdir("/Users/ankurwadhwa/Desktop/Spring17/CS 573-data mining/Homework/homework 2")


def zo_error(prediction,actual):
    zero_error=0
    for i in range(len(test)):
        if prediction[i]!=test[i][0]:
            zero_error=zero_error+1
    return(zero_error/len(test))

def process_str(s):
    return s.translate(str.maketrans('','',string.punctuation)).lower().split()
    
    
def read_dataset(file_name):
    dataset = []
    with open(file_name) as f:
        for line in f:
            index, class_label, text = line.strip().split('\t')
            words = process_str(text)
            dataset.append( (int(class_label), set(words)) )
    return dataset
    
    
def get_most_commons(dataset, skip=100, total=100):
    my_list = []
    for item in dataset:
        my_list += list(item[1])

    counter = Counter(my_list)

    temp = counter.most_common(total+skip)[skip:]
    words = [item[0] for item in temp]
    return words

    
def generate_vectors(dataset, common_words):
    d = {}
    for i in range(len(common_words)):
        d[common_words[i]] = i
    
    vectors = []
    for item in dataset:
        vector = [0] * len(common_words)
        for word in item[1]:
            if word in d:
                vector[d[word]] = 1

        vectors.append( (item[0], vector) )

    return vectors
#______________________________________________________________________________________________________________________________vv_HERE_vv

def select_variable(traindata,p_value=None): 
#Output is a dictionary of "Index" and something else
    gini_list=list()                        
    #This is a list that saves gini_indexes for all 1000 variables
    if p_value==None:
        for i in range(len(traindata[0][1])):   
        #This iterates over each column (Variable)
            gini=0.0
            left=list()
            right=list()
            for j in range(len(traindata)):    
            #This iterates inside a variable (iterates one column)
                if traindata[j][1][i]==1:
                    right.append(traindata[j][0])
                else:
                    left.append(traindata[j][0])
            if len(left)==0:
                gini=1
            if len(right)==0:
                gini=1
            if len(left) and len(right)!=0:
                p=sum(left)/len(left)   #This is fraction of positives
                q=sum(right)/len(right) #This is fraction of positives
                gini=((p*(1-p))*(len(left)/len(traindata)))+((q*(1-q))*(len(right)/len(traindata)))
            gini_list.append(gini)          
        children=split(traindata,np.argmin(gini_list))  #Physical Split, not phantom
        return {"index":np.argmin(gini_list),"children":children}
    else:
        best_gini=10
        best_index=1000
        features=generate_feature_subset(traindata,int(p_value))
        for i in list(features):   
        #This iterates over each column (Variable)    <- This changed
            gini=0.0
            left=list()
            right=list()
            for j in range(len(traindata)):   
             #This iterates inside a variable (iterates one column)
                if traindata[j][1][i]==1:
                    right.append(traindata[j][0])
                else:
                    left.append(traindata[j][0])
            if len(left)==0:
                gini=1
            if len(right)==0:
                gini=1
            if len(left) and len(right)!=0:
                p=sum(left)/len(left)   #This is fraction of positives
                q=sum(right)/len(right) #This is fraction of positives
                gini=((p*(1-p))*(len(left)/len(traindata)))+((q*(1-q))*(len(right)/len(traindata)))
            if gini<best_gini:
                best_gini=gini
                best_index=i
        children=split(traindata,best_index)  #Physical Split, not phantom
        return {"index":best_index,"children":children}
     
def split(traindata,index):
    left=list()
    right=list()
    for i in range(len(traindata)):
        if traindata[i][1][index]==0:
            left.append(traindata[i])       #This has entire data rows
        else:
            right.append(traindata[i])      #This has entire data rows
    return left , right
        
def terminal(traindata):
    prediction=[traindata[y][0] for y in range(len(traindata))]
    positives=sum(prediction)
    negitives=len(prediction)-positives
    if positives<negitives:
        return (0)
    else:
        return (1)

def make_tree(dictionary,max_depth,min_split,current_depth,p_value=None):
    left,right=dictionary['children']
    del(dictionary['children'])
    if not left or not right:
        dictionary['left']=dictionary['right']=terminal(left+right)
        return
    if current_depth>=max_depth:
        dictionary['left'],dictionary['right']=terminal(left),terminal(right)
        return
    if len(left)<min_split:
        dictionary['left']=terminal(left)
    else:
        dictionary['left']=select_variable(left,p_value)
        make_tree(dictionary['left'],max_depth,min_split,current_depth+1)
    if len(right)<min_split:
        dictionary['right']=terminal(right)
    else:
        dictionary['right']=select_variable(right,p_value)
        make_tree(dictionary['right'],max_depth,min_split,current_depth+1)
        
#These are for DT mainly, and are also to be called for each of bagging and randomforest
def build_decisiontree(traindata,max_depth,min_split=10000000000000,p_value=None):
#We have to make a dictionary, not an actual prediction.
    structure=select_variable(traindata,p_value)
    make_tree(structure,max_depth,min_split,1,p_value)
    return(structure)

def predict(tree,testfeaturematrix):  
    if testfeaturematrix[1][tree["index"]]==0:
        if isinstance(tree["left"],dict):
            return predict(tree["left"],testfeaturematrix)
        else:
            return tree["left"]
    if testfeaturematrix[1][tree["index"]]==1:
        if isinstance(tree["right"],dict):
            return predict(tree["right"],testfeaturematrix)
        else:
            return tree["right"] 

#The following are mainly for BAGGING:
def bootstrap(data):
    bootstrap=list()
    while len(bootstrap)<int(len(data)):
        i=randrange(len(data))
        bootstrap.append(data[i])
    return(bootstrap)
    
def bagging_prediction(list_of_trees,row):
    prediction_set=[predict(tree,row) for tree in list_of_trees]
    if sum(prediction_set)<(len(prediction_set)/2):
        prediction=0
    else:
        prediction=1
    return(prediction)

def build_bagging(trainmatrix,max_depth,min_size,ntrees,testmatrix):
    trees=list()
    for n in range(ntrees):
        print(n)
        bootstrap_sample=bootstrap(trainmatrix)
        single_tree=build_decisiontree(bootstrap_sample,max_depth,min_size)
        trees.append(single_tree)
    
    prediction=[bagging_prediction(trees,row) for row in testmatrix]   
    return(prediction)    

#The following are mainly for the RandomForest:
def generate_feature_subset(dataset,p_value=None):
    if p_value==None:    
        subset=sample(range(len(dataset[0][1])),(len(dataset[0][1]))**0.5)
    else:
        subset=sample(range(len(dataset[0][1])),p_value)
    return(subset)

def build_randomForest(trainmatrix,max_depth,min_size,ntrees,testmatrix,p_value=None):
    trees=list()
    if p_value==None:
        p_value=int(len(trainmatrix[0][1])**0.5)
    for n in range(ntrees):
        bootstrap_sample=bootstrap(trainmatrix)
        single_tree=build_decisiontree(bootstrap_sample,max_depth,min_size,p_value)
        trees.append(single_tree)
    prediction=[bagging_prediction(trees,row) for row in testmatrix]               
     #We call bagging prediction, since just aggregating
    return prediction
    
    
def svm(features, labels):
    # test sub-gradient SVM
    total = features.shape[1]
    lam = 1.; D = total
    x = features; y = (labels-0.5)*2
    w = np.zeros(D); wpr = np.ones(D)
    eta = 0.5; lam = 0.01; i = 0; MAXI = 100; tol = 1e-6
    while True:
        if np.linalg.norm(w-wpr) < tol or i > MAXI:
            break
        f = w @ x.T    
        pL = np.where(np.multiply(y,f) < 1, -x.T @ np.diag(y), 0)
        pL = np.mean(pL,axis=1) + lam*w
        wpr = w
        w = w - eta*pL
        i += 1
    return w

def svm_pred(w, features):
    return np.where((features @ w) >= 0, 1, 0)    
    
def generate_vectors2(dataset, common_words):
    d = {}
    for i in range(len(common_words)):
        d[common_words[i]] = i

    vectors = []
    labels = []
    for item in dataset:
        vector = [0] * len(common_words)
        # Intercept term.
        vector.append(1)

        for word in item[1]:
            if word in d:
                vector[d[word]] = 1

        vectors.append(vector)
        labels.append(item[0])

    return np.array(vectors), np.array(labels)    
def calc_error(pred, labels):
    error = sum(np.where(pred != labels, 1, 0))
    return (error / labels.size)
#______________________________________________________________________________________________________________________________
train=read_dataset("yelp_data.csv")
test=read_dataset("yelp_data.csv")
#top_ten = get_most_commons(train, skip=100, total=10)             
 #Dont need to print top ten words in this assignment
common_words=get_most_commons(train,skip=100,total=1000)
train_featurematrix=generate_vectors(train,common_words)
test_featurematrix=generate_vectors(test,common_words)
test_y=list()
for i in range(len(test_featurematrix)):
    test_y.append(test_featurematrix[i][0])

#___________________Decision Tree -PART(A)-___________________
decisiontree=build_decisiontree(train_featurematrix,10,10)
DT_pred=list()
for i in range(len(test_featurematrix)):
    DT_pred.append(predict(decisiontree,test_featurematrix[i]))
DT_error=zo_error(DT_pred,test_featurematrix)
print("ZERO-ONE-LOSS-DT",DT_error)

#___________________Bagged Tree -PART(B)-_____________________
baggedtree=build_bagging(train_featurematrix,10,10,50,test_featurematrix)
BT_error=zo_error(baggedtree,test_featurematrix)
#print("ZERO-ONE-LOSS-BT",BT_error)
#___________________Random Forest -PART(C)-___________________
randomForest=build_randomForest(train_featurematrix,10,10,50,test_featurematrix)
RF_error=zo_error(randomForest,test_featurematrix)
#print("ZERO-ONE-LOSS-RF",RF_error)

#___________________SVM_______________________________________
train_f,train_l=generate_vectors2(train, common_words)
test_f, test_l = generate_vectors2(test, common_words)
w = svm(train_f, train_l)
test_pred = svm_pred(w, test_f)
SVM_error=calc_error(test_pred, test_l)
#print('ZERO-ONE-LOSS-SVM', SVM_error) 