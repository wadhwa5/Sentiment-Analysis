import pandas as pd
import numpy as np
import string
import os
from collections import Counter
import collections
import math
import sys
from  matplotlib import pyplot
import matplotlib.pyplot as plt
from scipy.spatial import distance
import seaborn
from scipy import linalg
import sklearn
from sklearn import metrics
seaborn.set(style='ticks')
import random
random.seed(10)

os.chdir("/Users/ankurwadhwa/Desktop/Spring17/CS 573-data mining/Homework/Homework 5")

#input1 = sys.argv[1]
#input2 = np.int(sys.argv[2])         # for number of cluster
#input3 = np.int(sys.argv[3])         #

#k=input2

k=10

#digit_emb=pd.read_csv(input1,engine="python",names=("mage id","classLabel","emb1","emb2"))
digit_emb=pd.read_csv("digits-embedding.csv",engine="python",names=("mage id","classLabel","emb1","emb2"))
digit_raw=pd.read_csv("digits-raw.csv",engine="python",header=None)


demb=np.array(digit_emb)
draw=np.array(digit_raw)

digits=demb[:,(2,3)]
imgpix=draw[:,2:]



#---------------------------Functions-------------------------------------------
def first_centroid(data,k):
    center=data.copy()
    np.random.shuffle(center)
    return center[:k]


def dista(point,cluster):
    d=np.zeros((len(point),len(cluster)))
    d=np.zeros((len(point),len(cluster)))
    for i in range(0,len(cluster)):
          d[:,i]=np.sqrt(((point - cluster[i,:])**2).sum(axis=1))
       
    return np.argmin(d, axis=1)

def distance_matrix(point):
    d=np.zeros((len(point),len(point)))
    for i in range(0,len(point)):
          d[:,i]=np.sqrt(((point - point[i,:])**2).sum(axis=1))
    return d

def new_centroid(data,label):
    
    a=np.unique(label)
    newcen=np.zeros((len(a),2))
    for i in a:
        newcen[i,:]=data[label==i].mean(axis=0)
    return newcen

def kmean(data,k):
    first=first_centroid(data,k)
    clust=first
    for i in range(0,50):    
       newlabel=dista(data,clust)
       newcent=new_centroid(data,newlabel)
       clust=newcent
    return clust
   
def SSD (data,cluster,label):
    a=np.unique(label)
    ssd=np.zeros((len(a),1))
    for i in a:
        p=data[label==i]
        ssd[i]=np.sum(((p - cluster[i,:])**2).sum(axis=1))
    return np.sum(ssd)        

def SCh(data,label):
    l=np.unique(label)
    SC=[]
    distofall=distance_matrix(data)
    for i in (0,len(data[:,0])-1):
        Amat=distofall[i,label==label[i]]
        A=np.sum(Amat)/((len(Amat))-1)
        B=[]
        for j in l:
           if label[i] != j:
               B.append(np.mean(distofall[i,label==j])) 
        B=min(B)
        SCi=(B-A)/max(B,A)
        SC.append(SCi)
    return np.mean(SC)

def ent(pb,pactual,ppred):
    if pb==0:
        return 0
    else:
        return pb*np.log(pb/(pactual*ppred))

#
#def NMI(label_actual,label_pred):
#    ac=np.unique(label_actual)
#    pc=np.unique(label_pred)
#    p=np.zeros((len(ac),len(pc)))
#    for i in ac:
#        for j in pc:
#            p[i,j]=np.sum((label_actual==i) & (label_pred==j))/len(label_actual)
#    sumac=np.sum(p,axis=0)
#    sumpc=np.sum(p,axis=1)
#    n=0
#    aclog=0
#    pclog=0
#    for i in ac:
#        aclog=aclog+sumac[i]*np.log(sumac[i])
#        pclog=pclog+sumpc[i]*np.log(sumpc[i])
#    for j in pc:
#            n=n+ent(p[i,j],sumac[i],sumpc[j])
#    return (n/((-0.5)*(aclog+pclog)))

def NMI(label_actual,label_pred):
    ac=np.unique(label_actual)
    pc=np.unique(label_pred)
    p=np.zeros((len(ac),len(pc)))
    for i in range(len(ac)):
        for j in range(len(pc)):
            p[i,j]=np.sum((label_actual==ac[i]) & (label_pred==pc[j]))/len(label_actual)
    sumac=np.sum(p,axis=0)
    sumpc=np.sum(p,axis=1)
    n=0
    aclog=0
    pclog=0
    for i in range(len(pc)):
        aclog=aclog+sumac[i]*np.log(sumac[i])
    for i in range(len(ac)):
        pclog=pclog+sumpc[i]*np.log(sumpc[i])
    for i in range(len(ac)):
        for j in range(len(pc)):
                n=n+ent(p[i,j],sumac[j],sumpc[i])
    return (n/((-0.5)*(aclog+pclog)))



def hw5(datafull,k):
    label_actual=datafull[:,1]
    data=datafull[:,(2,3)]
    cluster=kmean(data,k)
    label_pred=dista(data,cluster)    
    ssd=SSD(data,cluster,label_pred)
    sc=SCh(data,label_pred)
    nmi=NMI(label_actual,label_pred)
    print("WC-SSD",ssd) 
    print("SC",sc)
    print("NMI",nmi)
    print("SC",metrics.silhouette_score(data,label_pred))
    print("NMI",metrics.normalized_mutual_info_score(label_actual, label_pred))
    
def ap(datafull,k):
    label_actual=datafull[:,1]
    data=datafull[:,(2,3)]
    cluster=kmean(data,k)
    label_pred=dista(data,cluster)    
    ssd=SSD(data,cluster,label_pred)
    sc=SCh(data,label_pred)
    nmi=NMI(label_actual,label_pred)
    print("WC-SSD",ssd) 
    print("SC",sc)
    print("NMI",nmi)
    sc1=metrics.silhouette_score(data,label_pred)
    nm1=metrics.normalized_mutual_info_score(label_actual, label_pred)
    print("SC",sc1)
    print("NMI",nm1)
    return ssd,sc1,nm1




        
#--------------------------Main Program ---------------------------------------

#first=first_centroid(demb[:,(2,3)],k)
#dist=np.zeros((len(demb[:,1]),k))
#
#fd1=dista(demb[:,(2,3)],first)
#
#try3=new_centroid(demb[:,(2,3)],fd1)
#
#try4=distance_matrix(demb[:,(2,3)])
#
#clust=first
#
#for i in range(0,50):    
#    newlabel=dista(demb[:,(2,3)],clust)
#    newcent=new_centroid(demb[:,(2,3)],newlabel)
#    clust=newcent
#    print(clust)




#------------------ Trial Plots------------------------------------------
#newclus1=np.column_stack((digits,lab))
#nc1=pd.DataFrame(data=newclus1, columns=("x1","x2","classLabel"))
#nc21=nc1.sample(n=1000)
#nc22 = seaborn.FacetGrid(data=nc21, hue='classLabel')
#nc22.map(plt.scatter, 'x1', 'x2').add_legend()

#------------------------------------Analysis --------------------------------
ball=digit_emb.copy()

b2=ball[ball.classLabel == 2]
b4=ball[ball.classLabel == 4]
b6=ball[ball.classLabel == 6]
b7=ball[ball.classLabel == 7]
f2=[b6,b7]
f1 = [b2, b4, b6,b7]



b2467 = np.array(pd.concat(f1))
b67 = np.array(pd.concat(f2))
ball=np.array(digit_emb.copy())


SSD_ball=[]
SSD_b2467=[]
SSD_b67=[]

SC_ball=[]
SC_b2467=[]
SC_b67=[]

NMI_ball=[]
NMI_b2467=[]
NMI_b67=[]


  
for k in (2,4,8,16,32):
    SSD_ball1,SC_ball1,NMI_ball1=ap(ball,k)
    SSD_ball.append(SSD_ball1)
    SC_ball.append(SC_ball1)
    NMI_ball.append(NMI_ball1)
    SSD_b24671,SC_b24671,NMI_b24671=ap(b2467,k)
    SSD_b2467.append(SSD_b24671)
    SC_b2467.append(SC_b24671)
    NMI_b2467.append(NMI_b24671)            
    SSD_b671,SC_b671,NMI_b671=ap(b67,k)
    SSD_b67.append(SSD_b671)
    SC_b67.append(SC_b671)
    NMI_b67.append(NMI_b671) 

kvalue=[2,4,8,16,32]
#---------_ALL data set-------------
plt.plot(kvalue,SSD_ball,)
plt.ylabel("SSD")
plt.xlabel("Number of Clusters")
plt.legend()
plt.title("SSD v/s Number of Clusters")
plt.savefig("Cpart_SSD.png")


plt.plot(kvalue,SC_ball,label="ALL Data")
plt.xlabel("Number of Cluster")
plt.ylabel("SC")
plt.legend()
plt.title("k vs SC for all data")
plt.savefig("Bpart_all_SC.png")


plt.plot(kvalue,NMI_ball,label="ALL Data")
plt.xlabel("Number of Cluster")
plt.ylabel("NMI")
plt.legend()
plt.title("k vs NMI for all data")
plt.savefig("Bpart_all_NMI.png")


#-----------b2467-----------
plt.plot(kvalue,SSD_b2467,label="B2467")
plt.ylabel("SSD")
plt.xlabel("Number of Cluster")
plt.legend()
plt.title("k vs SSD for 2,4,6,& 7 data")
plt.savefig("Bpart_B2467_SSD.png")


plt.plot(kvalue,SC_b2467,label="B2467")
plt.xlabel("Number of Cluster")
plt.ylabel("SC")
plt.legend()
plt.title("k vs SC for 2,4,6,& 7 data")
plt.savefig("Bpart_B2467_SC.png")


plt.plot(kvalue,NMI_b2467,label="B2467")
plt.xlabel("Number of Cluster")
plt.ylabel("NMI")
plt.legend()
plt.title("k vs NMI for 2,4,6,& 7 data")
plt.savefig("Bpart_B2467_NMI.png")

#-------------b67-------------


plt.plot(kvalue,SSD_b67,label="B67")
plt.ylabel("SSD")
plt.xlabel("Number of Cluster")
plt.legend()
plt.title("k vs SSD for 6,& 7 data")
plt.savefig("Bpart_B67_SSD.png")


plt.plot(kvalue,SC_b67,label="B67")
plt.xlabel("Number of Cluster")
plt.ylabel("SC")
plt.legend()
plt.title("k vs SC for 6,& 7 data")
plt.savefig("Bpart_B67_SC.png")


plt.plot(kvalue,NMI_b67,label="B67")
plt.xlabel("Number of Cluster")
plt.ylabel("NMI")
plt.legend()
plt.title("k vs NMI for 6,& 7 data")
plt.savefig("Bpart_B67_NMI.png")




#------------------------------PCA---------------------------------------------

def PCA(data,rescale=10):
    data=data-mean(data,axis=0)
    covm=np.cov(data,rowvar=False)
    eigenvalue,eigenvector=linalg.eigh(covm)
    index=np.argsort(eigenvalue)[::-1]
    eigenvector=eigenvector[:,index]
    eigenvalue=eigenvalue[index]
    eigenvector = eigenvector[:, :rescale]
    impvar=np.dot(eigenvector.T, data.T).T
    return impvar,eigenvector,eigenvalue
                 
#----------------- Bonus Q1---------------------    
bonus,eivec,eival=PCA(imgpix,10)   


#-----------------Bonus Q2 ----------------------

for i in range(0,10):
    pca_x=np.reshape(eivec[:,i], (28, 28))
    plt.matshow(pca_x,cmap='gray')
    plt.savefig("PCA_image" + str(i) + ".png")
##
###-----------------Bonus Q3 ----------------------
##
b1=np.column_stack((draw[:,0],draw[:,1],bonus[:,0],bonus[:,1]))
columns=("image id","classLabel","pca1","pca2")
b1data=pd.DataFrame(data=b1, columns=columns)
b1d=b1data.sample(n=1000)
b3p = seaborn.FacetGrid(data=b1d, hue='classLabel')
b3p.map(plt.scatter, 'pca1', 'pca2').add_legend()
plt.savefig("Bonus" + str(2) + ".png")

#-----------------Bonus Q4 ----------------------




