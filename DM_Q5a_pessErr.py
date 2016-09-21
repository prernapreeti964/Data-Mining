from array import *
import math
import operator
import sys
import time
import csv
import numpy as np
from scipy.spatial.distance import * 
import csv
import numpy as np
from scipy.spatial.distance import * 
from sklearn.metrics import f1_score


inputFile=sys.argv[1]
delimiter=sys.argv[2]

class Node:
    def __init__(self):
        spltPnt=0       
        left=None
        right=None
        nodeAttbr=''
        classLabel=''
       
       
       
def maximumClass(dataset,noOfCols):
    classColumn=[row[noOfCols-1] for row in dataset]
    maxCnt=0
    for i in np.unique(classColumn):
        cnt=classColumn.count(i)
        if(maxCnt<cnt):
            maxCnt=cnt
            classmax=i
    return maxCnt

def calcErrPessimistic(subset1,subset2):
    leftTreeCnt=maximumClass(subset1,len(subset1[0]))
    rightTreeCnt=maximumClass(subset2,len(subset2[0]))
    misClassLeft=len(subset1)-leftTreeCnt
    misClassRight=len(subset2)-rightTreeCnt
    errPess=(misClassLeft+misClassRight+(1*0.5))/(len(subset1)+len(subset2))
    return errPess
 
        
def testTrainPartitionData(dataset):
    np.random.shuffle(dataset)
    numFolds=10
    foldSize= len(dataset)/numFolds
    partitions=[]
    train=[]
    test=[]
    for i in range(0,numFolds):
        partitions.append(dataset[i*foldSize:(i+1)*(foldSize)])
    
    for i in range(0, numFolds):
        test.append(partitions[i])

    for i in range(0, numFolds):
        train.append(np.empty((len(dataset)-foldSize, len(dataset[0])), dtype=object))

    for i in range(0,numFolds):
        l=0
        for j in range(0, numFolds):
            for k in range(0,foldSize):
                if j!=i:            
                    train[i][l]=partitions[j][k]
                    l+=1
    return (train,test)
       
 
def createNode():
    return Node()
       
def stoppingCondition(dataSet,noOfCols):
    unique_val=[]
    for i in range(0,noOfCols-1):
        col=[row[i] for row in dataSet]
        unique_val.append(len(np.unique(col)))
    maxValue = max(unique_val)
    classColumn=[row[noOfCols-1] for row in dataSet]
    if(len(np.unique(classColumn))==1 or maxValue == 1 ):
        return True
    return False

def classifyLabel(dataset,noOfCols):
    classColumn=[row[noOfCols-1] for row in dataset]
    max=0
    for i in np.unique(classColumn):
        cnt=classColumn.count(i)
        if(max<cnt):
            max=cnt
            maxClass=i
    return maxClass

def splitDataset(dataset,testCondVal,colNum):
    dataset=np.array([np.array(x) for x in dataset])
    subset1=[]
    subset2=[]
    for i in range(0,len(dataset)):
        if(float(dataset[i,colNum]) <= testCondVal):
            subset1.append(dataset[i])
        else:
            subset2.append(dataset[i])
    return (subset1,subset2)

def createConfusionMat(testPartition,predClassCol):
    columnNumber=len(testPartition[0])
    classColumn=[rows[columnNumber-1] for rows in testPartition]
    uniqueClasses=np.unique(predClassCol)
    uniqueClasses_cnt=len(np.unique(predClassCol))
    confusionMat=np.zeros(shape=(uniqueClasses_cnt,uniqueClasses_cnt))
    index=0
    confCol=0
    for i in range(0,columnNumber):
        for j in range(0,uniqueClasses_cnt):
            if predClassCol[i]==uniqueClasses[j]:
                index=j
            if testPartition[i]==uniqueClasses[j]:
                confCol=j
        confusionMat[confCol][index]=confusionMat[confCol][index]+1
    balAccSum=0
    AccSum=0
    tot_sum=0
    for i in range(0,uniqueClasses_cnt):
        row_tot=0
        for j in range(0,uniqueClasses_cnt):
            row_tot=row_tot+confusionMat[i][j]
            if i==j:
                AccSum=AccSum+confusionMat[i][j]
                c=confusionMat[i][j]
            tot_sum=tot_sum+confusionMat[i][j]
        if(c==0 and row_tot==0):
            balAccSum=balAccSum
        else:
            balAccSum=balAccSum+(float(c)/float(row_tot))
    balAcc=float(balAccSum)/float(uniqueClasses_cnt)
    accuracy=float(AccSum)/float(tot_sum)
    fscore=f1_score(classColumn, predClassCol)
    
    return (accuracy,balAcc,fscore)




def TreeGrowthRecursive(dataset,noOfCols,errParent):
    if(stoppingCondition(dataset,noOfCols) == True or len(dataset) == 1):
        leaf=createNode()
        leaf.classLabel=classifyLabel(dataset,noOfCols)
        leaf.left=None
        leaf.right=None
        return leaf
    else:
        root=createNode()
        bestAttbr=calcBestSpltPnt(dataset)
        root.nodeAttbr=bestAttbr[0]
        root.spltPnt=bestAttbr[1]
        splits=splitDataset(dataset,bestAttbr[1],int(root.nodeAttbr))
        error=calcErrPessimistic(splits[0],splits[1])
        print "parent",errParent
        print "child",error
        if(error<=errParent):
            root.left=TreeGrowthRecursive(splits[0],noOfCols,error)
            root.right=TreeGrowthRecursive(splits[1],noOfCols,error)
        else:
            root.classLabel=classifyLabel(dataset,noOfCols)
            root.left=None
            root.right=None
  
    return root


def labelPredictClass(root,setOfAttbr):
    if root.left==None and root.right==None:
        return root.classLabel
    if float(setOfAttbr[root.nodeAttbr])<=root.spltPnt:
        class_label=labelPredictClass(root.left,setOfAttbr)
    else:
        class_label=labelPredictClass(root.right,setOfAttbr)
    return class_label

def predict(root,dataset):
    rows=len(dataset)
    dataset=np.array(dataset)
    mat=[]
    for i in range(0,rows):
        mat.append(labelPredictClass(root,dataset[i]))
    return mat

def calculateAcc(subdata,mat):
    colNum=len(subdata[0])
    column=[rows[colNum-1] for rows in subdata]
    cnt=0
    for i in range(0,len(subdata)):
        if str(mat[i])==str(column[i]):
            cnt+=1
    
    accuracy=float(cnt)/len(subdata)
    return accuracy



def calcGini(data, searchAttbr, classColumn):
    uniqueClassValues= np.unique(classColumn)
    splitList=[]
    splitList.append(float(searchAttbr[0])-0.005)
    comIndex=[]
    for i in range(0, len(searchAttbr)-1):
        weightedAverage=[]
        avg=(float(searchAttbr[i])+float(searchAttbr[i+1]))/2
        if(avg == float(searchAttbr[i])):
            comIndex.append(i+1)    
        splitList.append(avg)
        #print "same",comIndex
    splitList.append(float(searchAttbr[-1])+0.005)
    for i in range(0, len(splitList)):
        if(i in comIndex):
            weightedAverage.append(1)
        else:
            giniArray=  np.zeros((len(uniqueClassValues),2), dtype=float)
            ginsigLess=0.0
            ginGrtrSum=0.0
            ginlessrSum=0.0
            ginsigGrt=0.0
            for k in range(0, len(uniqueClassValues)):
                countGreater=0
                countLesser=0       
                for j2 in range(i, len(classColumn)):
                    if classColumn[j2]==uniqueClassValues[k]:
                        countGreater+=1  
                for j in range(0, i):
                    if classColumn[j]==uniqueClassValues[k]:
                        countLesser+=1          
                giniArray[k,0]=countLesser
                giniArray[k,1]=countGreater
                ginGrtrSum+=giniArray[k,1]
                ginlessrSum+=giniArray[k,0]
        
            for k in range(0, len(uniqueClassValues)):
                if ginlessrSum!=0:
                    ginsigLess+=((giniArray[k,0])/ginlessrSum)**2
                else: 
                    ginsigLess=1.0
                giniLesser=1-ginsigLess
                if ginGrtrSum!=0:
                    ginsigGrt+=((giniArray[k,1])/ginGrtrSum)**2
                else: 
                    ginsigGrt=1.0
                giniGreater=1-ginsigGrt
            weightedAverage.append((ginlessrSum*giniLesser/len(classColumn))+(ginGrtrSum*giniGreater/len(classColumn)))
    
    
    giniIndex=weightedAverage.index(min(weightedAverage))
                
    ginSpltList=[]
    ginSpltList.append(min(weightedAverage))
    ginSpltList.append(splitList[giniIndex])
    return  ginSpltList

def calcBestSpltPnt(dataset):
    
    numCol = len(dataset[0])
    min=1
    for i in range(0,numCol-1):        
        sortedDataset= sorted(dataset,key=lambda x: x[i])
        sortedDataset=np.array(sortedDataset)
        giniSplitDetailslist=calcGini(dataset, sortedDataset[:,i], sortedDataset[:,-1])
        
        if min>= giniSplitDetailslist[0]:
            min=giniSplitDetailslist[0]
            spltpnt=giniSplitDetailslist[1]
            minimum_index=i
            if min==0:
                break
    return (minimum_index,spltpnt)

def showData(node):
    if(node.left == None and node.right == None):
        print "class label ",node.classLabel
    else:
        showData(node.left)
        print "attribute ",node.nodeAttbr,"\tsplit_point ",node.spltPnt
        showData(node.right)

with open(inputFile) as file:
    reader = csv.reader(file,delimiter=sys.argv[2])
    readFile=list(reader)

training_test=testTrainPartitionData(readFile)
num_cols=len(readFile[0])
BalAcc=0.0
Accuracy=0.0
Fscore=0.0
for i in range(0,10):
    print "Training: ",i
    root=TreeGrowthRecursive(training_test[0][i],num_cols,1)
    showData(root)
    print "\n"
    predClassCol=predict(root,training_test[1][i])
    perf_eval=createConfusionMat(training_test[1][i],predClassCol)
    Accuracy=Accuracy+perf_eval[0]
    BalAcc=BalAcc+perf_eval[1]
    Fscore=Fscore+perf_eval[2]
print "avg acc:",float(Accuracy)/10.0
print "avg bal acc:",float(BalAcc)/10.0
print "avg fscore:",float(Fscore)/10.0