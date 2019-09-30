# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 17:32:26 2019

@author: Swati
"""
import scipy.stats as ss
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#data generation

def generateData(number_points):
    dataset1=np.random.multivariate_normal([1,0],[[1,0.75],[0.75,1]],number_points)
    dataset2=np.random.multivariate_normal([0,1],[[1,0.75],[0.75,1]],number_points)
    X=np.append(dataset1,dataset2,axis=0)
    testset1=np.random.multivariate_normal([1,0],[[1,0.75],[0.75,1]],500)
    testset2=np.random.multivariate_normal([0,1],[[1,0.75],[0.75,1]],500)
    test=np.append(testset1,testset2,axis=0)
    ze=np.zeros(number_points).T
    on=np.ones(number_points).T
    ze_test=np.zeros(500).T
    on_test=np.ones(500).T
    labels_train=np.append(ze,on,axis=0)
    labels_test=np.append(ze_test,on_test,axis=0)
    return (X,test,labels_train,labels_test)

def generateData_part_4():
    dataset1=np.random.multivariate_normal([1,0],[[1,0.75],[0.75,1]],500)
    dataset2=np.random.multivariate_normal([0,1],[[1,0.75],[0.75,1]],500)
    X=np.append(dataset1,dataset2,axis=0)
    testset1=np.random.multivariate_normal([1,0],[[1,0.75],[0.75,1]],500)
    testset2=np.random.multivariate_normal([0,1],[[1,0.75],[0.75,1]],500)
    test=np.append(testset1,testset2,axis=0)
    ze=np.zeros(700).T
    on=np.ones(300).T
    ze_test=np.zeros(500).T
    on_test=np.ones(500).T
    labels_train=np.append(ze,on,axis=0)
    labels_test=np.append(ze_test,on_test,axis=0)
    return (X,test,labels_train,labels_test)

def myNB(X,Y,X_test,Y_test):
    df=pd.DataFrame((X),columns=['X','Y'])
    df['label']=Y
    
    #Training
    #seperating the dataframe according to the labels
    label_0=df[df['label']==0]
    label_1=df[df['label']==1]
    del label_0['label']
    del label_1['label']
    
    #Calculating the mean of each class 0 and 1
    label_0_mean=label_0.mean()
    label_1_mean=label_1.mean()

    
    #Calculating the standard deviation according to the class 0 and 1 
    label_0_std=label_0.std()
    label_1_std=label_1.std()

    #Calculating the prior of both the class labels 0 and 1
    prior_0=len(label_0)/(len(label_0)+len(label_1))
    prior_1=len(label_1)/(len(label_0)+len(label_1))

    
    #Calculating summary that is mean and standard deviations of both the labels 
    class_0_summary=[label_0_mean['X'],label_0_std['X']],[label_0_mean['Y'],label_0_std['Y']]
    class_1_summary=[label_1_mean['X'],label_1_std['X']],[label_1_mean['Y'],label_1_std['Y']]

    #Testing
    #calculate the probability 
    class0=[]
    class1=[]
    df_test=pd.DataFrame((X_test),columns=['X','Y'])
    for rows in df_test.iterrows():
        class0.append((ss.norm.pdf(rows[1]['X'],label_0_mean['X'],label_0_std['X']))*(ss.norm.pdf(rows[1]['Y'],label_0_mean['Y'],label_0_std['Y']))*prior_0)
        class1.append((ss.norm.pdf(rows[1]['X'],label_1_mean['X'],label_1_std['X']))*(ss.norm.pdf(rows[1]['Y'],label_1_mean['Y'],label_1_std['Y'])*prior_1))
    df_test['class_0_prob']=class0
    df_test['class_1_prob']=class1
    
    #prediction
    prediction=[]
    for rows in df_test.iterrows():
        if rows[1]['class_0_prob']>rows[1]['class_1_prob']:
            prediction.append(0)
        else:
            prediction.append(1)
            
    df_test['predict']=prediction
    
    df_test['actual']=Y_test
    
    # Calculating accuracy
    count=0
    
    for rows in df_test.iterrows():
        if rows[1]['predict']==rows[1]['actual']:
            count+=1
    
    accuracy=count/(df_test.shape[0])
    err=1-accuracy
    tp=0
    fp=0
    tn=0
    fn=0
    class_0_x=[]
    class_0_y=[]
    class_1_x=[]
    class_1_y=[]
    for rows in df_test.iterrows():
        if rows[1]['predict']==rows[1]['actual']:
            if rows[1]['predict']==1:
                tp+=1
            else:
                tn+=1
        elif rows[1]['predict']==1 and rows[1]['actual']==0:
            fp+=1
        else:
            fn+=1
        if rows[1]['predict']==0:
            class_0_x.append(rows[1]['X'])
            class_0_y.append(rows[1]['Y'])
        else:
            class_1_x.append(rows[1]['X'])
            class_1_y.append(rows[1]['Y'])
            plt.figure('ScatterPlot of labeled Data')
    plt.scatter(class_0_x,class_0_y,1,'blue')        
    plt.scatter(class_1_x,class_1_y,1,'red')
    
    #Printing Accuracy
    accuracy=((tp+tn)/(tp+tn+fp+fn))*100
    print('accuracy: ',accuracy)
    
    #recall
    recall=(tp/(tp+fn))*100
    print('recall:',recall)
    
    #precision
    precision=(tp/(tp+fp))*100
    print('precision:',precision)
    
    #Confusion matrix
    conf_matrix=pd.DataFrame([[tp,fn],[fp,tn]],columns=['Actual 1','Actual 0'])
    print('Confusion matrix\n:',conf_matrix)
    
    
    #roc_calculating(df_test)
    po = pd.DataFrame(class0,columns=['class_0_prob'])
    po['class_1_prob'] = class1
    return (prediction,po,err)


def roc_calculating(df_test):
    tp=0
    fp=0
    tpr=[]
    fpr=[]
    actual_p=0
    actual_n=0
    fpr_prv=0
    auc=0
    #plt.clf()
    df_test=df_test.sort_values(by='class_1_prob', ascending=False)
    for rows in df_test.iterrows():
        if rows[1]['actual']==1:
            actual_p+=1
        else:
            actual_n+=1
    #print(actual_p,actual_n) 
    for rows in df_test.iterrows():
            if rows[1]['actual']==1:
                tp+=1
            else: 
                fp+=1
            tpr.append((tp/(actual_p)))
            fpr.append((fp/(actual_n)))
            auc+=((tp/actual_p))*((fp/actual_n)-fpr_prv)
            fpr_prv=(fp/actual_n)
    print('AUC:',auc)
            
    plt.figure('ROC for Part 1')
    plt.plot(fpr,tpr)
    plt.show()

def roc_calculating_1(df_test):
    tp=0
    fp=0
    tpr=[]
    fpr=[]
    actual_p=0
    actual_n=0
    fpr_prv=0
    auc=0
    #plt.clf()
    df_test=df_test.sort_values(by='class_1_prob', ascending=False)
    for rows in df_test.iterrows():
        if rows[1]['actual']==1:
            actual_p+=1
        else:
            actual_n+=1
    print(actual_p,actual_n) 
    for rows in df_test.iterrows():
            if rows[1]['actual']==1:
                tp+=1
            else: 
                fp+=1
            tpr.append((tp/(actual_p)))
            fpr.append((fp/(actual_n)))
            auc+=((tp/actual_p))*((fp/actual_n)-fpr_prv)
            fpr_prv=(fp/actual_n)
    print('AUC:',auc)
            
    plt.figure('ROC for Part 4')
    plt.plot(fpr,tpr)
    plt.show()

#calling the functions
def part_1_2():
    a,b,c,d=generateData(500)
    pred,posterior,err=myNB(a,c,b,d)
    frame = pd.DataFrame(posterior, columns=['class_0_prob','class_1_prob'])
    frame['actual']=d
    roc_calculating(frame)

def part_3():
    data_list=[10,20,50,100,300,500]
    accuracy=[]
    for i in range (len(data_list)):
        a,b,c,d=generateData(data_list[i])
        pred,posterior,err=myNB(a,c,b,d)
        accuracy.append(1-err)
    plt.figure('Accuracy vs Data points')
    plt.plot(data_list,accuracy)
    plt.show()
    
    
def part_4():
    a,b,c,d=generateData_part_4()
    pred,posterior,err=myNB(a,c,b,d)        
    print(type(posterior))
    frame = pd.DataFrame(posterior, columns=['class_0_prob','class_1_prob'])
    frame['actual']=d
    roc_calculating_1(frame)
    
    
print('Part 1 and Part 2')
print('-----------------')
part_1_2()
print('Part 3')
print('------')
part_3()
print('Part 4')
print('------')
part_4()