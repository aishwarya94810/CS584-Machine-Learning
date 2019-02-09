#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 19:29:08 2019

@author: aishu
"""
"""
NAME: AISHWARYA ANANTHARAM
CWID: A20396732
"""


# Load the necessary libraries
import matplotlib.pyplot as matplot
import numpy
import pandas
import sklearn.tree as tree
from scipy.stats import iqr
import math
from math import exp, expm1
from numpy import linalg as LIA1
import scipy
from scipy import linalg as LIA2
from sklearn.neighbors import NearestNeighbors as knn
from sklearn.neighbors import KNeighborsClassifier as knc


trainData = pandas.read_csv('/Users/aishu/Desktop/IITC/Spring2019/ML/Assignments/A1/NormalSample.csv',delimiter=',')
# Put the descriptive statistics into another dataframe
trainData_descriptive = trainData.describe()
print(trainData_descriptive)

#numpy.histogram(trainData_descriptive)
#nh=numpy.histogram(trainData_descriptive, bins='fd')
#print("numpy histogram is: ",nh)
#median of x
#https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.histogram.html 
median_x = trainData['x'].median()
print("median is: ",median_x)
#calculating the interquartile range(iqr)
iqr=iqr(trainData['x'])
print("The interquartile range is: ",iqr)
count_value = trainData['x'].count()

print("value of N is: ",count_value)
yy= math.pow(count_value, -1/3)
print(yy)

#using the Izenman (1991) method(formula)
h=2*(iqr) * yy
print("1a")
print("The recommended Bin width for histogram of x is: ", h)
print("\n")

print("----------------------------------------------")
#printing minimum value of x
print("Solution for 1b:")
min_value=trainData['x'].min()
print(min_value)
#printing maximum value of x

max_value=trainData['x'].max()
print(max_value)
print("\n")
print("----------------------------------------------")
#Source link: https://www.geeksforgeeks.org/floor-ceil-function-python/
#math.floor() gives the largest integer not greater than x.
print("1c")
a=int(math.floor(min_value))-1
print("Value of a is=",a)
#math.ceil() gives the Smallest integer not less than x.
b=int(math.ceil(max_value))
print("Value of b is=",b)

print("\n")
print("----------------------------------------------")
#question 1d
# Visualize the histogram of the x variable
#From the Week 1 MyFirstDecisionTree.py code 
#Source: https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.arange.html
print("1d")
trainData.hist(column='x', bins=numpy.arange(a,b+0.1,0.1)) #value of 2 represents the value of h which is 0.1,0.5, 1 and 2.
matplot.title("Histogram of x")
matplot.xlabel("x")
matplot.ylabel("Number of Observations")
matplot.xticks(numpy.arange(a,b,step=1))
matplot.grid(axis="x")
print("h=2")
matplot.show()
print("\n")
print("----------------------------------------------")
#SOURCE: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.quantile.html
print("2a")
q1=trainData['x'].quantile([0.25])
print("q1 is",q1)
minwhisker= q1-(1.5 * iqr)
print("Value of min whisker is:",minwhisker)
print("\n")
#upper or max whisker
q3=trainData['x'].quantile([0.75])
print("q3 is",q3)
maxwhisker=q3+(1.5*iqr)
print("Value of max whisker is:",maxwhisker)

print("----------------------------------------------")
print("2b")
trainData = pandas.read_csv('/Users/aishu/Desktop/IITC/Spring2019/ML/Assignments/A1/NormalSample.csv',delimiter=',')
# Put the descriptive statistics into another dataframe
trainData_descriptive = trainData.describe()
print(trainData_descriptive)

print("group=0")
m = trainData[trainData ['group']==0]
print("m",m)
m_describe = m.describe()
print(m_describe)

print('\n')

median_x = m['x'].median()
print("median is: ",median_x)

mean_x = m['x'].mean()
print("mean is: ",mean_x)

#printing minimum value of x  
min_value=m['x'].min()
print("min_value",min_value)

#printing maximum value of x
max_value=m['x'].max()
print("max_value",max_value)

q1_0 = numpy.percentile(m['x'], [25])
q2_0 = numpy.percentile(m['x'], [50])
q3_0 = numpy.percentile(m['x'], [75])
print("quartiles are",q1_0,q2_0,q3_0)

print("Value of min whisker is:",minwhisker)
print("\n")
#upper or max whisker
upper_q3_0=trainData['x'].quantile([0.75])
print("q3 is",upper_q3_0)
maxwhisker=upper_q3_0+(1.5*iqr)
print("Value of max whisker is:",maxwhisker)

print("\n\n")
print("----------------------------------------------")
print("2b")
print("group=1")
n = trainData[trainData ['group']==1]
print("n",n)
n_describe = n.describe()
print(n_describe)

print('\n')

median_x = n['x'].median()
print("median is: ",median_x)

mean_x = n['x'].mean()
print("mean is: ",mean_x)

#printing minimum value of x  
min_value=n['x'].min()
print("min_value",min_value)

#printing maximum value of x
max_value=n['x'].max()
print("max_value",max_value)

q1_1 = numpy.percentile(n['x'], [25])
q2_1 = numpy.percentile(n['x'], [50])
q3_1= numpy.percentile(n['x'], [75])
print("quartiles are",q1_1,q2_1,q3_1)

print("Value of min whisker is:",minwhisker)
print("\n")
#upper or max whisker
upper_q3_1=trainData['x'].quantile([0.75])
print("q3 is",q3)
maxwhisker=upper_q3_1+(1.5*iqr)
print("Value of max whisker is:",maxwhisker)


print("\n")
print("----------------------------------------------")
print("2c")
trainData.boxplot(column='x', vert=False)
matplot.title("Boxplot of x ")
matplot.suptitle("")
matplot.xlabel("x")
matplot.ylabel(" ")
matplot.grid(axis="y")
matplot.show()

print("----------------------------------------------")
print("2d")
#BOX PLOT
#Source: https://matplotlib.org/2.1.2/gallery/pyplots/boxplot_demo.html

boxplot_data=[trainData['x'],m['x'],n['x']]

matplot.boxplot(boxplot_data, vert=False)

matplot.title("Boxplot of x by Levels of group")
matplot.suptitle("")
matplot.xlabel("x")
matplot.ylabel("group")
matplot.grid(axis="y")
matplot.show()

print("----------------------------------------------")



print("3a")
#group by fraud='1'
fraudData = pandas.read_csv('/Users/aishu/Desktop/IITC/Spring2019/ML/Assignments/A1/Fraud.csv',delimiter=',')
fraud_details= fraudData.groupby('FRAUD').describe()
print("Details of fraud",fraud_details)

fraud_1 = fraudData[fraudData ['FRAUD']==1].count()
#print("Fradulant investigations are: ",fraud_1)
fraud_total_count = fraudData['FRAUD'].count()
#print("Total number of fraud's are: ",fraud_total_count)

percentage_of_fradulant_invst = fraud_1/fraud_total_count * 100
print("percentage of fradulant investigations are: ",percentage_of_fradulant_invst)

fraud_0 = fraudData[fraudData ['FRAUD']==0].count()
#print("Fradulant investigations are: ",fraud_0)
fraud_total_count = fraudData['FRAUD'].count()
print("Total number of fraud's are: ",fraud_total_count)

print("----------------------------------------------")
print("3b")


fraudData.boxplot(column='TOTAL_SPEND', by='FRAUD' ,vert=False)
matplot.title(" ")
matplot.xlabel("TOTAL_SPEND")
matplot.ylabel("FRAUD")
matplot.show()


fraudData.boxplot(column='DOCTOR_VISITS', by='FRAUD' ,vert=False)
matplot.title(" ")
matplot.xlabel("DOCTOR_VISITS")
matplot.ylabel("FRAUD")
matplot.show()


fraudData.boxplot(column='NUM_CLAIMS', by='FRAUD' ,vert=False)
matplot.title(" ")
matplot.xlabel("NUM_CLAIMS")
matplot.ylabel("FRAUD")
matplot.show()

fraudData.boxplot(column='MEMBER_DURATION', by='FRAUD' ,vert=False)
matplot.title(" ")
matplot.xlabel("MEMBER_DURATION")
matplot.ylabel("FRAUD")
matplot.show()

fraudData.boxplot(column='OPTOM_PRESC', by='FRAUD' ,vert=False)
matplot.title(" ")
matplot.xlabel("OPTOM_PRESC")
matplot.ylabel("FRAUD")
matplot.show()

fraudData.boxplot(column='NUM_MEMBERS', by='FRAUD' ,vert=False)
matplot.title(" ")
matplot.xlabel("NUM_MEMBERS")
matplot.ylabel("FRAUD")
matplot.show()


print("----------------------------------------------")


print("3c(i)")
fraud_trainData = fraudData[['TOTAL_SPEND','DOCTOR_VISITS','NUM_CLAIMS','MEMBER_DURATION','OPTOM_PRESC','NUM_MEMBERS']]
ftrain=fraud_trainData.describe()
print("ftrain",ftrain)

fraud_testData=fraudData[['FRAUD']]
ftest=fraud_testData.describe()
print("ftest",ftest)

fraud_matrix = numpy.matrix(fraud_trainData)

orthnormalize = LIA2.orth(fraud_trainData)
print("The orthonormalization of x= \n", orthnormalize)
print(orthnormalize.shape)
print("\n")

print("\n")
print("----------------------------------------------")


print("3c(ii)")
# Check columns of the ORTH function
check_transpose = orthnormalize.transpose().dot(orthnormalize)
print("Also Expect an Identity Matrix = \n", check_transpose)
print("\n")
# Here is the transformation matrix
# Orthonormalized the training data
fraud_x = numpy.matrix(fraud_trainData.values)

xtx = fraud_x.transpose() * fraud_x
print("t(fraud_x) * fraud_x = \n", xtx)
print("\n")
# Eigenvalue decomposition
evals, evecs = numpy.linalg.eigh(xtx)
print("Eigenvalues of x = \n", evals)
print("\n")
print("Eigenvectors of x = \n",evecs)
print("\n")
# Here is the transformation matrix
transf = evecs * numpy.linalg.inv(numpy.sqrt(numpy.diagflat(evals)));
print("Transformation Matrix = \n", transf)
print("\n")
# Here is the transformed X
transf_x = fraud_x * transf;
print("The Transformed x = \n", transf_x)
print("\n")
# Check columns of transformed X
xtx = transf_x.transpose() * transf_x;
print("Expect an Identity Matrix = \n", xtx)

print("\n")
print("----------------------------------------------")


print("3d(i)")
print("KNN")
# Specify the kNN
kNNSpec = knc(n_neighbors = 5, algorithm = 'brute', metric = 'euclidean')

# Build nearest neighbors
nbrs = kNNSpec.fit(transf_x,fraud_testData)
#print(nbrs)
#knn_score = kNNSpec.accuracy_score(fraud_trainData , fraud_testData)
#print(knn_score)

#Source:https://www.ritchieng.com/machine-learning-k-nearest-neighbors-knn/
y_pred = nbrs.predict(fraud_trainData)
#print(transf_x)
accs=kNNSpec.score(transf_x,fraud_testData, sample_weight= None )

print("Score value is:",accs)

'''
n_value = kNNSpec
m_value=nbrs.kneighbors(fraud_testData, n_neighbors = 5, return_distance=True )

predict_val=nbrs.predict(fraud_testData)
print(predict_val)
'''

print("----------------------------------------------")

print("3e")
                    

focal=[[7500,15,3,127,2,2]]

print("Input variable names are:TOTAL_SPEND','DOCTOR_VISITS','NUM_CLAIMS','MEMBER_DURATION','OPTOM_PRESC','NUM_MEMBERS'")
print("focal variables are: ",focal)

focal_test=focal *transf
focal_neighbors=nbrs.kneighbors(focal_test,return_distance = False)
print("\n My neighbors:", focal_neighbors)
    
myNeighbors = nbrs.kneighbors(focal, return_distance = False)
print("My Neighbors = \n", myNeighbors)

# Here is the transformation matrix
transf = evecs * numpy.linalg.inv(numpy.sqrt(numpy.diagflat(evals)));
print("Transformation Matrix = \n", transf)

print("\n")
print("----------------------------------------------")
print("3f")
nbrs.predict(focal * transf)
print_proba=nbrs.predict_proba(focal)

print("prediceted focal:",print_proba)



