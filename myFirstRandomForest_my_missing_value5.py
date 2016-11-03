# -*- coding: utf-8 -*-
"""
Created on Thu Nov 03 10:31:22 2016

@author: Gaetani_Pa
"""

import pandas as pd
import numpy as np
import csv as csv
from sklearn.ensemble import RandomForestClassifier


###############################################################################################################
# Create a new train.csv (mttrain.csv) to FILL IN MISSING VALUES for Age creating a new variable NewAge

train_file = open('train.csv', 'rb')
train_file_object = csv.reader(train_file)
header = train_file_object.next() 

train_file_filled = open("mytrain.csv", "wb")
train_file_filled_object = csv.writer(train_file_filled)
train_file_filled_object.writerow(["PassengerId","Survived","Pclass","Name","Sex","Age","SibSp","Parch","Ticket","Fare","Cabin","Embarked","NewAge"])

for row in train_file_object:

    #age missing values raplaced with 36 for passengers in class 1, 29 fro passengers in class 2, 22 for passengers in class 3    
    
    if row[2] =='1' and len(row[5])==0: 
        train_file_filled_object.writerow([row[0],row[1],row[2],row[3],row[4],row[5],row[6],row[7],row[8],row[9],row[10],row[11],36])      
    if row[2]=='2' and len(row[5])==0:
        train_file_filled_object.writerow([row[0],row[1],row[2],row[3],row[4],row[5],row[6],row[7],row[8],row[9],row[10],row[11],29]) 
    if row[2]=='3' and len(row[5])==0:
        train_file_filled_object.writerow([row[0],row[1],row[2],row[3],row[4],row[5],row[6],row[7],row[8],row[9],row[10],row[11],22])
    if len(row[5])>0:
        train_file_filled_object.writerow([row[0],row[1],row[2],row[3],row[4],row[5],row[6],row[7],row[8],row[9],row[10],row[11],row[5]])
    

train_file.close()
train_file_filled.close()


#mytrain.csv in a data frame
train_df = pd.read_csv('mytrain.csv', header=0)

# female = 0, Male = 1
train_df['Gender'] = train_df['Sex'].map( {'female': 0, 'male': 1} ).astype(int)

# All missing Embarked -> just make them embark from most common place
if len(train_df.Embarked[ train_df.Embarked.isnull() ]) > 0:
    train_df.Embarked[ train_df.Embarked.isnull() ] = train_df.Embarked.dropna().mode().values

Ports = list(enumerate(np.unique(train_df['Embarked'])))    # determine all values of Embarked,
Ports_dict = { name : i for i, name in Ports }              # set up a dictionary in the form  Ports : index
train_df.Embarked = train_df.Embarked.map( lambda x: Ports_dict[x]).astype(int)     # Convert all Embark strings to int



# Remove not-used variables from train_df
ids_train = train_df['PassengerId'].values    #save passengers ID
survived_train=train_df['Survived'].values    #save survive real (observed)
train_df = train_df.drop(['Name', 'Sex','Age', 'Ticket', 'Cabin', 'PassengerId'], axis=1) 

trainAccurancy_df=train_df.drop(['Survived'], axis=1 )  #created to validate the model later





###############################################################################################################
# Create a new test.csv (mytest.csv) to FILL IN MISSING VALUES for Age creating a new variable NewAge

test_file = open('test.csv', 'rb')
test_file_object = csv.reader(test_file)
header = test_file_object.next() 

test_file_filled = open("mytest.csv", "wb")
test_file_filled_object = csv.writer(test_file_filled)
test_file_filled_object.writerow(["PassengerId","Pclass","Name","Sex","Age","SibSp","Parch","Ticket","Fare","Cabin","Embarked","NewAge"])

for row in test_file_object:

    #age missing values raplaced with 36 for passengers in class 1, 29 fro passengers in class 2, 22 for passengers in class 3    
    
    if row[1] =='1' and len(row[4])==0: 
        test_file_filled_object.writerow([row[0],row[1],row[2],row[3],row[4],row[5],row[6],row[7],row[8],row[9],row[10],36])      
    if row[1]=='2' and len(row[4])==0:
        test_file_filled_object.writerow([row[0],row[1],row[2],row[3],row[4],row[5],row[6],row[7],row[8],row[9],row[10],29]) 
    if row[1]=='3' and len(row[4])==0:
        test_file_filled_object.writerow([row[0],row[1],row[2],row[3],row[4],row[5],row[6],row[7],row[8],row[9],row[10],22])
    if len(row[4])>0:
        test_file_filled_object.writerow([row[0],row[1],row[2],row[3],row[4],row[5],row[6],row[7],row[8],row[9],row[10],row[4]])
    

test_file.close()
test_file_filled.close()


#mytest.csv in a data frame
test_df = pd.read_csv('mytest.csv', header=0)

# female = 0, Male = 1
test_df['Gender'] = test_df['Sex'].map( {'female': 0, 'male': 1} ).astype(int)

# All missing Embarked -> just make them embark from most common place
if len(test_df.Embarked[ test_df.Embarked.isnull() ]) > 0:
    test_df.Embarked[ test_df.Embarked.isnull() ] = test_df.Embarked.dropna().mode().values
# Again convert all Embarked strings to int
test_df.Embarked = test_df.Embarked.map( lambda x: Ports_dict[x]).astype(int)

# All the missing Fares -> assume median of their respective class
if len(test_df.Fare[ test_df.Fare.isnull() ]) > 0:
    median_fare = np.zeros(3)
    for f in range(0,3):                                              # loop 0 to 2
        median_fare[f] = test_df[ test_df.Pclass == f+1 ]['Fare'].dropna().median()
    for f in range(0,3):                                              # loop 0 to 2
        test_df.loc[ (test_df.Fare.isnull()) & (test_df.Pclass == f+1 ), 'Fare'] = median_fare[f]

# Collect the test data's PassengerIds before dropping it
ids = test_df['PassengerId'].values
# Remove not-used variables from test_df
test_df = test_df.drop(['Name', 'Sex','Age', 'Ticket', 'Cabin', 'PassengerId'], axis=1) 





#############################################################################################
# The data is now ready to go. So lets fit to the train, then predict to the test!
# Convert back to a numpy array
train_data = train_df.values
train_accurnacy_data=trainAccurancy_df.values
test_data = test_df.values


print 'Training...'
forest = RandomForestClassifier(n_estimators=100)
forest = forest.fit( train_data[0::,1::], train_data[0::,0] )
output1 = forest.predict(train_accurnacy_data).astype(int)
    


#predictive power valuation
predictive_power_file = open("myfirstforestaccurancy5.csv", "wb")
open_file_object = csv.writer(predictive_power_file)
open_file_object.writerow(["PassengerId","Survived_real","Survived_predicted"])
open_file_object.writerows(zip(ids_train, survived_train, output1))
predictive_power_file.close()



csv_file_object1 = csv.reader(open("myfirstforestaccurancy5.csv", 'rb'))
header1 = csv_file_object1.next()
data_predicted=[]
for row in csv_file_object1:
    data_predicted.append(row[0:])
data_predicted = np.array(data_predicted)

number_survived_real = np.sum(data_predicted[0::,1].astype(np.float))
number_survived_predicted = np.sum(data_predicted[0::,2].astype(np.float))

proportion_survivors_predicted = number_survived_predicted / number_survived_real

print number_survived_real
print number_survived_predicted
print proportion_survivors_predicted


print 'Predicting...'
output = forest.predict(test_data).astype(int)

#model application to test.csv
predictions_file = open("myfirstforest5.csv", "wb")
open_file_object = csv.writer(predictions_file)
open_file_object.writerow(["PassengerId","Survived"])
open_file_object.writerows(zip(ids, output))
predictions_file.close()

print 'Done.'










