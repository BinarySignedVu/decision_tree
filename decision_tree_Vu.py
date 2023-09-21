#-------------------------------------------------------------------------
# AUTHOR: Vu Nguyen
# FILENAME: decision_tree_Vu.py
# SPECIFICATION: This program forms a decision tree given curated data
# FOR: CS 4210- Assignment #1
# TIME SPENT: 8 hours
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard
# dictionaries, lists, and arrays

#importing some Python libraries
from sklearn import tree
import matplotlib.pyplot as plt
import csv

db = []
X = []
Y = []

# temp function to print matrix
def print_matrix(matrix):
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            text = matrix[i][j]
            if type(text) == int:
                print(f'{text}', end='\t')
            else:
                print(f'{text[0:4]}', end='\t')
        print()

#reading the data in a csv file
with open('contact_lens.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    for i, row in enumerate(reader):
        if i > 0: #skipping the header
            db.append(row)
            print(row)

#transform the original categorical training features into numbers and add to the 4D array X.
#For instance Young = 1, Prepresbyopic = 2, Presbyopic = 3
# so X = [[1, 1, 1, 1], [2, 2, 2, 2], ...]]
#--> add your Python code here
# Mapping for each category
age_map = {'Young': 1, 'Prepresbyopic': 2, 'Presbyopic': 3}
spectacle_map = {'Myope': 1, 'Hypermetrope': 2}
astigmatism_map = {'No': 1, 'Yes': 2}
tear_map = {'Normal': 1, 'Reduced': 2}

# Construct X by mapping each category
for row in db:
    age, spectacle, astigmatism, tear = row[:-1]
    X.append([age_map[age], spectacle_map[spectacle], astigmatism_map[astigmatism], tear_map[tear]])

# Print transformed feature matrix
print_matrix(X)
print()

#transform the original categorical training classes into numbers and add to the vector Y.
#For instance Yes = 1, No = 2, so Y = [1, 1, 2, 2, ...]
#--> add your Python code here
lens_map = {'Yes': 1, 'No': 2}
Y = [lens_map[row[-1]] for row in db]

for i in Y:
    print(i, end=' ')
print()

#fitting the decision tree to the data
clf = tree.DecisionTreeClassifier(criterion='entropy')
clf = clf.fit(X, Y)

#plotting the decision tree
tree.plot_tree(clf, feature_names=['Age', 'Spectacle', 'Astigmatism', 'Tear'], class_names=['Yes','No'], filled=True, rounded=True)
plt.show()
