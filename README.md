# Voice-Categoriztion-SVM with GridSearchCV

### Import Libraries
- import pandas as pd 
- import numpy as np
- import matplotlib.pyplot as plt 
- import seaborn as sns
- import warnings
- warnings.filterwarnings("ignore")

### Load Dataset

* (rows, col)=(3168, 21)
* 20 features
* Label=> male and female
* from data its looking like a noramlize data as its values in between 0-1

### Data Pre-Processing

* no null values
* all data in float expect the target  
* use LabelEncoder for target coloum (male=1, female=0)

### Scaling-Preprocessing

* StandardScalar=> used std to scale data in bet -1 to 1
* MinMaxScalar=> used normalize the data into any range
* scalling part is applied on features

### Train-test Split

- print(X_train.shape)=>(2217, 20)
- print(X_test.shape)=>(951, 20)

### Train a Model
SVM Model

### Model Evalution
'''
- ****** prediction on test data *******
- Confusion Matrix
- [[477  11]
-  [ 10 453]]
- ---------------------------------------------------

- Classification Report
-               precision    recall  f1-score   support
 
-            0       0.98      0.98      0.98       488
-            1       0.98      0.98      0.98       463
-     accuracy                           0.98       951


### Hyperparameter

* kernel = 'linear','rbf','Poly'
* C value, Higher the 'C' values decision boundry will be hard margin. Lower the C value decision boundry will be soft margin.
* gamma = Choose this as decimal values [0.1, 0.01, 0.0001]
* for multiclass classification choose option b/w 'ovr' (one vs rest all) & ovo (one class vs another class)


**How to select best parameter for model?**

**Ans:** Use GridSearchCV
* GridSearchCv can used for any algorithm to select best hyperparameter


### Pickling of Model
 
### Model Deployment on render
1. README file
2. requirements.txt
3. app.py
4. style.css
5. main.html
6. Dockerfile
7. main.yaml

### Deploy on Render