---
layout: post
title: Fraud Detection
tags: [Fraud,Credit Card]
---

Hey guys,

From my experience, I can tell that almost everyone has already received a suspicious call asking for confidential information or clicked on a link that could steal important information. In this kind of situation, the person is trying to use your information to access your credit card. It is just one example of many frauds that may happen.

The goal of this study is to develop a **Fraud Detection** algorithm using several machine learning techniques.

_Dataset :_ I used the Kaggle dataset - [Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud) The datasets contains transactions made by credit cards in September 2013 by European cardholders, as mentioned in Kaggle description.

**Key Takeaways:**
1. Skewed dataset: How to deal with it.
2. Principal Component Analysis.
3. Application of Multivariate Gaussian distribution and other machine learning methods.



```python
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

%matplotlib inline
```


```python
#Importing dataset

df=pd.read_csv('../data/creditcard.csv')
```

## Introduction

The dataset contains numerical values after PCA transformation. _The original data could not be shared due to confidentiality issues_.
Additionally, two features are provided: **Time and Amount**.

The output is a categoric variable that contains two values - 0 as no fraud and 1 as a fraud.

There are 30 features: Time, Amount and 28 numerical vectors after PCA transformation.


```python
#Let's split the data into three segments: Output, components and other variables
V=[]
for i in range(1,29):
    V.append('V'+str(i))
components = df[V]
output = df['Class']
other_var=df[['Time','Amount']]

df.head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Time</th>
      <th>V1</th>
      <th>V2</th>
      <th>V3</th>
      <th>V4</th>
      <th>V5</th>
      <th>V6</th>
      <th>V7</th>
      <th>V8</th>
      <th>V9</th>
      <th>...</th>
      <th>V21</th>
      <th>V22</th>
      <th>V23</th>
      <th>V24</th>
      <th>V25</th>
      <th>V26</th>
      <th>V27</th>
      <th>V28</th>
      <th>Amount</th>
      <th>Class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>-1.359807</td>
      <td>-0.072781</td>
      <td>2.536347</td>
      <td>1.378155</td>
      <td>-0.338321</td>
      <td>0.462388</td>
      <td>0.239599</td>
      <td>0.098698</td>
      <td>0.363787</td>
      <td>...</td>
      <td>-0.018307</td>
      <td>0.277838</td>
      <td>-0.110474</td>
      <td>0.066928</td>
      <td>0.128539</td>
      <td>-0.189115</td>
      <td>0.133558</td>
      <td>-0.021053</td>
      <td>149.62</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.0</td>
      <td>1.191857</td>
      <td>0.266151</td>
      <td>0.166480</td>
      <td>0.448154</td>
      <td>0.060018</td>
      <td>-0.082361</td>
      <td>-0.078803</td>
      <td>0.085102</td>
      <td>-0.255425</td>
      <td>...</td>
      <td>-0.225775</td>
      <td>-0.638672</td>
      <td>0.101288</td>
      <td>-0.339846</td>
      <td>0.167170</td>
      <td>0.125895</td>
      <td>-0.008983</td>
      <td>0.014724</td>
      <td>2.69</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.0</td>
      <td>-1.358354</td>
      <td>-1.340163</td>
      <td>1.773209</td>
      <td>0.379780</td>
      <td>-0.503198</td>
      <td>1.800499</td>
      <td>0.791461</td>
      <td>0.247676</td>
      <td>-1.514654</td>
      <td>...</td>
      <td>0.247998</td>
      <td>0.771679</td>
      <td>0.909412</td>
      <td>-0.689281</td>
      <td>-0.327642</td>
      <td>-0.139097</td>
      <td>-0.055353</td>
      <td>-0.059752</td>
      <td>378.66</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.0</td>
      <td>-0.966272</td>
      <td>-0.185226</td>
      <td>1.792993</td>
      <td>-0.863291</td>
      <td>-0.010309</td>
      <td>1.247203</td>
      <td>0.237609</td>
      <td>0.377436</td>
      <td>-1.387024</td>
      <td>...</td>
      <td>-0.108300</td>
      <td>0.005274</td>
      <td>-0.190321</td>
      <td>-1.175575</td>
      <td>0.647376</td>
      <td>-0.221929</td>
      <td>0.062723</td>
      <td>0.061458</td>
      <td>123.50</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2.0</td>
      <td>-1.158233</td>
      <td>0.877737</td>
      <td>1.548718</td>
      <td>0.403034</td>
      <td>-0.407193</td>
      <td>0.095921</td>
      <td>0.592941</td>
      <td>-0.270533</td>
      <td>0.817739</td>
      <td>...</td>
      <td>-0.009431</td>
      <td>0.798278</td>
      <td>-0.137458</td>
      <td>0.141267</td>
      <td>-0.206010</td>
      <td>0.502292</td>
      <td>0.219422</td>
      <td>0.215153</td>
      <td>69.99</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2.0</td>
      <td>-0.425966</td>
      <td>0.960523</td>
      <td>1.141109</td>
      <td>-0.168252</td>
      <td>0.420987</td>
      <td>-0.029728</td>
      <td>0.476201</td>
      <td>0.260314</td>
      <td>-0.568671</td>
      <td>...</td>
      <td>-0.208254</td>
      <td>-0.559825</td>
      <td>-0.026398</td>
      <td>-0.371427</td>
      <td>-0.232794</td>
      <td>0.105915</td>
      <td>0.253844</td>
      <td>0.081080</td>
      <td>3.67</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>4.0</td>
      <td>1.229658</td>
      <td>0.141004</td>
      <td>0.045371</td>
      <td>1.202613</td>
      <td>0.191881</td>
      <td>0.272708</td>
      <td>-0.005159</td>
      <td>0.081213</td>
      <td>0.464960</td>
      <td>...</td>
      <td>-0.167716</td>
      <td>-0.270710</td>
      <td>-0.154104</td>
      <td>-0.780055</td>
      <td>0.750137</td>
      <td>-0.257237</td>
      <td>0.034507</td>
      <td>0.005168</td>
      <td>4.99</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>7.0</td>
      <td>-0.644269</td>
      <td>1.417964</td>
      <td>1.074380</td>
      <td>-0.492199</td>
      <td>0.948934</td>
      <td>0.428118</td>
      <td>1.120631</td>
      <td>-3.807864</td>
      <td>0.615375</td>
      <td>...</td>
      <td>1.943465</td>
      <td>-1.015455</td>
      <td>0.057504</td>
      <td>-0.649709</td>
      <td>-0.415267</td>
      <td>-0.051634</td>
      <td>-1.206921</td>
      <td>-1.085339</td>
      <td>40.80</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>7.0</td>
      <td>-0.894286</td>
      <td>0.286157</td>
      <td>-0.113192</td>
      <td>-0.271526</td>
      <td>2.669599</td>
      <td>3.721818</td>
      <td>0.370145</td>
      <td>0.851084</td>
      <td>-0.392048</td>
      <td>...</td>
      <td>-0.073425</td>
      <td>-0.268092</td>
      <td>-0.204233</td>
      <td>1.011592</td>
      <td>0.373205</td>
      <td>-0.384157</td>
      <td>0.011747</td>
      <td>0.142404</td>
      <td>93.20</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>9.0</td>
      <td>-0.338262</td>
      <td>1.119593</td>
      <td>1.044367</td>
      <td>-0.222187</td>
      <td>0.499361</td>
      <td>-0.246761</td>
      <td>0.651583</td>
      <td>0.069539</td>
      <td>-0.736727</td>
      <td>...</td>
      <td>-0.246914</td>
      <td>-0.633753</td>
      <td>-0.120794</td>
      <td>-0.385050</td>
      <td>-0.069733</td>
      <td>0.094199</td>
      <td>0.246219</td>
      <td>0.083076</td>
      <td>3.68</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>10 rows × 31 columns</p>
</div>



## Output - Class

**We begin analyzing the output variable - Class**

As we can see, the _data is skewed_ with only 0.17% of Class 1.

There are several ways to deal with skewed data. Here, we use:
* **Metric** - F1 score. Accuracy is not used here since it would give inaccurate results. Since F1 score is a harmonic mean between precision and recall, it is chosen to measure the performance of statistical learning models.


* **Data collection** - Even though it was not applied in this study, I added data collection because, if feasible, it is a good option. Data collection was not performed here since all numerical values are PCA transformations already and the original data was not given.


* **Training/Dev/Test set**: Caution must be taken when splitting the dataset to proper statistical model training. Here, the dataset is split as follows:
    * Training set: 80%
    * Dev set: 10%
    * Test set: 10%

Fortunately, there are no missing values as well.


```python
#Output
sns.countplot(x='Class',data=df)

#Data is skewed

zero_prop = round(output.value_counts()[0]/output.count()*100,4)
one_prop = round(100-zero_prop,4)

print('Class 0 is {} % of the data set and Class 1 is {} %.'.format(zero_prop,one_prop))
print('Data is skewed.')
```

    Class 0 is 99.8273 % of the data set and Class 1 is 0.1727 %.
    Data is skewed.



![png](/assets/img/post/fraud_detection/output_6_1.png)



```python
#Missing values
print('The output variable has {} % of missing values.'.format(output.isnull().sum()/len(output)*100))
```

    The output variable has 0.0 % of missing values.


## Feature Analysis - PCA

The principal component analysis is a statistical method to high dimensional data onto a lower-dimensional space by maximizing the variance of each dimension. 

In machine learning, it is commonly used as a way to prevent overfitting, but it is not recommended to be the first solution since it throws away information from the data - regularization may be the first option.

In this study, the numerical values after PCA transformation were already given. Here, I analyze the number of components we may consider following a threshold. The threshold is based on the explained variance and set equals 90%.


```python
#Components of PCA
### Explained Variance
var_vec = components.var(axis=0)
total_var = var_vec.sum()
exp_var = var_vec/total_var
exp_var_cum=np.cumsum(np.round(exp_var, decimals=4)*100)
thresh = np.ones(len(exp_var_cum))*90

#PCA plot
plt.subplots(figsize=(12,5))
plt.plot(exp_var_cum,label='components')
plt.plot(range(len(exp_var_cum)),thresh,label='threshold')
plt.tight_layout()
plt.ylabel('% of Explained Variance')
plt.xlabel('Components')
plt.title('PCA - Explained Variance')
plt.legend()

#Conclusion

#We can see that using until 20 components can explain at least 90% of the total variance.
#OBS.: Total variance is equal to the sum of the variance of all components.
```




    <matplotlib.legend.Legend at 0x7f1b1445b7d0>




![png](/assets/img/post/fraud_detection/output_9_1.png)


### Output and PCA: Plot

Even though we selected the first 20 components. Here, we are going to use V1, V2, and V3 to plot them with the output.
Using more than 3 components makes it difficult to visualize their relation with the output.



```python
#PCA and output plot

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(components['V1'],components['V2'],components['V3'],c=output)
ax.view_init(60, 35)
```


![png](/assets/img/post/fraud_detection/output_11_0.png)


### Features - Missing Values

Fortunately, there is no missing values.


```python
#Missing values - PCA Components
print('The components have {} % of missing values.'.format(components.isnull().sum().sum()/len(components)*100))
```

    The components have 0.0 % of missing values.



```python
#Other variables - Time and Amount
print('The other variables have {} % of missing values.'.format(other_var.isnull().sum().sum()/len(other_var)*100))
```

    The other variables have 0.0 % of missing values.


### Time and Amount variables: Distribution

It does not make sense to analyze the distribution of numerical since they are results of PCA transformation. Then, I analyze the distribution of Time and Amount variables and perform transformations, if needed.


```python
#Other variables - Plot
fig,axes=plt.subplots(1,2,figsize=(10, 5))

sns.distplot(other_var['Time'],ax=axes[0])
sns.distplot(other_var['Amount'],ax=axes[1])

plt.tight_layout()
```


![png](/assets/img/post/fraud_detection/output_16_0.png)



```python
#### Data Transformation - Box Cox x+1
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PowerTransformer

t = other_var['Time'].to_numpy()+1
amt = other_var['Amount'].to_numpy()+1

#Time
power = PowerTransformer(method='box-cox',standardize=True)
power.fit(t.reshape(-1,1))
t_std = power.transform(t.reshape(-1,1))

#Amount
power.fit(amt.reshape(-1,1))
amt_std = power.transform(amt.reshape(-1,1))

#Plot
fig,axes=plt.subplots(1,2,figsize=(10, 5))

sns.distplot(t_std,ax=axes[0])
axes[0].set_xlabel('Time')
sns.distplot(amt_std,ax=axes[1])
axes[1].set_xlabel('Amount')
plt.tight_layout()
```


![png](/assets/img/post/fraud_detection/output_17_0.png)



```python
#Updating new variables
df['Time'] = t_std
df['Amount'] = amt_std
```

### Model selection

**Note: Data is skewed.**
 We we have to be careful with several aspects: metrics, model selection,train/dev/test set and
 consider if data generation is an option.
 
 Metrics to be used - **_F1 score_**
 
I tried four different statistical models:
* Multivariate Gaussian Distribution
* Logistic Regression
* Support Vector Machine
* Random Forest


```python
### Splitting dataset

###### Class 0

class_0 = df[df['Class']==0].drop(['Class'],axis=1)
out_0 = output[output==0]

########## Without Class and Amount
[zero_train,zero_temp,y_zero_train,y_zero_temp] = train_test_split(class_0,out_0,test_size=0.2,random_state=53,shuffle=True)
[zero_dev,zero_test,y_zero_dev,y_zero_test] = train_test_split(zero_temp,y_zero_temp,test_size=0.5,random_state=50,shuffle=True)

##### Class 1
class_1 = df[df['Class']==1].drop(['Class'],axis=1)
out_1 = output[output==1]

########## Without Class and Amount
[one_train,one_temp,y_one_train,y_one_temp] = train_test_split(class_1,out_1,test_size=0.2,random_state=53,shuffle=True)
[one_dev,one_test,y_one_dev,y_one_test] = train_test_split(one_temp,y_one_temp,test_size=0.5,random_state=50,shuffle=True)


##### Consolidating(full - all variables, w/o full - no time and amount var)
## Train
x_train_full = pd.concat([zero_train, one_train], axis=0)
x_train = x_train_full.drop(['Time','Amount'],axis=1)
y_train = pd.concat([y_zero_train, y_one_train], axis=0)

#Dev
x_dev_full = pd.concat([zero_dev, one_dev], axis=0)
x_dev = x_dev_full.drop(['Time','Amount'],axis=1)
y_dev = pd.concat([y_zero_dev, y_one_dev], axis=0)

#Test
x_test_full = pd.concat([zero_test, one_test], axis=0)
x_test = x_test_full.drop(['Time','Amount'],axis=1)
y_test = pd.concat([y_zero_test, y_one_test], axis=0)
```

### Multivariate Gaussian Distribution

Here, we assume that the dataset follows a normal distribution. Then, we calculate the probability of each example and based on a threshold, we classify it.


```python
### Model - Anomaly Detection (Suppose all variable follow a normal distribution)
from scipy.stats import multivariate_normal

cov = np.cov(x_train,rowvar = False)
mu = np.mean(x_train)
prob_train = multivariate_normal.pdf(x_train,mean=mu,cov=cov)

#Choosing the best threshold
prob_dev = multivariate_normal.pdf(x_dev,mean=mu,cov=cov)
```


```python
### Dev set - Choosing the best threshold
from sklearn.metrics import f1_score

def select_eps(y_dev,prob_dev):
    bestEps = 0;
    bestF1 = 0;
    F1= 0;
    
    stepsize = (max(prob_dev)-min(prob_dev))/1000
    for eps in np.arange(min(prob_dev),max(prob_dev)+stepsize,stepsize):
        
        pred = prob_dev < eps

        F1 = f1_score(y_dev,pred)
        
        if F1 > bestF1:
            bestF1 = F1
            bestEps = eps
        
    return bestEps,bestF1
```


```python
# The best eps
[eps,F1] = select_eps(y_dev,prob_dev)
```


```python
#Test set Score
prob_test = multivariate_normal.pdf(x_test,mean=mu,cov=cov)
pred = prob_test < eps
F1_test = f1_score(y_test,pred)

print('Based on F1 score as metric, assuming the dataset follows a multivariate gaussian distribution, the score is {}%'.format(round(F1_test*100,4)))
```

    Based on F1 score as metric, assuming the dataset follows a multivariate gaussian distribution, the score is 0.7166%


### Logistic Regression - Train & Dev set

Here, I try two different logistic regression models:
* Without regularization - All components
* Without regularization - 20 components


```python
##### Data set with the first 20 components
## Train
x_train_20 = x_train.drop(['V21','V22','V23','V24','V25','V26','V27','V28'],axis=1)

#Dev
x_dev_20 = x_dev.drop(['V21','V22','V23','V24','V25','V26','V27','V28'],axis=1)

#Test
x_test_20 = x_test.drop(['V21','V22','V23','V24','V25','V26','V27','V28'],axis=1)
```


```python
### Creating a function to find the best regularization parameter and its F1 score
def best_f1(C,x_train,y_train,x_dev,y_dev):
    
    bestC = 0
    bestF1 = 0
    F1 = []
    
    for value in C:
        lreg = LogisticRegression(penalty='l2',C=value,fit_intercept = True, max_iter=500,random_state=46)
        lreg.fit(x_train,y_train)
        pred = lreg.predict(x_dev)
        F1.append(f1_score(y_dev,pred))
        
           
    return F1
```


```python
# Logistic Regression

from sklearn.linear_model import LogisticRegression

#First choice - No regularization
lreg1 = LogisticRegression(penalty='l2',C=1,fit_intercept = True, max_iter=500,random_state=46)
lreg1.fit(x_train,y_train)
pred1 = lreg1.predict(x_dev)
F1_reg1 = f1_score(y_dev,pred1)

#Second choice - No regularization, but with 20 components
lreg2 = LogisticRegression(penalty='l2',C=1,fit_intercept = True, max_iter=500,random_state=46)
lreg2.fit(x_train_20,y_train)
pred2 = lreg2.predict(x_dev_20)
F1_reg2 = f1_score(y_dev,pred2)

#Choosing the best logistic regression
F1 = [F1_reg1,F1_reg2]
i = F1.index(max(F1))

if i == 0:
    print('Logistic regression without regularization (All components) is the best model')
    print('F1 score: {}'.format(round(F1[i],4)))
else:
    print('Logistic regression without regularization (20 components) is the best model')
    print('F1 score: {}'.format(round(F1[i],4)))
```

    Logistic regression without regularization (All components) is the best model
    F1 score: 0.7674418604651163


### Support Vector Machine - Train & Dev set

Based on the score on dev set, we can see that SVM using only the first 20 components enhanced the performance and performed faster. Even though we could lose data information selecting only 20 components instead of a full pack, it actually helped!


```python
# SVM
from sklearn.svm import SVC
import time

# All components
t0 = time.time()
svc = SVC(kernel='rbf')
svc.fit(x_train,y_train)
pred = svc.predict(x_dev)
f1_svc1 = f1_score(y_dev,pred)
tf = time.time()
delta1 = tf-t0

# 20 components
t0 = time.time()
svc.fit(x_train_20,y_train)
pred = svc.predict(x_dev_20)
f1_svc2 = f1_score(y_dev,pred)
tf = time.time()
delta2 = tf-t0

print('All components : SVC model - f1 score: {} and it took {} seconds'.format(round(f1_svc1,4),round(delta1,4)))
print('20 components : SVC model - f1 score: {} and it took {} seconds'.format(round(f1_svc2,4),round(delta2,4)))

### Took too long
```

    All components : SVC model - f1 score: 0.8444444444444446 and it took 696.1894538402557
    20 components : SVC model - f1 score: 0.8695652173913043 and it took 447.6987636089325


### Random Forest - Train & Dev set


```python
# Random Forest - Finding the best number of trees
from sklearn.ensemble import RandomForestClassifier

est = [100,250,500,750,1000]
f1_forest = []
for value in est:
    forest = RandomForestClassifier(n_estimators=value,criterion ='gini' , max_features ='sqrt',n_jobs=-1)

    forest.fit(x_train,y_train)
    pred = forest.predict(x_dev)
    f1_forest.append(f1_score(y_dev,pred))
    
#Plot
plt.plot(est,f1_forest)
plt.xlabel('Number of trees')
plt.ylabel('F1 score')
plt.title('Random Forest - Optimum number of trees')
```




    Text(0.5, 1.0, 'Random Forest - Optimum number of trees')




![png](/assets/img/post/fraud_detection/output_33_1.png)



```python
est = [10,25,50,75,100]
f1_forest = []
for value in est:
    forest = RandomForestClassifier(n_estimators=value,criterion ='gini' , max_features ='sqrt',n_jobs=-1)

    forest.fit(x_train,y_train)
    pred = forest.predict(x_dev)
    f1_forest.append(f1_score(y_dev,pred))
    
#Plot
plt.plot(est,f1_forest)
plt.xlabel('Number of trees')
plt.ylabel('F1 score')
plt.title('Random Forest - Optimum number of trees')
```




    Text(0.5, 1.0, 'Random Forest - Optimum number of trees')




![png](/assets/img/post/fraud_detection/output_34_1.png)



```python
### Random Forest - best number of trees -> 10

forest = RandomForestClassifier(n_estimators=10,criterion = 'gini', max_features='sqrt',random_state=47)
t0=time.time()
forest.fit(x_train,y_train)
pred = forest.predict(x_dev)
f1_forest = f1_score(y_dev,pred)
t1=time.time()
delta1 = t1-t0

print('Random forest with 10 trees using all components: f1 score: {} and it took {} seconds'.format(round(f1_forest,4),round(delta1,4)))
```

    Random forest with 10 trees using all components: f1 score: 0.9263 and it took 25.2593 seconds


**Random forest performed better and faster than SVM**. Therefore, it is the best model for this study.

Maybe the fact that SVM performed worse than Random forest is the difficulty of classifying points basend on the distance/margin.

After choosing Random Forest, **let's try to improve the model even more by adding the two variables - Time and Amount and/or using 20 PCA components**.


```python
### Random Forest - best number of trees -> 10
### Trying with 20 components

forest = RandomForestClassifier(n_estimators=10,criterion = 'gini', max_features='sqrt',random_state=47)
t0 = time.time()
forest.fit(x_train_20,y_train)
pred=forest.predict(x_dev_20)
f1_forest_20 = f1_score(y_dev,pred)
t1=time.time()
delta2=t1-t0

print('Random forest with 10 trees and using the first 20 components - f1 score: {} and it took {} seconds'.format(round(f1_forest_20,4),round(delta2,4)))
```

    Random forest with 10 trees and using the first 20 components - f1 score: 0.913 and it took 21.3366 seconds



```python
#### Adding two features for the Random Forest

forest = RandomForestClassifier(n_estimators=10,criterion = 'gini', max_features='sqrt',random_state=47)
t0 = time.time()
forest.fit(x_train_full,y_train)
pred=forest.predict(x_dev_full)
f1_forest_20 = f1_score(y_dev,pred)
t1=time.time()
delta2=t1-t0

print('Random forest with 10 trees and using the all components - f1 score: {} and it took {} seconds'.format(round(f1_forest_20,4),round(delta2,4)))
```

    Random forest with 10 trees and using the all components - f1 score: 0.9263 and it took 42.6033 seconds


Both options did not improve the model. Therefore, I am going to use _Random Forest with all PCA features_.

### Test set

Then, let's see how it performs on test set!


```python
### Final score - Using only the PCA components and random forest
forest = RandomForestClassifier(n_estimators=10,criterion = 'gini', max_features='sqrt',random_state=47)

t0 = time.time()
forest.fit(x_train,y_train)
pred=forest.predict(x_test)
f1_forest = f1_score(y_test,pred)
t1=time.time()
delta2=t1-t0

print('Random forest - f1 score: {} and it took {} seconds'.format(round(f1_forest,4),round(delta2,4)))
```

    Random forest - f1 score: 0.8444 and it took 38.6097 seconds


### Conclusion

It is amazing to see this usage of data to prevent fraudulent transactions or any other application. 

The numbers that could be disconnected at first glance or "just numbers", actually, they were used for a greater good. Extracting information from data and seeing a trend behind numbers is what I find exciting about data science.

Here, the random forest model can be used to help the detection of fraudulent transactions.


```python

```
