# Table of Contents

- [[#Kaggle Intro to Machine Learning Tutorial|Kaggle Intro to Machine Learning Tutorial]]
	- [[#Kaggle Intro to Machine Learning Tutorial#Glossary of Terms|Glossary of Terms]]
	- [[#Kaggle Intro to Machine Learning Tutorial#Intro to how Models Work|Intro to how Models Work]]
	- [[#Kaggle Intro to Machine Learning Tutorial#Building your model|Building your model]]
	- [[#Kaggle Intro to Machine Learning Tutorial#Model Validation|Model Validation]]
	- [[#Kaggle Intro to Machine Learning Tutorial#Underfitting and Overfitting|Underfitting and Overfitting]]
	- [[#Kaggle Intro to Machine Learning Tutorial#Random Forests|Random Forests]]
- [[#Kaggle Intermediate Machine Learning Tutorial|Kaggle Intermediate Machine Learning Tutorial]]
	- [[#Kaggle Intermediate Machine Learning Tutorial#Review and Setting up a basic model|Review and Setting up a basic model]]
	- [[#Kaggle Intermediate Machine Learning Tutorial#Dealing with Missing Values|Dealing with Missing Values]]
- [[#Useful links|Useful links]]
	- [[#Useful links#Intro Links|Intro Links]]
		- [[#Intro Links#Pandas Documentation|Pandas Documentation]]
		- [[#Intro Links#Skikit Learn|Skikit Learn]]
		- [[#Intro Links#Andrej Karpathy Neural Network Playlist|Andrej Karpathy Neural Network Playlist]]
		- [[#Intro Links#3B1B Neural Network|3B1B Neural Network]]
		- [[#Intro Links#Neural Network Intro Textbook|Neural Network Intro Textbook]]
	- [[#Useful links#Models|Models]]
		- [[#Models#Decision Tree Regressor|Decision Tree Regressor]]
		- [[#Models#Random Forest Regressor|Random Forest Regressor]]



## Kaggle Intro to Machine Learning Tutorial
### Glossary of Terms

- **Fitting / Training**: The process of capturing patterns in data
- **Training Data**: Data used to fit the model
- **Leaf**: The point in the decision tree when a decision is made
- **Underfitting**: Where a model does poorly in both training data and validation (including other new data) due to too few parameters. Theres a lot of data in each parameter, but that leads to consistently off predictions **(failing to capture relevant patterns, again leading to less accurate predictions)**
- **Overfitting**: Where a model matches the training data almost perfectly, but does poorly in validation and other new data due to too many parameters and too few data points within each parameter **(capturing spurious patterns that won't recur in the future, leading to less accurate predictions)**
### Intro to how Models Work

For simplicity, we'll start with the simplest possible decision tree.
![First Decision Trees](https://storage.googleapis.com/kaggle-media/learn/images/7tsb5b1.png)

It divides houses into only two categories. The predicted price for any house under consideration is the historical average price of houses in the same category.

We use data to decide how to break the houses into two groups, and then again to determine the predicted price in each group. This step of capturing patterns from data is called **fitting** or **training** the model. The data used to **fit** the model is called the **training data**.

The details of how the model is fit (e.g. how to split up the data) is complex enough that we will save it for later. After the model has been fit, you can apply it to new data to **predict** prices of additional homes.

To add more precision and accuracy to the model we can add more parameters

We use data to decide how to break the houses into two groups, and then again to determine the predicted price in each group. This step of capturing patterns from data is called **fitting** or **training** the model. The data used to **fit** the model is called the **training data**.

### Building your model

The steps to building and using a model are:
- **Define:** What type of model will it be? A decision tree? Some other type of model? Some other parameters of the model type are specified too.
- **Fit:** Capture patterns from provided data. This is the heart of modeling.
- **Predict:** Just what it sounds like
- **Evaluate**: Determine how accurate the model's predictions are.

```python title:Intro
from sklearn.tree import DecisionTreeRegressor

# Define model. Specify a number for random_state to ensure same results each
# run
melbourne_model = DecisionTreeRegressor(random_state=1)

# Fit model
melbourne_model.fit(X, Y)

# Make predictions
melbourne_model.predict(X)
```

Where X and Y represent the following:
X - Features (the things used to estimate)
Y - Prediction target (the thing to predict)

Many machine learning models allow some randomness in model training. Specifying a number for `random_state` ensures you get the same results in each run (it determines the seed of the random number generator). This is considered a good practice. You use any number, and model quality won't depend meaningfully on exactly what value you choose. 

We now have a fitted model that we can use to make predictions.

We get the same result if we apply the test to the training data since it's just going down the same decision tree
### Model Validation

The scikit-learn library has a function `train_test_split` to break up the data into two pieces. We'll use some of that data as training data to fit the model, and we'll use the other data as validation data to calculate `mean_absolute_error`.

```python title:splitting_data
from sklearn.model_selection import train_test_split

# split data into training and validation data, for both features and target
# The split is based on a random number generator. Supplying a numeric value to
# the random_state argument guarantees we get the same split every time we
# run this script.
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)
# Define model
melbourne_model = DecisionTreeRegressor()
# Fit model
melbourne_model.fit(train_X, train_y)

# get predicted prices on validation data
val_predictions = melbourne_model.predict(val_X)
print(mean_absolute_error(val_y, val_predictions))
```

Just read the data, as it allows us to basically **"measure"** the accuracy of the model. 

### Underfitting and Overfitting

You can see in scikit-learn's [documentation](http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html) that the decision tree model has many options (more than you'll want or need for a long time). The most important options determine the tree's depth. Recall from [the first lesson in this course](https://www.kaggle.com/dansbecker/how-models-work) that a tree's depth is a measure of how many splits it makes before coming to a prediction. This is a relatively shallow tree

**Overfitting**: When we divide the houses amongst many leaves, we also have fewer houses in each leaf. Leaves with very few houses will make predictions that are quite close to those homes' actual values, but they may make very unreliable predictions for new data (because each prediction is based on only a few houses). This is overfitting Where a model matches the training data almost perfectly, but does poorly in validation and other new data.

**Underfitting**: At another extreme, if a tree divides houses into only 2 or 4, each group still has a wide variety of houses. Resulting predictions may be far off for most houses, even in the training data (and it will be bad in validation too for the same reason). When a model fails to capture important distinctions and patterns in the data, so it performs poorly even in training data, that is called **underfitting**.
linkcode

![underfitting_overfitting](https://storage.googleapis.com/kaggle-media/learn/images/AXSEOfI.png)
Let's define a function which allows us to experiment to try and optimize the max-tree depth:
```python title:function_to_test_different_tree_depths
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor

def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes,
								random_state=0)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return(mae)
    
# And then we can use a for loop to compare the values:
# compare MAE with differing values of max_leaf_nodes
for max_leaf_nodes in [5, 50, 500, 5000]:
    my_mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
    print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %d" %
    (max_leaf_nodes, my_mae))

## The above loop simply sets max_leaf_nodes to 5, 50, 500, and 5000 respectively. Calculates the mean-absolute-error, and prints it out
```

### Random Forests

*In order to combat overfitting and underfitting, people have come across some creative strategies:*

One of them is random forest, which uses many trees, and it makes a prediction by averaging the predictions of each component tree. It generally has much better predictive accuracy than a single decision tree and it works well with default parameters. If you keep modeling, you can learn more models with even better performance, but many of those are sensitive to getting the right parameters.

Assuming we've done the data-organization (Getting train_X, train_y, val_X, val_y)
```python title:tree_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

forest_model = RandomForestRegressor(random_state=1)
forest_model.fit(train_X, train_y)
melb_preds = forest_model.predict(val_X)
print(mean_absolute_error(val_y, melb_preds))
```

## Kaggle Intermediate Machine Learning Tutorial

### Review and Setting up a basic model

```python title:Review_And_Doing
# Importing everything as necessary
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Setting up the dataframe and making partitions
X_full = pd.read_csv('../input/train.csv', index_col='Id')
X_test_full = pd.read_csv('../input/test.csv', index_col='Id')

# Obtain target and predictors
y = X_full.SalePrice
features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
X = X_full[features].copy()
X_test = X_test_full[features].copy()
# the .copy() makes a copy instead of modifying the original list

# Break off validation set from training data
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0) # Splitting into 80-20 buckets

## Making 5 different model variations on the random tree to find the most
## Optimal settings
model_1 = RandomForestRegressor(n_estimators=50, random_state=0)
model_2 = RandomForestRegressor(n_estimators=100, random_state=0)
model_3 = RandomForestRegressor(n_estimators=100, criterion='absolute_error', random_state=0)
model_4 = RandomForestRegressor(n_estimators=200, min_samples_split=20, random_state=0)
model_5 = RandomForestRegressor(n_estimators=100, max_depth=7, random_state=0)

models = [model_1, model_2, model_3, model_4, model_5]

# Function for comparing different models
def score_model(model, X_t=X_train, X_v=X_valid, y_t=y_train, y_v=y_valid):
    model.fit(X_t, y_t)
    preds = model.predict(X_v)
    return mean_absolute_error(y_v, preds)

for i in range(0, len(models)):
    mae = score_model(models[i])
    print("Model %d MAE: %d" % (i+1, mae))
```
***RandomForestRegressor parameters***

**n_estimators**:
How many independent trees the forest averages over.

The model prediction is:

$$\hat{y}(x) = \frac{1}{T} \sum_{t=1}^T h_t(x)$$
Variance decreases approximately as:
$$\mathrm{Var}(\hat{y}) \approx \frac{\sigma^2}{T}$$
(assuming trees are uncorrelated — in practice they are partially correlated).
*Conceptually*
- More trees → lower variance, more stable predictions.  
- Too few trees → noisy ensemble.  
- Diminishing returns after ~100–300 trees in many problems.

**random_state**: 
Seed to make the random decisions

**criterion**:
The function to measure the quality of a split
*Common options*: `"squared_error"`, `"absolute_error"`

*Mathematically*  
- *Squared error* impurity for a node \(S\):
  
  $$\text{Impurity}(S) = \sum_{i \in S} (y_i - \bar{y}_S)^2$$
	Optimizes toward the **mean** of targets in a node.

- *Absolute error* impurity:
  $$\text{Impurity}(S) = \sum_{i \in S} |y_i - \text{median}(S)|$$
	Optimizes toward the **median** of targets in a node.

*Conceptually*
- `"squared_error"` is **sensitive to outliers** (trees will try to reduce large squared residuals).  
- `"absolute_error"` is **more robust** to outliers (focuses on median behavior).  
- Changing the criterion changes *which splits* are considered best.

**min_samples_split**:
The minimum number of samples required to split an internal node

*Mathematically*  
If node size $|N| < \text{min\_samples\_split}$, the node becomes a leaf and is not split.

*Conceptually*
- Larger values → **fatter leaves**, fewer splits → **higher bias, lower variance**.  
- Smaller values (e.g., 2) → allow very deep/small leaves → can overfit noisy data.  
- Helps regularize trees by preventing splits on tiny sample subsets.

*Best Practice*
Generally speaking, we like to use 
$$\sqrt{\# \text{ features}}$$

**max_depth**: 
The maximum depth of the tree. If ``None``, then nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples.

*Mathematically*  
A full binary tree of depth \(d\) can have up to:
$$2^d \text{ leaves}$$
Leaves correspond to constant prediction regions; deeper trees allow more complex piecewise-constant functions.

*Conceptually*
- **Large or None** → very flexible trees, low bias, high variance (risk overfitting).  
- **Small** → simpler trees, higher bias, lower variance (smoother predictions).  
- Example: `max_depth=7` → at most \(2^7 = 128\) leaves (much simpler than unconstrained trees).

### Dealing with Missing Values

It makes a lot of intuitive sense regarding why we would not want to have missing values in the dataset, and therefore it's important to understand how to deal with them as they show up. There's different strategies to go about it, and this section elaborates on the different techniques which are used to deal with such instances and their perks and pitfalls.

#### The 3 Approaches
##### 1. Dropping Columns
The simplest option is to drop the column with any missing values. However, as anticipated this leads to extensive data loss. In case where there's a lot of/a majority of a column is missing, this approach may make sense, but imaging completely eliminating a potential feature over one missing value.

```python
# Get names of columns with missing values
cols_with_missing = [col for col in X_train.columns
                     if X_train[col].isnull().any()]

# Drop columns in training and validation data
reduced_X_train = X_train.drop(cols_with_missing, axis=1)
reduced_X_valid = X_valid.drop(cols_with_missing, axis=1)

print("MAE from Approach 1 (Drop columns with missing values):")
print(score_dataset(reduced_X_train, reduced_X_valid, y_train, y_valid))
```
##### 2. Imputation
Imputation fills in the missing values with some number. For instance, we can fill in the mean value along each column. The imputed value won't be exactly right in most cases, but it usually leads to more accurate models than you would get from dropping the column entirely.

```python
from sklearn.impute import SimpleImputer

# Imputation
my_imputer = SimpleImputer()
imputed_X_train = pd.DataFrame(my_imputer.fit_transform(X_train))
imputed_X_valid = pd.DataFrame(my_imputer.transform(X_valid))

# Imputation removed column names; put them back
imputed_X_train.columns = X_train.columns
imputed_X_valid.columns = X_valid.columns

print("MAE from Approach 2 (Imputation):")
print(score_dataset(imputed_X_train, imputed_X_valid, y_train, y_valid))
```
##### 2. Imputation++
Imputation is the standard approach, and it usually works well. However, imputed values may be systematically above or below their actual values (which weren't collected in the dataset). Or rows with missing values may be unique in some other way. In that case, your model would make better predictions by considering which values were originally missing.

This involves making another column which keeps track of wether the value was initially missing from the dataset or not.

```python
# Make copy to avoid changing original data (when imputing)
X_train_plus = X_train.copy()
X_valid_plus = X_valid.copy()

# Make new columns indicating what will be imputed
for col in cols_with_missing:
    X_train_plus[col + '_was_missing'] = X_train_plus[col].isnull()
    X_valid_plus[col + '_was_missing'] = X_valid_plus[col].isnull()

# Imputation
my_imputer = SimpleImputer()
imputed_X_train_plus = pd.DataFrame(my_imputer.fit_transform(X_train_plus))
imputed_X_valid_plus = pd.DataFrame(my_imputer.transform(X_valid_plus))

# Imputation removed column names; put them back
imputed_X_train_plus.columns = X_train_plus.columns
imputed_X_valid_plus.columns = X_valid_plus.columns
```

### Dealing with Categorical Variables

A **categorical variable** takes only a limited number of values.

- Consider a survey that asks how often you eat breakfast and provides four options: "Never", "Rarely", "Most days", or "Every day". In this case, the data is categorical, because responses fall into a fixed set of categories.
- If people responded to a survey about which what brand of car they owned, the responses would fall into categories like "Honda", "Toyota", and "Ford". In this case, the data is also categorical.

You will get an error if you try to plug these variables into most machine learning models in Python without preprocessing them first. In this tutorial, we'll compare three approaches that you can use to prepare your categorical data.

#### The 3 Approaches
##### 1. Dropping the Categorical Variables

Don't do this, just don't. But if need be. Just check which columns have non-numeric values and completely remove them
##### 2. Ordinal Coding

Assigning each unique value to an integer. e.g. if we were to ask people if they have breakfast:
This approach assumes an ordering of the categories: "Never" (0) < "Rarely" (1) < "Most days" (2) < "Every day" (3).
This assumption makes sense in this example, because there is an indisputable ranking to the categories. Not all categorical variables have a clear ordering in the values, but we refer to those that do as **ordinal variables**. For tree-based models (like decision trees and random forests), you can expect ordinal encoding to work well with ordinal variables.
##### 3. One-Hot Encoding

**One-hot encoding** creates new columns indicating the presence (or absence) of each possible value in the original data.

![tut3_onehot](https://storage.googleapis.com/kaggle-media/learn/images/TW5m0aJ.png)

In the original dataset, "Color" is a categorical variable with three categories: "Red", "Yellow", and "Green". The corresponding one-hot encoding contains one column for each possible value, and one row for each row in the original dataset. Wherever the original value was "Red", we put a 1 in the "Red" column; if the original value was "Yellow", we put a 1 in the "Yellow" column, and so on.

In contrast to ordinal encoding, one-hot encoding _does not_ assume an ordering of the categories. Thus, you can expect this approach to work particularly well if there is no clear ordering in the categorical data (e.g., "Red" is neither _more_ nor _less_ than "Yellow"). We refer to categorical variables without an intrinsic ranking as **nominal variables**.

**Probably won't use it for categorical variables taking a large number of options (n > 15)**


## Useful links

### Intro Links
#### Pandas Documentation 

https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.dropna.html
This is the Data manipulation Library which I will be using to play around with data in Python
#### Skikit Learn 
https://scikit-learn.org/stable/index.html
This is the into Python machine learning library to learn the basics of Machine Learning and AI

#### Andrej Karpathy Neural Network Playlist
https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ

#### 3B1B Neural Network
https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi

#### Neural Network Intro Textbook
http://neuralnetworksanddeeplearning.com
### Models

#### Decision Tree Regressor

**Decision Tree Regressor Explained**
https://towardsdatascience.com/decision-tree-regressor-explained-a-visual-guide-with-code-examples-fbd2836c3bef/

**Understanding Decision Tree Regressor**
https://farshadabdulazeez.medium.com/understanding-decision-tree-regressor-an-in-depth-intuition-a1d3af182efd
#### Random Forest Regressor

A Random Forest Regressor is a learning method used for regression tasks, predicting continuous numerical values. It operates by constructing a multitude of decision trees during training and then averaging their individual predictions to produce a more robust and accurate final prediction.

**Random Forest Regression | Towards Data Science**
https://towardsdatascience.com/random-forest-regression-5f605132d19d/

