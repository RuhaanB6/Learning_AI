# Table of Contents

- [[#Kaggle Intro to Machine Learning Tutorial|Kaggle Intro to Machine Learning Tutorial]]
	- [[#Kaggle Intro to Machine Learning Tutorial#Glossary of Terms|Glossary of Terms]]
	- [[#Kaggle Intro to Machine Learning Tutorial#Intro to how Models Work|Intro to how Models Work]]
	- [[#Kaggle Intro to Machine Learning Tutorial#Building your model|Building your model]]
	- [[#Kaggle Intro to Machine Learning Tutorial#Model Validation|Model Validation]]
	- [[#Kaggle Intro to Machine Learning Tutorial#Underfitting and Overfitting|Underfitting and Overfitting]]
	- [[#Kaggle Intro to Machine Learning Tutorial#Random Forests|Random Forests]]
- [[#Kaggle Intermediate Machine Learning|Kaggle Intermediate Machine Learning]]
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

## Kaggle Intermediate Machine Learning

## Useful links

### Intro Links
#### Pandas Documentation 
```embed
title: "pandas.DataFrame.dropna — pandas 2.3.3 documentation"
image: "https://pandas.pydata.org/docs/_static/pandas.svg"
description: ""
url: "https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.dropna.html"
favicon: ""
aspectRatio: "40.33333333333333"
```
This is the Data manipulation Library which I will be using to play around with data in Python
#### Skikit Learn 
```embed
title: "scikit-learn: machine learning in Python — scikit-learn 1.7.2 documentation"
image: "https://scikit-learn.org/stable/_images/sphx_glr_plot_classifier_comparison_001_carousel.png"
description: ""
url: "https://scikit-learn.org/stable/index.html"
favicon: ""
aspectRatio: "31.666666666666664"
```
This is the into Python machine learning library to learn the basics of Machine Learning and AI

#### Andrej Karpathy Neural Network Playlist
```embed
title: "Neural Networks: Zero to Hero"
image: "https://i.ytimg.com/vi/VMj-3S1tku0/hqdefault.jpg?sqp=-oaymwEXCOADEI4CSFryq4qpAwkIARUAAIhCGAE=&rs=AOn4CLAsbch9qh_yKE5PEvVrJKLFzLfYYQ&days_since_epoch=20405"
description: ""
url: "https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ"
favicon: ""
aspectRatio: "56.25"
```

#### 3B1B Neural Network
```embed
title: "Neural networks"
image: "https://i9.ytimg.com/s_p/PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi/landscape_mqdefault.jpg?sqp=CJCd2MgGir7X7AMICNGyiNsFEAE=&rs=AOn4CLDgNnpg5OvoeywdojFDeFM5Bgy5Ew&v=1533155665&days_since_epoch=20405"
description: "Learn the basics of neural networks and backpropagation, one of the most important algorithms for the modern world."
url: "https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi"
favicon: ""
aspectRatio: "56.25"
```

#### Neural Network Intro Textbook
```embed
title: "Neural networks and deep learning"
image: "http://neuralnetworksanddeeplearning.com/images/arrow.png"
description: ""
url: "http://neuralnetworksanddeeplearning.com"
favicon: ""
aspectRatio: "91.45299145299145"
```

### Models

#### Decision Tree Regressor

```embed
title: "Decision Tree Regressor, Explained: A Visual Guide with Code Examples | Towards Data Science"
image: "https://towardsdatascience.com/wp-content/uploads/2024/10/1qTpdMoaZClu-KDV3nrZDMQ.png"
description: "Trimming branches smartly with Cost-Complexity Pruning"
url: "https://towardsdatascience.com/decision-tree-regressor-explained-a-visual-guide-with-code-examples-fbd2836c3bef/"
favicon: ""
aspectRatio: "52.77777777777778"
```

```embed
title: "Understanding Decision Tree Regressor: An In-Depth Intuition"
image: "https://miro.medium.com/v2/resize:fit:1024/1*bSJ1fhsnbxzyrhARn7qgKw.png"
description: "Decision Tree Regression: Predictive branching for non-linear relationships"
url: "https://farshadabdulazeez.medium.com/understanding-decision-tree-regressor-an-in-depth-intuition-a1d3af182efd"
favicon: ""
aspectRatio: "56.25"
```

#### Random Forest Regressor

A Random Forest Regressor is a learning method used for regression tasks, predicting continuous numerical values. It operates by constructing a multitude of decision trees during training and then averaging their individual predictions to produce a more robust and accurate final prediction.

```embed
title: "Random Forest Regression | Towards Data Science"
image: "https://towardsdatascience.com/wp-content/uploads/2022/03/1TegTeL2BoLb6TQvUXnWKKQ-scaled.jpeg"
description: "A basic explanation and use case in 7 minutes"
url: "https://towardsdatascience.com/random-forest-regression-5f605132d19d/"
favicon: ""
aspectRatio: "150.9433962264151"
```

