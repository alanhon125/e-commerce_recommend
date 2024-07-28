# e-commerce purchase item recommendation
a simple e-commerce recommendation system powered with AI (Neural Collaborative Filtering (NCF))

Before you start, please download the .ipynb, dataset from Kaggle and unzip the file from below link (P.S.: please keep all files in the same directory):

https://www.kaggle.com/datasets/mkechinov/ecommerce-purchase-history-from-electronics-store/versions/1/data

This is a dataset of purchase from April 2020 to November 2020 from a large home appliances and electronics online store.

Each row in the file represents an event. All events are related to products and users. Each event is like many-to-many relation between products and users.

# Prerequisite
we are using the LibRecommender library in Python to train an NCF model for the purchase dataset on item recommendation. Please install:
- Python>=3.9
- LibRecommender==1.5.1
- numpy==1.26.4
- pandas==2.2.2
- scikit-learn==1.5.1
- tensorflow==2.14.0
- torch==2.4.0

# Data cleaning
The dataset may contain duplicate record of purchase and with none value. We need to drop duplicates and NaN values before training.

Also, NCF requires rating value to make prediction. We need to compute the rating based on the frequency of purchase by user and category. The higher rating implies higher frequency and preference to purchase item in that category.

At the same time, we create a dictionary to map the product id into category-brand name for better understanding of the product type.

Finally we keep only 4 columns (["user_id", "product_id", "rating", "event_time"]) in the training and testing dataset

# Dataset preparation for training and testing

- Split the dataset into training, validation, and test data with ratio 80%, 10%, 10% respectively
- Convert the pandas dataframe into a compatible datatype for LibRecommender
- Since we are not using any other feature other than the interaction between the user and an item. Thus, the DatasetPure function builds the datasets from a Pure Collaborative Filtering perspective
- Configure NCF for rating task (Rating is usually used when we have a dataset around explicit feedback (direct rating, starts given by customers), but the dataset doesn't provide customer's rating and so we compute the frequency of purchase by category as rating instead)

# Training

Fit NCF model with training dataset and evaluate with testing dataset
Once training is done, you can test the model for various scenarios, e.g. 
- input an user_id and a product_id to predict preference of user to the specific product
- input an user_id and the number of items that to be recommended to make a list of recommendation
