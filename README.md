# House Price Prediction Project with TensorFlow

This Python project focuses on predicting house prices based on various features using TensorFlow. The dataset contains information about house sales in King County, including features like the number of bedrooms, square footage, and location. The aim of this project is to create a neural network in order to predict a house price based on the given features. The primary steps of the project include exploratory data analysis (EDA), feature engineering, model building, evaluation, and improvement.

## Libraries
pandas
numpy
matplotlib
seaborn
tensorflow
scikit-learn

## Getting Started
Follow these steps:

1. Clone the Repository: Clone this GitHub repository to your local machine.

2. Install Dependencies: Ensure you have the required Python libraries installed. You can install them using the following command:
``````
pip install pandas numpy matplotlib seaborn tensorflow scikit-learn
``````
For Jupyter notebooks :
``````
!pip install pandas numpy matplotlib seaborn tensorflow scikit-learn
``````
3. Data File: A .csv file is stored in the /data folder. The dataset used for this project is available on  [Kaggle](https://www.kaggle.com/datasets/harlfoxem/housesalesprediction?select=kc_house_data.csv). 

4. Run the Code and make sure to update file paths if needed.

5. While running the code a model will be saved in the /models folder.


## Data Preparation and Feature Engineering
Data cleaning such as dropping unnecessary columns.
Convert the 'date' column to a datetime object and create new features like 'month' and 'year.'
Scale the numerical features using MinMaxScaler.

## Exploratory Data Analysis (EDA)
Visualize the distribution of the target variable (house prices) and explore relationships between features and the target. Analyze geographical information using scatterplots of latitude and longitude. Remove expensive outliers to improve data visualization.

## Model Building
Create a neural network model using TensorFlow's Sequential API. Define the model architecture with multiple dense layers and an output layer.
Compile the model using the mean squared error (MSE) loss and the Adam optimizer. Fit the model to the training data.

## Model Evaluation
Evaluate the model's performance on the test set using metrics like mean absolute error (MAE), root mean squared error (RMSE), and explained variance score.
Compare model predictions to actual house prices with scatterplots.
## Model Experimentation
Create different neural network models with different hyperparameters and compare the results.
## Results
The project aims to predict house prices based on various features, and the results include insights into the model's performance and its ability to make predictions.

## Model saving, re-loading and re-evaluating
This project also include the re-evaluation of a saved model in order to make sure that all model files were correctly saved in the local machine for later use. 

## Acknowledgments
This project was developed as part of the "Python for Data Science and Machine Learning Bootcamp" course from Udemy, instructed by Jose Portilla and Pierian Data.
