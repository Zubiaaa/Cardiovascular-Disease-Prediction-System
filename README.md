# Cardiovascular-Disease-Prediction

## Dataset
Source of data: https://archive.ics.uci.edu/dataset/45/heart+disease


License of data: This dataset is licensed under a Creative Commons Attribution 4.0 International (CC BY 4.0) license.


Dataset Description: The heart disease data was collected in a previous study by conducting experiments on patients through angiography [1]. This dataset is multivariate in nature that contains 920 records and 14 attributes. All attributes are numerical, which includes binary, ordinal, discrete, and continuous variables.

## Machine Learning Models Description:

### Overview of boosting ensemble learning model:
The boosting model is a meta model that makes strong predictions by learning from the incorrect predictions made by base models by assigning weights. The AdaBoost classifier (a pre-existing function in Python) is used as a boosting ensemble learning technique. Without the genetic algorithm, 100 decision tree classifiers are used as the base models, and 50 decision tree classifiers are used after applying the genetic algorithm. The train set is split into several subsets to train each base model separately. Predictions are made on the test set. Each base model gets notified by the last base model about the incorrect predictions it made. This results in reducing bias and variance by converting weak learners (base models) into strong ones.



### Overview of blending ensemble learning model:
A blending ensemble learning model blends predictions obtained by several machine learning models with different characteristics to build a robust model that overcomes the negative impact of single models and adds up the positive impacts. As the blending ensemble model does not have a pre-existing function in Python, it is built manually. A combination of five base models, including a Support Vector Machine (SVM), a K-Nearest Neighbors (KNN), a Decision Tree (DT) classifier, a Random Forest (RF) classifier, and a Gradient Boosting (GB) classifier, are used to build the input features for the meta model. A meta model, Logistic Regression (LR), is used to make final predictions by taking input from the training and validation sets' predictions.


## Tools used:
The Python (3.10.0) programming language and Jupyter Notebook (6.5.4) tool was used to build this system.


## Intructions on how to run cardiovascular disease prediction system:
Either use "cleaned_heart_disease_dataset.csv" that is already saved in the "dataset" folder or run the notebook file “data_preprocessing.ipynb” to clean the original dataset and save the cleaned dataset. If the former is true, please skip step 1 below and follow the second step only. If the latter is the case, please follow the two steps below.
1. First, run the “data_preprocessing.ipynb” notebook file to clean the original dataset. By running this notebook, the cleaned data will be saved, which will be used in the second notebook called "heart_disease_classification.ipynb." The cleaned dataset called "cleaned_heart_disease_dataset.csv" will be saved to the same folder or location where you are running the “data_preprocessing.ipynb” notebook.
2. Second, run the “heart_disease_classification.ipynb” notebook to execute the genetic algorithm and machine learning classification.

Note: Do not forget to put the correct file locations of the .csv files in both notebook files if you have saved the original or cleaned dataset in a separate location. For example, when running the read_csv() and to_csv() methods. Also, as one cell is connected to another, run them one after the other in order.


## Intructions on how to run boosting ensemble learning model:
1. Firstly, tune the hyperparameters of the model. 
2. Secondly, split the cleaned dataset into train and test sets. 
3. Then conduct data normalisation, followed by creating the boosting ensemble model (AdaBoost classifier). 
4. Next, fit training data into the model and then make predictions. 
5. Then check performance metrics, accuracy, precision, recall, F1-score, and confusion matrix. 
6. All the above steps should be conducted twice, first with the entire cleaned dataset (before the genetic algorithm) and next with the dataset output by the genetic algorithm (after the genetic algorithm). 
7. Lastly, compare the performance of the model before and after applying the genetic algorithm. 
8. Keep in mind that each run of the model might produce different results.


## Intructions on how to run blending ensemble learning model:
1. Firstly, tune the hyperparameters of the model. 
2. Secondly, create two separate classes to build the blending ensemble learning model manually before and after the application of the genetic algorithm. 
3. The classes should contain splitting the cleaned dataset into train, test, and validation sets. 
4. Then conduct data normalisation, followed by defining five base models (SVM, KNN, decision tree, random forest, and gradient boosting) and a meta model (logistic regression). 
5. Next, train the base models with a train set and make predictions using a validation set (validation predictions) and a test set (test predictions). 
6. Then use the validation predictions to train the meta model and test the meta model using test predictions. 
7. To test the meta model, accuracy, precision, recall, F1-score, and confusion matrix should be checked. 
8. Lastly, compare the performance of the model before and after applying the genetic algorithm. 
9. Keep in mind that each run of the model might produce different results.


## Required Packages: 
This section includes the names of packages that are required to install to run “data_preprocessing.ipynb” and “heart_disease_classification.ipynb” notebook files. 
1. Pandas 
2. NumPy 
3. Statistics 
4. SciPy 
5. Matplotlib
6. Seaborn
7. Statsmodels
8. Scikit-learn
When installing these packages, make sure all their dependencies are also installed.







[1] R. Detrano, A. Janosi, W. Steinbrunn, M. Pfisterer, J. J. Schmid, S. Sandhu, K. H. Guppy, S. Lee and V. Froelicher, “International application of a new probability algorithm for the diagnosis of coronary artery disease” The American journal of cardiology, vol. 64, no. 5, pp. 304-310, Aug. 1989.

