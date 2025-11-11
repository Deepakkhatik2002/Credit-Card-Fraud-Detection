# Credit-Card-Fraud-Detection-Using-Machine-Learning

## ABSTRACT

Credit card fraud is a significant problem, with billions of dollars lost each year. Machine learning can be used to detect credit card fraud by identifying patterns that are indicative of fraudulent transactions. This project develops a machine-learning model to detect credit card fraud using a **Random Forest Classifier**. The model is trained on a dataset of historical credit card transactions and evaluated on a holdout dataset of unseen transactions, assessing its performance with metrics like accuracy, precision, recall, and F1-score.

**Keywords:** Credit Card Fraud Detection, Fraud Detection, Fraudulent Transactions, Machine Learning, **Random Forest Classifier**, Imbalanced Data.

---

## Overview

With the increase of people using credit cards in their daily lives, credit card companies should take special care of the security and safety of the customers. According to (Credit card statistics 2021), the number of people using credit cards worldwide was 2.8 billion in 2019; also, 70% of those users own a single card. Reports of credit card fraud in the U.S. rose by 44.7% in 2020.

There are two kinds of credit card fraud. The first is having a credit card account opened under your name by an identity thief. Reports of this fraudulent behavior increased 48% in 2020. The second type is when an identity thief uses an existing account you created, usually by stealing the information on the credit card. Reports on this type of Fraud increased 9% in 2020 (Daly, 2021). Those statistics caught our attention as the numbers have increased drastically and rapidly throughout the years, which motivated us to resolve the issue analytically by using a machine learning model to detect fraudulent credit card transactions.

---

## Project Goal

The main aim of this project is the detection of fraudulent credit card transactions, as it is essential to figure out the fraudulent transactions so that customers do not get charged for the purchase of products that they did not buy. This will be achieved by building and evaluating a Machine Learning model. The project explores the dataset, preprocesses the data, trains a classifier, and evaluates its performance using various metrics.

---

## Data Source

The dataset was retrieved from an open-source website, Kaggle.com. It contains data on transactions made in 2013 by European credit card users in two days only.

* **Dataset:** [Kaggle Credit Card Fraud Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
* **Rows:** 284,808
* **Attributes:** 31
* **Details:**
    * Due to confidentiality, 28 of the attributes (V1-V28) are the result of a PCA transformation.
    * The remaining attributes are **'Time'**, **'Amount'**, and the target variable **'Class'**.
    * The 'Class' attribute is binary, where **`1`** indicates a fraudulent transaction and **`0`** indicates a valid (normal) transaction.
* **Key Finding:** The dataset is **highly imbalanced**.
    * **Valid Transactions (Class 0):** 284,315
    * **Fraud Cases (Class 1):** 492

---

## Project Workflow

1.  **Import Libraries:** Imported `numpy`, `pandas`, `matplotlib`, `seaborn`, and `sklearn`.
2.  **Load Data:** Loaded the `creditcard.csv` file into a pandas DataFrame.
3.  **Exploratory Data Analysis (EDA):**
    * Analyzed statistical summaries of the data using `.describe()`.
    * Investigated the class imbalance.
    * Compared the 'Amount' statistics for both fraudulent and valid transactions.
    * Plotted a correlation matrix (`sns.heatmap`) to visualize relationships between features.
4.  **Data Preprocessing:**
    * Separated the features (`X`) from the target variable (`Y`).
    * Split the data into training and testing sets using an 80/20 split (`test_size=0.2`).
5.  **Model Training:**
    * The project implements a **Random Forest Classifier** (`sklearn.ensemble.RandomForestClassifier`).
    * The model was `fit` on the training data (`xTrain`, `yTrain`).
6.  **Model Evaluation:**
    * The trained model was used to make predictions (`yPred`) on the test set (`xTest`).
    * Performance was evaluated using a confusion matrix and standard classification metrics.

---

## Algorithm Used

* **Random Forest Classifier:** An ensemble learning method that operates by constructing a multitude of decision trees at training time. For classification tasks, the output of the random forest is the class selected by the most trees. It is robust and performs well on imbalanced datasets.

---

## Results

The Random Forest model was evaluated on the unseen test set, yielding the following performance metrics:

* **Accuracy:** 0.99956
* **Precision:** 0.97402
* **Recall:** 0.76530
* **F1-Score:** 0.85714
* **Matthews Correlation Coefficient (MCC):** 0.86318

A confusion matrix was also generated to visualize the model's performance in distinguishing between 'Normal' and 'Fraud' classes.


---

## Conclusion

This project successfully implemented a **Random Forest Classifier** to detect credit card fraud on a highly imbalanced dataset. The model achieved a very high accuracy (99.956%), which is expected given the dataset's nature.

More importantly, the model showed:
* **High Precision (97.4%):** This indicates that when the model flags a transaction as fraud, it is correct 97.4% of the time. This is excellent for minimizing false positives (i.e., not bothering valid customers).
* **Good Recall (76.5%):** This means the model successfully identified and "caught" 76.5% of all actual fraudulent transactions in the test set.

The strong F1-Score (85.7%) and MCC (86.3%) confirm that the model is robust and provides a good balance between precision and recall, making it effective for this classification task.

---

## Future Work

There are many ways to improve the model, such as:
* Using different data sampling techniques (like SMOTE or ADASYN) to handle the imbalanced data before training.
* Applying the model to different datasets with various sizes and data types.
* Experimenting with different data splitting ratios.
* Tuning the Random Forest's hyperparameters (e.g., `n_estimators`, `max_depth`).
* Exploring other algorithms (like Gradient Boosting, e.g., XGBoost, or Logistic Regression) and comparing their performance.
* Merging telecom data to calculate the location of people to have better knowledge of the location of the card owner while his/her credit card is being used; this will ease the detection because if the card owner is in Dubai and a transaction of his card was made in Abu Dhabi, it will easily be detected as Fraud.
