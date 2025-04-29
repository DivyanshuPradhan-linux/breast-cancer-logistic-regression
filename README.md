# breast-cancer-logistic-regression
Classification with Logistic Regression

## Objective

The objective of this task was to build a binary classifier using logistic regression. This involves understanding the fundamental concepts of logistic regression, handling a binary classification dataset, evaluating the model's performance using various metrics, and understanding the role of the sigmoid function and threshold tuning.

## Tools Used

* **Scikit-learn:** For implementing the logistic regression model, data splitting, feature scaling, and evaluation metrics.
* **Pandas:** For data loading and manipulation.
* **Matplotlib:** For data visualization, specifically for plotting the ROC curve and the impact of threshold tuning.
* **NumPy:** For numerical operations, particularly for handling arrays and linspace for threshold tuning visualization.

## Dataset

The dataset used for this task is the **Breast Cancer Wisconsin (Diagnostic) Dataset**. This dataset is a standard binary classification dataset where the goal is to predict whether a tumor is malignant or benign based on various features computed from digitized images of fine needle aspirates (FNA) of breast masses.

The dataset was obtained from the link provided in the task description[cite: 5].

*Note: Ensure the dataset file (e.g., `wdbc.data`) is in the same directory as the Python script, or update the file path in the code.*

## How to Run the Code

1.  **Prerequisites:** Make sure you have Python and the necessary libraries installed. You can install the libraries using pip:
    ```bash
    pip install pandas scikit-learn matplotlib numpy
    ```
2.  **Download the Dataset:** Download the Breast Cancer Wisconsin dataset using the link provided in the task document[cite: 5]. Save the file (e.g., `wdbc.data`) in the same directory as your Python script.
3.  **Run the Python Script:** Execute the provided Python script (`main.py` or whatever you named it) from your terminal:
    ```bash
    python main.py
    ```

The script will load the data, split it into training and testing sets, standardize the features, train the logistic regression model, evaluate its performance, and display the confusion matrix, precision, recall, and ROC-AUC score. It will also display plots for the ROC curve and the precision/recall trade-off with threshold tuning.

## Task Breakdown and Implementation

The task was implemented following the steps outlined in the mini-guide[cite: 3]:

1.  **Choose a binary classification dataset:** The Breast Cancer Wisconsin dataset was selected.
2.  **Train/test split and standardize features:** The dataset was split into training and testing sets (75% train, 25% test), and features were standardized using `StandardScaler` to ensure they have zero mean and unit variance, which is important for many machine learning algorithms including logistic regression.
3.  **Fit a Logistic Regression model:** A `LogisticRegression` model from scikit-learn was initialized and trained on the scaled training data.
4.  **Evaluate with confusion matrix, precision, recall, ROC-AUC:** The model's performance was evaluated on the test set using the following metrics:
    * **Confusion Matrix:** A table that summarizes the performance of a classification algorithm[cite: 7, 5].
    * **Precision:** The ratio of correctly predicted positive observations to the total predicted positives[cite: 7, 5].
    * **Recall (Sensitivity):** The ratio of correctly predicted positive observations to the all observations in the actual class[cite: 7, 5].
    * **ROC-AUC Curve:** The Receiver Operating Characteristic (ROC) curve is a plot showing the performance of a classification model at all classification thresholds. The Area Under the Curve (AUC) measures the entire area underneath the entire ROC curve[cite: 7, 5].
5.  **Tune threshold and explain sigmoid function:** The code demonstrates how precision and recall change with different probability thresholds. An explanation of the sigmoid function and its role in logistic regression is also included.

## Key Concepts and Interview Questions Addressed

This task provided hands-on experience with several key machine learning concepts relevant to the interview questions mentioned in the task document[cite: 6]:

* **Logistic Regression vs. Linear Regression:** Understanding how logistic regression is used for classification (predicting a categorical outcome) compared to linear regression which is used for regression (predicting a continuous outcome)[cite: 6]. Logistic regression uses the sigmoid function to output a probability.
* **The Sigmoid Function:** Understanding the mathematical form and purpose of the sigmoid function in mapping linear outputs to probabilities between 0 and 1[cite: 6].
* **Precision vs. Recall:** Differentiating between precision and recall and understanding when each metric is more important depending on the problem context[cite: 6].
* **ROC-AUC Curve:** Interpreting the ROC curve and the significance of the AUC score as a measure of the model's ability to distinguish between classes[cite: 6].
* **Confusion Matrix:** Understanding the components of a confusion matrix (True Positives, True Negatives, False Positives, False Negatives) and how to derive other evaluation metrics from it[cite: 6].
* **Handling Imbalanced Classes:** Although not explicitly implemented in the basic code, the task prompts consideration of what happens when classes are imbalanced and potential strategies to address it[cite: 8, 6].
* **Choosing the Threshold:** Understanding that the default threshold of 0.5 can be adjusted based on the desired trade-off between precision and recall[cite: 8, 6].
* **Multi-class Logistic Regression:** Considering whether logistic regression can be extended to handle problems with more than two classes[cite: 9, 6]. (Yes, through techniques like One-vs-Rest or Multinomial Logistic Regression).

This project fulfills the requirements of Task 4 by demonstrating the implementation and evaluation of a binary logistic regression classifier.
