![image](https://github.com/sjaggi1/Breast-Cancer-Prediction/assets/144943702/88fd8568-0b9f-41e0-a944-e9c2698dad75)# Breast-Cancer-Prediction
The system aims to predict the class of breast cancer as Benign or Malignant. This involves using machine learning for making predictions accurately. This project utilizes logistic regression models to predict breast cancer based on key clinical features. 

# System Approach
  # Data Preprocessing:
  In the data preprocessing phase the data was had undergone the steps like loading of the dataset, splitting the dataset into train and test set. 
  The dataset need not require handling missing values because it was mentioned that there is missing data in the dataset in the UCI Machine Learning Repository
  # Machine Learning Algorithm:
  I implemented the Logistic Regression Model which is a Classification algorithm to predict the breast cancer. Also the model showed nearly the same level of accuracy for model such as Random Forest     
  Classification, Support Vector Classification.
  # Deployment:
  The Logistic Regression Model was deployed on a web application made using HTML, CSS, JS and Flask programming was used to deploy the model on the website in order to make the predictions. 
  # Evaluation:
  The models performance was accessed based on its confusion matrix and the accuracy score. The model accuracy came out to be 95%. 
  Also the deployed model gave out correct predictions that helps in predicting the Class of Breast Cancer weather it is 2 i.e. Benign or weather it is 4 i.e. Malignant.
  # Result:  
  Based on the input data entered through the web site, it is then passed through the model for processing the result and prediction weather the person has Breast cancer or not and the class of cancer person is 
  suffering from; 2 for benign, 4 for malignant. 

# Libraries Required to Build the Model:
1. NumPy: Used for numerical operations and handling arrays.
2. Pandas: Utilized for data manipulation and analysis.
3. Matplotlib: Employed for data visualization.
4. Scikit-learn: Used for model building, including:
    a) LogisticRegression: For implementing the regression model.
    b) train_test_split: For splitting the dataset into training and testing sets.
    c) confusion_matrix and accuracy_score: For evaluating model performance.
    d) cross_val_score: For cross-validation.
5. Pickle: Used for saving and loading the trained model.

# Dataset Information
The different variable information are given bellow:
   Sample code number:            id number
1. Clump Thickness:               1 - 10
2. Uniformity of Cell Size:       1 - 10
3. Uniformity of Cell Shape:      1 - 10
4. Marginal Adhesion:             1 - 10
5. Single Epithelial Cell Size:   1 - 10
6. Bare Nuclei:                   1 - 10
7. Bland Chromatin:               1 - 10
8. Normal Nucleoli:               1 - 10
9. Mitoses:                       1 - 10
10. Class:                        (2 for benign, 4 for malignant)

# Results
The accuracy of the machine learning model: 

The accuracy of the machine learning model using k-fold cross validation: 


PREDICTED RESULT:

ACTUAL OUTCOME OF THE MODEL: 






# Conclusion
1. The breast cancer detection model using logistic regression successfully demonstrates the potential of machine learning in medical diagnosis. By analyzing clinical features the model effectively predicts the likelihood of malignancy aiding in early detection and significantly improving patient outcomes. The interpretability and accuracy of logistic regression make it a valuable tool for healthcare professionals, providing insights that enhance diagnostic decision-making.
2. The findings indicate that logistic regression can accurately classify cases underscoring its utility in real-world medical applications. However, challenges such as data quality and feature selection were encountered during implementation. Addressing these challenges through robust data preprocessing and advanced feature engineering can further improve model performance.
3. Future improvements may include incorporating more diverse datasets and exploring additional machine learning algorithms to enhance accuracy. The importance of accurate predictions cannot be overstated as early detection plays a crucial role in patient medical condition.
4. Additionally, the methodology used in this project has potential applications beyond breast cancer detection. Accurate predictions are vital not only in healthcare but also in various sectors such as agriculture, financial sector. Reliable prediction models can help manage resources effectively, contributing to smoother operations and improved user experiences.


