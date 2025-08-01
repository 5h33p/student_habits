# 🧠 Student Grade Predictor

This project predicts student exam scores using machine learning based on their lifestyle and habits. 

📌 **Key Steps**
- Data preprocessing (scaling, encoding)
- EDA (heatmaps, scatter plots)
- Model training with pipelines (Linear Regression, Ridge, Lasso, SVR)
- Performance evaluation (R², MSE, MAE)
- Comparison with dummy baseline

📊 **Tools:** Python, Pandas, Scikit-learn, Seaborn


# 1 Introduction

The main goal of this project is to create a machine learning model capable of predicting students' grades based on their habits.  
The process includes:
- [Data Gathering](#21-getting-data)  
- [Exploratory Analysis](#22-exploratory-analysis)  
- [Preprocessing](#23-preprocessing-data)  
- [Feature/Model Selection](#24-featuremodel-selection)
- [Model Train and Test](#25-model-train-and-test)
- [Pipeline Creation](#26-model-training-pipeline)  
- [Result Evaluation](#27-evaluation-results)  

In summary, the trained model was able to get better results than a dummy model used by comparison. The Python notebook [Development](./Development.ipynb) is divided according to the sections presented in this document.

# 2 Body

This section presents the process of getting, processing and loading the data (**ETL**), as well as the methods used to **select the features**, **create the regression model**, **search** for the **best hyperparameters**, **train the final model**, **create the pipeline** and **evaluate** the **results** based on the results of a dummy model.

## 2.1 Getting Data

Firstly, the data was extracted from a [Kaggle repository](https://www.kaggle.com/datasets/aryan208/student-habits-and-academic-performance-dataset) and the first rolls were printed to check the data columns, rows and null values. A copy of the dataframe was also uploaded in case the kaggle dataset source is no more available. The dataframe has 31 columns, which of them 30 are features (*student_id*, *age*, *gender*, etc.) and one is the target (*exame_score*).

## 2.2 Exploratory Analysis

This section presents the exploratory analysis conducted in order to visualize any correlation between features and exame score. In order to achieve this result, heat maps and scatter plots were applied on the data.


The numerical features were separated and their correlation summarized in the [heatmap](#heatmap) showed next.

<img id='heatmap' width="1272" height="766" alt="image" src="https://github.com/user-attachments/assets/c3e4bdb4-bc6b-44f0-883e-c29fca363734" />

The heatmap results demonstrate that the *exam_score* column has some correlation with the following numerical features:
|Feature|Correlation|
| --- | --- |
|[*study_hours_per_day*](#study-hours-plot)|  +
|[*sleep_hours*](#sleep-hours-plot) |  +
|*exercise_frequency* |  +
|[*previous_gpa*](#previous-gpa-plot) |  +
|[*stress_level*](#stress-level-plot)|  -
|[*screen_time*](#screen-time-plot) |  +
|[*motivation_level*](#motivation-level-plot) |  +
|[*exam_anxiety_score*](#exam-anxiety-plot)|  -

A correlation scatter plot is available for most of these features in the [Appendix](#appendix) section. No further exploratory analysis was conducted on the data.

## 2.3 Preprocessing Data
This section shows the preprocess applied to the dataset, applying feature **scaling**.

## 2.4 Feature/Model Selection
## 2.5 Model Train and Test
## 2.6 Model Training Pipeline
## 2.7 Evaluation Results
# 3 Conclusion
# 4 Appendix

## Previous GPA Plot
<img id='previous gpa plot' width="1238" height="855" alt="image" src="https://github.com/user-attachments/assets/a5e14030-2b62-44ac-9444-efdc22ee232f" />


## Exam Anxiety Plot
<img id='exam anxiety plot' width="1238" height="855" alt="image" src="https://github.com/user-attachments/assets/8a9e6560-d744-44fa-9eee-768be78480f7" />

## Stress Level Plot
<img id='stress level plot' width="1238" height="855" alt="image" src="https://github.com/user-attachments/assets/d3719c79-4dc4-4532-ac6d-af4175121870" />

## Sleep Hours Plot
<img id='sleep hours plot' width="1238" height="855" alt="image" src="https://github.com/user-attachments/assets/c72d58f3-42ea-4737-a7f2-e1fa918a33af" />

## Screen Time Plot
<img id='screen time plot' width="1238" height="855" alt="image" src="https://github.com/user-attachments/assets/d936cbc6-1957-4b8f-bdb4-72b8c35902ce" />

## Motivation Level Plot
<img id='motivation level plot' width="1238" height="855" alt="image" src="https://github.com/user-attachments/assets/d5bd8c6f-c951-4aa0-a292-61e7fcd9c80f" />

## Study Hours Plot
<img id='study hours plot' width="1238" height="855" alt="image" src="https://github.com/user-attachments/assets/5748d195-40ba-4884-8289-8652c3dad4ea" />
