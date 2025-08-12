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
- [Feature Selection](#24-feature-selection)
- [Model Training and Test](#25-model-training-and-test)
- [Pipeline Creation](#26-model-training-pipeline)  
- [Result Evaluation](#27-evaluation-results)  

In summary, the trained model was able to get better results (R² = 0.87) than a dummy model used by comparison. The Python notebook [Development](./Development.ipynb) is divided according to the sections presented in this document. In the [Conclusion section](#3-conclusion) there are some improvements observed by myself that can be made in future ML projects.

# 2 Body

This section presents the process of getting, processing and loading the data (**ETL**), as well as the methods used to **select the features**, **create the regression model**, **search** for the **best hyperparameters**, **train the final model**, **create the pipeline** and **evaluate** the **results** based on the results of a dummy model.

## 2.1 Getting Data

Firstly, the data is extracted from a [Kaggle repository](https://www.kaggle.com/datasets/aryan208/student-habits-and-academic-performance-dataset) and the first rows are printed to check the data columns, rows and null values. A copy of the dataframe is also uploaded in case the Kaggle dataset source is no longer available. The dataset contains 31 columns: 30 feature variables (e.g., *student_id*, *age*, *gender*, etc.) and 1 target variable (*exam_score*).

## 2.2 Exploratory Analysis

This section presents the exploratory analysis conducted to visualize correlation between features and exam score. In order to achieve this result, heat maps and scatter plots are applied on the data.


The numerical features are separated and their correlation summarized in the [heatmap](#heatmap) shown below.

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
This section shows the preprocessing applied to the dataset, applying feature **scaling**.


Firstly, each column is converted to the right data type, between (`float64`, `int64` or `object`). The Scikit-learn function `ColumnTransformer()` used in the preprocessing needs the categorical and numerical features to be named, leading to the classification between **categorical** and **numerical features**. The process demands less lines, since the entire dataset already has the right data types. The entire preprocessing step is shown below.

```python
# Separating features from scores
X = data.drop('exam_score', axis=1)
y = data.exam_score
data.study_hours_per_day = data.study_hours_per_day.astype(float)
categorical_data = X.loc[:, (X.dtypes=='object') | (X.dtypes=='category')] #Getting categorical data to preprocess
# print(categorical_data.nunique())

# Separating categorical and numerical data column names
cat_columns = categorical_data.columns
num_columns = X.drop(cat_columns, axis=1).columns

# Creating ColumnTransformer object
preprocessor = ColumnTransformer(
    transformers=[('num', StandardScaler(), num_columns),
                  ('cat', OneHotEncoder(handle_unknown='ignore', drop='first'), cat_columns)])

# Fitting preprocessor to the data
processor = preprocessor.fit(X)
X = processor.transform(X)
X = pd.DataFrame(X, columns=processor.get_feature_names_out())
```

Once the features are preprocessed, we can move on to the next step since this is a regression model and therefore there is no need of scaling the output.

## 2.4 Feature Selection
This section demonstrates the process of **feature selection**, where four different feature selection methods were employed:
- Information Gain
- Fisher's Score
- Correlation Coefficient
- Variance Threshold

The next table presents the ten best features according to each feature selection method.

| | Info | Fisher | Correlation | Variance
-- | -- |-- |-- |-- |
1 | *previous_gpa* |*previous_gpa*| *previous_gpa*| *social_activity*
2 |*motivation_level* |*motivation_level* |*motivation_level* |*exam_anxiety_score*
3 |*exam_anxiety_score* |*study_hours_per_day* |*study_hours_per_day*| *social_media_hours*
4 |*study_hours_per_day*| *exam_anxiety_score* |*exam_anxiety_score* |*screen_time*
5 |*screen_time* |*screen_time* |*screen_time* |*time_management_score*
6 |*study_environment_Dorm* |*study_environment_Dorm*| *study_environment_Dorm*| *parental_support_level*
7 |*sleep_hours* |*access_to_tutoring_Yes* |*access_to_tutoring_Yes* |*stress_level*
8 |*stress_level* |*stress_level* |*stress_level* |*mental_health_rating*
9 |*social_activity* |*dropout_risk_Yes* |*study_environment_QuietRoom* |*student_id*
10 |*attendance_percentage_90.0*| *sleep_hours* |*sleep_hours*| *age*

The ten features selected to train the model were the mode (most frequent features) from the results. The selected features are presented in the following list.

- *previous_gpa*: the student's last grade
- *motivation_level*: a motivation score of the student ranging from 0 to 10
- *study_hours_per_day*: number of hours studied by the student
- *exam_anxiety_score*: an anxiety score ranging from 0 to 10
- *screen_time*: number of hours spent by the student on a screen
- *study_environment_Dorm*: whether the student's study environment is his dormitory
- *access_to_tutoring_Yes*: whether the student has access extracurricular classes
- *stress_level*: a stress level score of the student ranging from 0 to 10
- *dropout_risk_Yes*: whether the student has dropout risk
- *sleep_hours*: daily hours of sleep of the student

## 2.5 Model Training and Test

This section presents the training process, as well as the metrics used to select the regression model. In order to get the best model, 3 different regression models were considered and tested:
- Linear Regression
- Lasso
- Ridge

  Firstly, the dataset was divided into train and test sets using the `train_test_split()` function, where 20% of the data was used as test data and the `random_state` parameter was set to 8 for reproducibility, as shown below.
  ```python
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=8)
  ```

In order to get the best model and hyperparameters, the `GridSearchCV()` method from the sklearn library was applied to the three regression models with the following search grids.

### LinearRegression
|*fit_intercept*|
|---|
| Yes|
|No|

### Lasso
|*alpha*|
|---|
|0.001|
|0.01|
|0.1|
|1|
|10|
|100|

### Ridge
|*alpha*|
|---|
|0.001|
|0.01|
|0.1|
|1|
|10|
|100|

The *threading* library was used to process the grid search in parallel. The best results of each search are presented below.
Model| Hyperparameters|	Train Score|	Test Score|
|---|---|---|---|
LinearRegression	|*fit_intercept*: True	|0.87063|	0.86906
Ridge	|*alpha*: 10.0	|0.87063	|0.86906
Lasso	|*alpha*: 0.01|	0.87065|	0.86907

As shown in the table, the best estimator is Lasso, with a higher score in both the *train_data* and *test_data*. The best model has the hyperparameter *alpha* set to 0.01.
## 2.6 Model Training Pipeline

This section shows the creation of a `Pipeline` object to resume the process presented.

The code used to create and fit the `Pipeline` is the following.
```python
final_preprocessor = ColumnTransformer(
    transformers=[('num', StandardScaler(), num_columns),
                  ('cat', OneHotEncoder(handle_unknown='ignore', drop='first'), cat_columns)])


final_pipeline = Pipeline([('scaler', final_preprocessor),
                          ('selector', SelectKBest(f_classif, k=10)),
                          ('model', Lasso(alpha=0.01, random_state=8))])



final_pipeline.fit(data.drop(columns='exam_score'), y)
final_pipeline.score(X, y)
```

## 2.7 Evaluation Results
This section presents the metrics used to evaluate the trained model.

A `DummyRegressor` from the sklearn library was created using the **median** as strategy and its performance was compared to the final pipeline object over the metrics below.
- Mean Squared Error (MSE)
- Mean Absolute Error (MSA)
- R² Score

The resulting plot is shown below.

<img width="1502" height="806" alt="image" src="https://github.com/user-attachments/assets/3fafb308-bd79-407f-a34e-b4583d3c1c69" />

The final model had a better performance than the dummy model, i.e. the final model presented less total error in both, the training and test set, as well as a better R2 score of about 0.87. This result demonstrates that the ML model obtained is meaningful, it can describe the data and make new predictions. The exact values presented in the graph are shown next.

|Final Model|Train Score |Test Score |
|---|---|--- |
|Mean Absolute Error| 3.195566| 3.199250|
|Mean Squared Error |17.424589| 17.381873|
|R2 Score |0.870698| 0.869077


|Dummy Model|Train Score| Test Score |
|---|---|--- |
Mean Absolute Error| 9.187828| 9.108313 |
|Mean Squared Error |149.778484| 147.137563 |
|R2 Score |-0.111456 |-0.108260

The next histogram presents the distribution of residuals with the model developed. It's noticeable that the frequency peak is around +2 points, i.e. the model frequently predicts less than the actual exam score the student has / had.

<img width="1255" height="701" alt="image" src="https://github.com/user-attachments/assets/1604c146-3931-46b7-8122-83b89c9efe43" />

To measure the accuracy of the model, since the residuals follow a normal distribution pattern with relatively low variance, the box plot shown below presents two tolerance bands that include a part of the interquartile range (IQR). The ±2 score points tolerance band brings the model's predictions to roughly 50% of accuracy.

<img width="1201" height="701" alt="image" src="https://github.com/user-attachments/assets/4b44cb36-b6cc-4330-9833-40e5106c3874" />

In order to check how accuracy improves based on the tolerance range, the plot shwon next was created. It's possible to see that the accuracy improvement stagnates around ±10 to ±12 score points, with an accuracy over ≈95%, making it the best tolerance prediction margins.

<img width="1233" height="701" alt="image" src="https://github.com/user-attachments/assets/de512038-1c73-4989-9254-44e5bd42a5f3" />

# 3 Conclusion

As presented, the model can assimilate the data well. It's possible to vary the tolerance in order to get more precise outcomes, achieving more than 95% accuracy between ±10 to ±12 points. It's important to keep in mind that this project was made with studying purposes only, to show and improve my skills as an ML developer. Some traits that should be thought about in the next projects are:
- **Better defining data types of each column**: one must spend more time changing the column types to the right ones, specially the boolean variables, that must be, as better practice, kept out of the encoding and scaling process.
- **Checking for correlations between the categorical features and the target**
- **Comparing models with different numbers of features**
- **Trying using a DeepLearning model as regression model**
- **Include more statistical insights in the exploratory analysis and in the model evaluation**
- **Filter outliers from data**
- **Deploy model via Flask**
- **Integrate with SQL databases**

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
