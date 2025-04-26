# Student Success Predictor

This project leverages machine learning to predict student academic success based on lifestyle factors. Using a Decision Tree Classifier with a max depth of 5, the model analyzes a student lifestyle dataset, including features like study hours, stress levels, sleep, and physical activity, to classify students into high-performing (Grades > 8) or not. The model achieves strong accuracy, significantly outperforming the baseline, making it a valuable tool for understanding the impact of lifestyle on academic performance.

## Project Overview

- **Dataset**: The dataset contains 12,000 rows of student lifestyle data with the following columns:  
  - `Student_ID` (dropped during preprocessing)  
  - `Gender` (dropped during preprocessing)  
  - `Grades` (target variable, converted to binary)  
  - `Study_Hours_Per_Day`  
  - `Extracurricular_Hours_Per_Day`  
  - `Sleep_Hours_Per_Day`  
  - `Social_Hours_Per_Day`  
  - `Physical_Activity_Hours_Per_Day`  
  - `Stress_Level` (categorical: Low, Moderate, High)

## Project Steps

### 1. Data Preprocessing
- Loads the dataset using pandas.  
- Converts `Grades` to a binary target (1 if Grades > 8, 0 otherwise).  
- Drops irrelevant columns (`Student_ID`, `Gender`).

### 2. Exploratory Data Analysis (EDA)
- **Correlation Heatmap**: Visualizes correlations between numeric features.  
- **Stress vs. Grades**: A count plot showing the relationship between stress levels and grades.  
- **Stress Distribution**: A pie chart of stress levels.  
- **Activity Hours by Grade**: A stacked bar chart showing average hours spent on activities, grouped by grades.  
- **Grade Balance**: A bar plot of the target variable's distribution.

### 3. Model Training
- **Pipeline**: Uses an encoder for the categorical `Stress_Level` and a Decision Tree Classifier with `max_depth=5`.  
- **Data Splitting**:  
  - Train: 64% (after splitting 80% of data, then 80% of that for training)  
  - Validation: 16% (20% of the 80% training split)  
  - Test: 20%  
- **Hyperparameter Tuning**: Tests `max_depth` from 1 to 19, plotting training and validation accuracy.

### 4. Model Visualization
- **Decision Tree Plot**: Visualizes the trained decision tree with a depth of 5.  
- **Feature Importance**: A bar plot showing the importance of each feature in the decision tree.

## Feature Importance
The bar plot reveals the importance of each feature in predicting student success:  
- **Study_Hours_Per_Day**: Most important (~0.6 importance score). More study hours strongly correlate with better grades.  
- **Stress_Level**: Second most important (~0.3 importance score). High stress negatively impacts grades.  
- **Physical_Activity_Hours_Per_Day**: Moderate importance (~0.1 importance score).  
- **Sleep_Hours_Per_Day**, **Social_Hours_Per_Day**, **Extracurricular_Hours_Per_Day**: Minimal importance (<0.05 importance score).

## Model Performance
- **Baseline Accuracy**: 63% (majority class proportion in training data).  
- **Training Accuracy**: 97%  
- **Validation Accuracy**: 96%  
- **Test Accuracy**: 97%  

The model significantly outperforms the baseline, achieving high accuracy across all splits, indicating good generalization.

## Dependencies
This project uses the following Python libraries and tools:  
- pandas: For data loading and manipulation.  
- seaborn: For creating visualizations like heatmaps and count plots.  
- matplotlib: For generating plots such as pie charts and bar charts.  
- scikit-learn: For machine learning tasks, including the Decision Tree Classifier and data splitting.  
- category_encoders: For encoding categorical variables like `Stress_Level`.

## Usage
1. Clone the repository.  
2. Ensure you have the dataset with the specified columns in your working directory.  
3. Run the script in a Jupyter Notebook or Python environment.  
4. The script will generate visualizations and print accuracy scores.

## Future Improvements
- Experiment with other models (e.g., Random Forest, XGBoost).  
- Add feature engineering (e.g., interaction terms).  
- Address potential overfitting by further tuning `max_depth` or using regularization.

---

