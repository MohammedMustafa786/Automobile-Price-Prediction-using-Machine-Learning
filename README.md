# Automobile Price Prediction: Preprocessing, EDA, and Modeling

## Overview

This project provides a comprehensive walkthrough of a data science workflow using the "Automobile Data Set" from the UCI Machine Learning Repository. The primary objective is to clean, analyze, and model the dataset to accurately predict vehicle prices based on their features.

The notebook demonstrates key data science techniques, including:
* In-depth data cleaning and preprocessing.
* Feature engineering and transformation.
* Exploratory Data Analysis (EDA) to find feature-price relationships.
* Implementation of `ColumnTransformer` and `Pipeline` to prevent data leakage and create a reproducible workflow.
* Comparison of different regression models, including Linear Regression and a Random Forest Regressor.

**Dataset:** `Raw_Automobile_dataset.csv` (Automobile Data Set from UCI)

## Project Workflow

The notebook follows these key steps:

1.  **Data Cleaning & Preprocessing:**
    * Loaded the raw `.csv` file and assigned the correct headers.
    * Identified missing values (represented by `?`) and replaced them with `np.nan`.
    * Imputed missing data using appropriate strategies (mean for numerical features, mode for categorical features).
    * Corrected the data types for all columns (e.g., converting `price` and `horsepower` to numeric).

2.  **Feature Engineering:**
    * Created new features by converting units (MPG to L/100km) for `city-mpg` and `highway-mpg`.
    * Applied one-hot encoding to categorical variables like `drive-wheels`, `engine-location`, and `fuel-type` to prepare them for modeling.
    * Normalized numerical features (`length`, `width`, `height`) using max-scaling.
    * Used data binning (`pd.cut`) to group `horsepower` and `price` into 'Low', 'Medium', and 'High' categories for analysis.

3.  **Exploratory Data Analysis (EDA):**
    * Analyzed the correlation between numerical features and `price` using a Pearson correlation heatmap.
    * Visualized the relationships between key predictors and the target variable (`price`) using:
        * **Regression Plots (regplot):** To show the positive correlation for features like `curb-weight`, `engine-size`, and `horsepower`.
        * **Box Plots (boxplot):** To understand the price differences between categorical features like `fuel-type` and `aspiration`.

4.  **Modeling and Evaluation:**
    Three different regression models were built and evaluated to find the best-performing approach. A `Pipeline` was used to ensure that preprocessing steps (imputation, scaling) were learned *only* from the training data, preventing data leakage.

    * **Model 1: Simple Linear Regression (No Pipeline)**
        * **Description:** A baseline model using only manually selected numeric features.
        * **Result (R²):** `0.6368`

    * **Model 2: Linear Regression (with Pipeline)**
        * **Description:** A more robust model using a `ColumnTransformer` and `Pipeline` to handle imputation, `StandardScaler` for numeric features, and `OneHotEncoder` for categorical features.
        * **Result (R²):** `0.7812`

    * **Model 3: Random Forest Regressor (with Pipeline)**
        * **Description:** The final model using the same pipeline but substituting the regressor with a `RandomForestRegressor`.
        * **Key Improvement:** A **log-transform** (`np.log`) was applied to the target variable (`price`) before training to correct its heavy right-skew. Predictions were converted back to the original scale using `np.exp`.
        * **Result (R²):** `0.9143`

## Model Comparison & Results

| Model | Target Transform | R-squared (R²) Score | Mean Squared Error (MSE) |
| :--- | :--- | :--- | :--- |
| Linear Regression (Basic) | None | 0.6368 | 17,680,392.08 |
| Linear Regression (Pipeline)| None | 0.7812 | 26,765,883.16 |
| **Random Forest (Pipeline)**| **Log Transform** | **0.9143** | **10,489,000.85** |

The distribution plots of actual vs. predicted prices clearly show the Random Forest model's superior accuracy, as its predictions closely follow the actual price distribution.

## Conclusion

The **Random Forest Regressor (Model 3)** was the clear winner, achieving an **R² score of 0.9143**.

This project highlights the importance of:
1.  **Thorough EDA:** Identifying the right-skewed nature of the `price` variable was critical.
2.  **Target Transformation:** Applying a log-transform to the target variable significantly improved the model's performance by normalizing its distribution.
3.  **Using Pipelines:** `ColumnTransformer` and `Pipeline` are essential tools for creating a reproducible and robust machine learning workflow that prevents common errors like data leakage.

## How to Run

1.  **Clone the repository:**
    ```bash
    git clone [your-repository-url]
    cd [repository-name]
    ```

2.  **Install the required libraries:**
    ```bash
    pip install pandas numpy matplotlib seaborn scikit-learn
    ```

3.  **Launch the Jupyter Notebook:**
    ```bash
    jupyter notebook Automobile_Price_Prediction.ipynb

    ```
