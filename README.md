# CO2 Emission Prediction in Rwanda

This Jupyter Notebook presents a data analysis and machine learning project focused on predicting CO2 emissions in Rwanda. The notebook explores various data preprocessing steps, feature engineering, model selection, and evaluation techniques to create a predictive model for CO2 emissions.

## Table of Contents

- [Introduction](#introduction)
- [Dependencies](#dependencies)
- [Dataset](#dataset)
- [Methodology](#methodology)
  - [Importing Libraries](#importing-libraries)
  - [Reading and Exploring Data](#reading-and-exploring-data)
  - [Data Preprocessing](#data-preprocessing)
  - [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
  - [Feature Engineering](#feature-engineering)
  - [Model Selection](#model-selection)
  - [Model Training and Evaluation](#model-training-and-evaluation)
- [Conclusion](#conclusion)
- [Usage](#usage)
- [Acknowledgments](#acknowledgments)
- [License](#license)

## Introduction

The purpose of this project is to develop a machine learning model that predicts CO2 emissions in Rwanda. The project involves a comprehensive data analysis, including data preprocessing, exploratory data analysis, feature engineering, and the selection of suitable machine learning algorithms.

## Dependencies

The following Python libraries are required to run this notebook:

- pandas
- numpy
- geopandas
- shapely
- folium
- matplotlib
- seaborn
- scikit-learn
- xgboost
- lightgbm
- fasteda
- optuna
- haversine

You can install these dependencies using the following command:

```bash

pip install pandas numpy geopandas shapely folium matplotlib seaborn scikit-learn xgboost lightgbm fasteda optuna haversine
```

## Dataset

The dataset for this project consists of CSV files: 'train.csv' and 'test.csv'. These files contain relevant features and CO2 emission values that are used for training and evaluating the predictive model.

## Methodology

### Importing Libraries

The initial step involves importing necessary Python libraries for data analysis, visualization, and modeling.

### Reading and Exploring Data

The provided CSV files are read into dataframes using the `pandas` library. Exploratory data analysis techniques are applied to understand the dataset's characteristics and structure.

### Data Preprocessing

Data preprocessing steps include handling missing values, removing irrelevant features, and transforming the target variable for better model performance.

### Exploratory Data Analysis (EDA)

EDA techniques are employed to analyze the distribution of emissions, identify trends, and explore relationships between features.

### Feature Engineering

Feature engineering involves creating new features from existing ones, such as calculating distances between locations and considering temporal factors.

### Model Selection

Different machine learning algorithms, including XGBoost, LightGBM, and RandomForestRegressor, are considered for predicting CO2 emissions.

### Model Training and Evaluation

Selected models are trained on the training dataset and evaluated using appropriate evaluation metrics, such as root mean squared error (RMSE).

## Conclusion

This Jupyter Notebook provides a comprehensive analysis and predictive model for CO2 emission prediction in Rwanda. The project showcases data preprocessing, exploratory data analysis, feature engineering, and model training, all aimed at accurate emission predictions.

## Usage

To use this notebook:

1. Install the required dependencies as mentioned in the 'Dependencies' section.
2. Place the 'train.csv' and 'test.csv' files in the same directory as this notebook.
3. Run the code cells sequentially to perform data analysis and model training.
4. Review the generated predictions and model evaluation results.

## Acknowledgments

This project is based on real-world data and aims to contribute to a better understanding of CO2 emissions in Rwanda. Special thanks to the developers and contributors of the open-source libraries used in this notebook.

## License

This project is licensed under the [MIT License](LICENSE).
