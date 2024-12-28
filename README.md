# Predictive Modeling with Decision Trees

This project implements and evaluates predictive modeling approaches using decision trees on two distinct datasets from the UCI Machine Learning Repository. The analysis covers both classification and regression problems using different tools and methodologies.

## Datasets

### 1. Vertebral Column Dataset
- **Source**: UCI Machine Learning Repository
- **Format**: Two ARFF files (column_2C_weka.arff and column_3C_weka.arff)
- **Features**: 7 attributes including pelvic_incidence, pelvic_tilt, lumbar_lordosis_angle, etc.
- **Target**: Classification of spinal conditions (Hernia, Spondylolisthesis, Normal)
- **Records**: 310

### 2. Wine Dataset
- **Source**: UCI Machine Learning Repository
- **Features**: 14 attributes
- **Records**: 178
- **Type**: Regression problem with numerical data

## Tools Used

- WEKA (Waikato Environment for Knowledge Analysis)
- Python with scikit-learn
- Jupyter Notebook

## Methodology

### Part 1: Classification (Vertebral Column Dataset)

1. **Data Partitioning**
   - Multiple splits tested: 50:50, 60:40, 70:30 (train:validation)
   - Used J48 model in WEKA
   - Gini index as splitting criterion

2. **Performance Metrics**
   - Kappa Statistics
   - Mean Absolute Error (MAE)
   - Root Mean Squared Error (RMSE)
   - Relative Absolute Error (RAE)
   - Relative Root Squared Error (RRSE)
   - Accuracy

3. **Feature Selection**
   - Comparison between full feature set (7 attributes) and reduced feature set (4 attributes)

### Part 2: Regression (Wine Dataset)

1. **Data Preprocessing**
   - Feature selection
   - Normalization
   - Exploratory analysis

2. **Model Construction**
   - Used GridSearchCV for parameter optimization
   - Multiple train-test splits: 80:20, 75:25, 70:30
   - Decision tree parameters:
     - Criterion: Gini index
     - min_samples_split: 2
     - min_samples_leaf: 1
     - random_state: 42
     - cross_validation: 10

## Key Findings

### Vertebral Column Classification
- Best accuracy achieved: 80.6452% (both 50:50 and 70:30 splits)
- Potential confusion identified between Hernia and Normal classes
- Reduced feature set (4 attributes) showed improved class separation

### Wine Dataset Regression
- Best performance achieved with 70:30 split (96.3% accuracy)
- Model performed consistently across different train-test splits
- Feature reduction from 14 to 7 attributes showed minimal performance impact

## Project Structure

```
├── data/
│   ├── column_2C_weka.arff
│   ├── column_3C_weka.arff
│   └── wine.data
├── notebooks/
│   ├── vertebral_column_analysis.ipynb
│   └── wine_analysis.ipynb
├── results/
│   ├── decision_trees/
│   └── performance_metrics/
└── README.md
```

## Requirements

- Python 3.x
- scikit-learn
- pandas
- numpy
- WEKA 3.8+

## Usage

1. Clone the repository
2. Install required dependencies
3. Run Jupyter notebooks for detailed analysis
4. Use WEKA interface for vertebral column classification

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- UCI Machine Learning Repository for providing the datasets
- Waikato University for WEKA tool




# Clustering Analysis with K-means and Hierarchical Clustering

This project demonstrates the implementation and evaluation of unsupervised learning techniques using K-means clustering and Hierarchical clustering on the Wine dataset from UCI Machine Learning Repository.

## Dataset

### Wine Dataset
- **Source**: UCI Machine Learning Repository
- **Records**: 178 wine samples
- **Features**: Chemical properties including:
  - Alcohol
  - Malic acid
  - Ash
  - Alkalinity of ash
  - Magnesium
  - Total phenols
  - Flavanoids
  - Nonflavanoid phenols
  - Proanthocyanins
  - Color intensity
  - Hue
  - OD280/OD315 of diluted wines

## Methodology

### Data Preprocessing
- Removed class labels for unsupervised learning
- Implemented two normalization techniques:
  - MinMaxScaler (0-1 normalization)
  - StandardScaler (mean=0, std=1)
- StandardScaler showed superior performance and was selected for final analysis

### K-means Clustering
1. **Parameters**
   - n_clusters: 2 through 10
   - n_init: 10 (number of random initializations)
   - random_state: 42 (for reproducibility)

2. **Evaluation Metrics**
   - Within-cluster sum of squares (WCSS)
   - Between-cluster sum of squares (BCSS)
   - Silhouette coefficients
   - Davies-Bouldin index

### Hierarchical Clustering
- Implemented Agglomerative Clustering
- Compared three linkage methods:
  - Single linkage (shortest distance)
  - Complete linkage (largest distance)
  - Average linkage (average distance)
- Generated dendrograms for visual analysis

## Key Findings

### K-means Results
Best performing configurations:
- K=3 showed highest accuracy (69.66%)
- Optimal metrics for K=3:
  - WCSS: 1277.93
  - BCSS: 227471.27
  - Silhouette: 0.28
  - DB Index: 1.39

### Hierarchical Clustering Results
- Generated distinct hierarchical structures for each linkage method
- Produced visual dendrograms showing cluster relationships
- Demonstrated different clustering patterns based on linkage criteria

## Project Structure

```
├── data/
│   └── wine.data
├── notebooks/
│   ├── kmeans_clustering.ipynb
│   └── hierarchical_clustering.ipynb
├── results/
│   ├── kmeans/
│   │   └── cluster_metrics/
│   └── hierarchical/
│       └── dendrograms/
└── README.md
```

## Requirements

- Python 3.x
- scikit-learn
- numpy
- pandas
- matplotlib
- seaborn
- scipy

## Usage

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Run the Jupyter notebooks for detailed analysis

## Code Examples

```python
# Example of K-means clustering implementation
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=3, n_init=10, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

# Example of Hierarchical clustering implementation
from sklearn.cluster import AgglomerativeClustering

hierarchical = AgglomerativeClustering(n_clusters=3)
clusters = hierarchical.fit_predict(X_scaled)
```

## Visualization Examples

The project includes various visualizations:
- Scatter plots of clustering results
- Elbow curves for optimal K selection
- Dendrograms for hierarchical clustering
- Performance metric plots

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- UCI Machine Learning Repository for providing the Wine dataset
- scikit-learn documentation and community




# Comprehensive Machine Learning Project

A systematic implementation of various machine learning techniques covering classification, clustering, and regression tasks, with extensive data preprocessing and model evaluation.

## Overview

This project demonstrates a complete machine learning pipeline including:
- Data preprocessing and cleaning
- Model selection and training
- Validation techniques
- Performance evaluation
- Result visualization

## Project Components

### 1. Data Preprocessing

#### Cleaning Techniques
- Missing data handling through mean imputation
- Duplicate instance removal
- Outlier detection using IQR method
- Influential datapoint detection for regression
- Normality checking using descriptive statistics
- Data transformation (StandardScaler, vectorization, datetime conversion)
- Feature selection using PCA
- Dataset balancing through random oversampling

### 2. Model Implementation

#### Classification Models
- Linear SVM
- Multinomial Naive Bayes
- K-Nearest Neighbors (KNN)
- Logistic Regression

#### Regression Models
- Linear Regression
- Random Forest Regression

#### Clustering Algorithms
- K-means
- Agglomerative Clustering
- DBSCAN

### 3. Validation Approaches

- K-fold Cross Validation
- Percentage Split (75% training, 25% validation)
- Hold-out dataset evaluation

### 4. Evaluation Metrics

#### Classification Metrics
- Accuracy
- Precision
- Recall
- F1-score
- ROC AUC

#### Clustering Metrics
- Silhouette coefficient
- Davies-Bouldin index
- Within-Cluster Sum of Squares
- Between-Cluster Sum of Squares

#### Regression Metrics
- Mean Squared Error
- R-squared

## Project Structure

```
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   ├── preprocessing/
│   ├── classification/
│   ├── clustering/
│   └── regression/
├── src/
│   ├── preprocessing/
│   ├── models/
│   └── evaluation/
├── results/
│   ├── figures/
│   └── metrics/
└── README.md
```

## Visualizations

The project includes various visualization techniques:
- Confusion Matrices
- ROC Curves
- Precision-Recall Curves
- Cluster Plots
- Scatter Plots
- Distribution Plots

## Requirements

```python
# Core requirements
numpy
pandas
scikit-learn
matplotlib
seaborn
scipy

# Additional libraries
imbalanced-learn  # for oversampling
```

## Usage

1. Clone the repository
```bash
git clone https://github.com/username/ml-project.git
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Run preprocessing
```bash
python src/preprocessing/main.py
```

4. Train models
```bash
python src/models/train.py
```

5. Evaluate results
```bash
python src/evaluation/evaluate.py
```

## Results

Detailed results and analysis can be found in the `results/` directory, including:
- Performance metrics for each model
- Comparison of different approaches
- Visualization outputs
- Hold-out test results

## Key Findings

- Successfully implemented and compared multiple ML approaches
- Demonstrated importance of thorough data preprocessing
- Evaluated model performance using various metrics
- Generated comprehensive visualizations for result interpretation
- Validated results using multiple approaches

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Kaggle for providing datasets
- scikit-learn documentation and community
- Various machine learning resources and tutorials

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request
