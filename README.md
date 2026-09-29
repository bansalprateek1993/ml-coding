# ML Coding - Machine Learning Algorithms from Scratch

A comprehensive collection of machine learning algorithms implemented from scratch and with scikit-learn. This project covers supervised learning, unsupervised learning, and recommendation systems with practical examples and datasets.

## 📚 Project Structure

### Core Algorithms

#### **Classification**
- **Logistic Regression** (`logistic_regression/`)
  - Binary and multi-class classification
  - Uses breast cancer dataset
  - Implements gradient descent optimization
  - Training, validation, and test splits

- **K-Nearest Neighbors (KNN)** (`knn/`)
  - Distance-based classification
  - Configurable k-values
  - Euclidean distance metric

#### **Regression**
- **Linear Regression** (`regression/`)
  - From-scratch implementation using least squares
  - California housing price prediction
  - Feature scaling with StandardScaler
  - Train/validation/test split workflow

- **General Regression** (`reg/`)
  - Multiple regression approaches
  - Linear regression implementation
  - K-Means integration
  - Multi-class logistic regression

#### **Clustering**
- **K-Means** (`kmeans/`)
  - Unsupervised clustering algorithm
  - Centroid-based optimization
  - Dataset loading and preprocessing

#### **Deep Learning**
- **Neural Networks** (`neural_network/`)
  - From-scratch implementation
  - Forward and backward propagation
  - Breast cancer classification task
  - Batch processing with feature scaling

### Recommendation Systems
- **Collaborative Filtering** (`collabrative_filtering/`)
  - User-based collaborative filtering
  - Cosine similarity computation
  - User-item interaction matrix
  - Rating prediction using similar users
  - Content-based filtering approaches
  - PCA for dimensionality reduction
  - Reinforcement learning components

### Utilities
- **Revise** (`Revise/`)
  - Neural network implementations
  - Rate limiting algorithms
  - Median latency calculations

## 🛠️ Technologies & Dependencies

- **Python 3.x**
- **NumPy** - Numerical computations
- **Pandas** - Data manipulation and analysis
- **Scikit-learn** - Machine learning utilities and datasets
- **Matplotlib** - Data visualization

### Installation

```bash
pip install -r requirements.txt
```

## 📊 Datasets Used

1. **Breast Cancer Dataset** (UCI Machine Learning Repository)
   - Classification task (malignant vs benign)
   - ~569 samples, 30 features
   - Used in: Logistic Regression, Neural Networks

2. **California Housing Dataset**
   - Regression task (price prediction)
   - ~20,640 samples, 8 features
   - Used in: Linear Regression modules

## 🚀 Getting Started

### Running Classification Examples

```bash
# Logistic Regression
python logistic_regression/main.py

# Neural Network Classification
python neural_network/main.py
```

### Running Regression Examples

```bash
python regression/main.py
python reg/main.py
```

### Using Collaborative Filtering

```bash
python collabrative_filtering/cf.py
```

## 📖 Key Features

### Data Pipeline
- Data loading from scikit-learn datasets
- Exploratory data analysis (shape, dtypes, statistics)
- Missing value detection and handling
- Duplicate removal
- Feature scaling with StandardScaler
- Train/validation/test splits with stratification

### Model Training
- Batch and online learning
- Gradient descent optimization
- Cross-validation support
- Similarity-based predictions

### Evaluation
- Classification metrics
- Regression error metrics
- Performance tracking across splits

## 🔍 Algorithm Implementations

### Collaborative Filtering Pipeline
1. **User-Item Interaction Matrix** - Captures ratings
2. **Cosine Similarity** - Computes user-to-user similarity
3. **Similarity Matrix** - N×N matrix of user similarities
4. **Prediction** - Weighted average of similar users' ratings
5. **Masking** - Prevents recommending already-rated items

### Neural Network Components
- Activation functions (ReLU, Sigmoid, Softmax)
- Loss functions (Cross-entropy, MSE)
- Backpropagation algorithm
- Layer-wise computations

### Linear Regression
- Normal equation solution
- Gradient descent optimization
- Feature normalization

## 📋 Project Workflow

Each algorithm module typically includes:
1. **Data Loading** - Fetch datasets from scikit-learn
2. **Data Inspection** - Analyze structure and statistics
3. **Data Cleaning** - Handle missing values and duplicates
4. **Preprocessing** - Scaling and splitting
5. **Model Training** - Fit the algorithm
6. **Evaluation** - Performance metrics on test set

## 🎯 Learning Outcomes

This project demonstrates:
- How ML algorithms work under the hood
- Proper data preprocessing techniques
- Train/validation/test split methodology
- Feature scaling importance
- Performance evaluation methods
- Similarity-based learning approaches
- Deep learning fundamentals

## 📝 Notes

- The `delete/` directory contains deprecated implementations
- All algorithms support both numpy arrays and pandas DataFrames
- StandardScaler is used consistently for feature normalization
- Random state is fixed for reproducibility (random_state=42)

## 🔄 Contributing

To add new algorithms:
1. Create a new directory with your algorithm name
2. Implement core functions
3. Add a `main.py` with example usage
4. Document the algorithm and usage

## 📄 License

This is an educational project for learning machine learning concepts.

---

**Repository**: [bansalprateek1993/ml-coding](https://github.com/bansalprateek1993/ml-coding)
