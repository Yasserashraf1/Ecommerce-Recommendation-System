# Renty E-commerce Recommendation System

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![LightFM](https://img.shields.io/badge/LightFM-1.17-green.svg)](https://github.com/lyst/lightfm)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A comprehensive hybrid recommendation system built with LightFM for the Renty e-commerce platform. This system leverages both collaborative filtering and content-based approaches to deliver personalized product recommendations based on user demographics, purchase behavior, and product features.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Architecture](#architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Model Performance](#model-performance)
- [Project Structure](#project-structure)
- [Technical Details](#technical-details)
- [Future Improvements](#future-improvements)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## 🎯 Overview

The Renty E-commerce Recommendation System is designed to enhance user experience and increase sales by providing intelligent, personalized product recommendations. The system analyzes historical transaction data, user demographics, and product characteristics to predict items that customers are most likely to purchase.

### Key Highlights

- **Hybrid Approach**: Combines collaborative filtering with content-based filtering
- **Advanced Feature Engineering**: RFM-based customer segmentation, temporal features, and TF-IDF text features
- **Robust Performance**: Test AUC of 0.84+ with low overfitting (gap < 0.05)
- **Production Ready**: Includes model persistence and inference pipeline

## ✨ Features

### Model Features

- **User Features**
  - Demographics: Gender, Marital Status, Education Level, Occupation, Home Ownership
  - Income Segmentation: 6-level income brackets (VeryLow to Premium)
  - Family Structure: Children categories (NoChildren, OneChild, TwoChildren, ManyChildren)
  - Behavioral Metrics: Total orders, unique products purchased, average order quantity
  - Customer Segments: Bronze, Silver, Gold, Platinum (RFM-inspired)
  - Temporal Features: Season, day of week, weekend indicator

- **Item Features**
  - Product Categories: Extracted from model names
  - Popularity Metrics: 5-level percentile ranking (Niche to Viral)
  - Text Features: TF-IDF extracted from product descriptions (50 features)
  - Customer Reach: Unique customer count per product

### System Capabilities

- ✅ Personalized top-N recommendations
- ✅ Filtering of already-purchased items
- ✅ Real-time prediction for new users (cold-start handling)
- ✅ Comprehensive evaluation metrics (Precision@K, Recall@K, AUC)
- ✅ Hyperparameter optimization with regularization
- ✅ Model serialization for deployment

## 📊 Dataset

### Data Overview

- **Total Transactions**: 55,666 orders
- **Unique Users**: 17,293 customers
- **Unique Products**: 130 items
- **Date Range**: January 2020 - June 2022
- **Sparsity**: 97.52%

### Data Schema

| Column | Type | Description |
|--------|------|-------------|
| OrderDate | datetime64 | Date of purchase |
| StockDate | datetime64 | Date item was stocked |
| OrderNumber | object | Unique order identifier |
| ProductKey | int64 | Product identifier |
| CustomerKey | int64 | Customer identifier |
| TerritoryKey | int64 | Geographic territory |
| OrderLineItem | int64 | Line item number |
| OrderQuantity | int64 | Quantity ordered |
| ModelName | object | Product model name |
| ProductDescription | object | Product description text |
| MaritalStatus | object | Customer marital status |
| Gender | object | Customer gender |
| AnnualIncome | int64 | Annual income ($) |
| TotalChildren | int64 | Number of children |
| EducationLevel | object | Education level |
| Occupation | object | Customer occupation |
| HomeOwner | object | Home ownership status |

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Data Preprocessing                        │
│  • Load Excel data                                           │
│  • Convert datetime columns                                  │
│  • Remove duplicates                                         │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              Advanced Feature Engineering                    │
│  • Temporal features (year, month, quarter, season)         │
│  • Income brackets & children categories                    │
│  • User engagement metrics (RFM-inspired)                   │
│  • Customer segmentation (Bronze/Silver/Gold/Platinum)      │
│  • Item popularity percentiles                               │
│  • MinMax scaling for numerical features                    │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              Text Feature Extraction (TF-IDF)               │
│  • Extract 50 features from product descriptions           │
│  • Use unigrams and bigrams                                 │
│  • Filter by min document frequency                         │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                LightFM Data Preparation                      │
│  • Build user-item interaction matrices                     │
│  • Create user feature matrix (17,293 × 17,330)            │
│  • Create item feature matrix (130 × 206)                   │
│  • Temporal train/test split (80/20)                        │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│         Hyperparameter Tuning & Model Training              │
│  • Grid search with regularization                          │
│  • WARP loss function (implicit feedback)                   │
│  • L2 regularization (item_alpha, user_alpha)              │
│  • Best params: 50 components, 0.05 learning rate          │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              Model Evaluation & Deployment                   │
│  • Precision@K, Recall@K, AUC metrics                       │
│  • Model persistence (pickle)                                │
│  • Recommendation generation API                             │
└─────────────────────────────────────────────────────────────┘
```

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- (Optional) Google Colab account for notebook execution

### Setup Instructions

1. **Clone the repository**

```bash
git clone https://github.com/yourusername/renty-recommendation-system.git
cd renty-recommendation-system
```

2. **Create virtual environment** (recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

### Required Packages

```
pandas>=1.3.0
numpy>=1.21.0
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=1.0.0
scipy>=1.7.0
lightfm>=1.17
```

**Note**: The project uses a specific LightFM fork. Install with:

```bash
pip install git+https://github.com/daviddavo/lightfm
```

## 💻 Usage

### Quick Start

#### 1. Train the Model

```python
# Load and train the complete pipeline
from main import main

filepath = "path/to/GradProject_final_1.xlsx"
model, dataset, user_features, item_features, df, results = main(filepath)
```

#### 2. Generate Recommendations

```python
from recommendation_utils import get_recommendations

# Get recommendations for a specific user
user_id = 14574
recommendations = get_recommendations(
    model=model,
    user_id=user_id,
    dataset=dataset,
    user_features=user_features,
    item_features=item_features,
    df=df,
    n_recommendations=10,
    filter_already_purchased=True
)

print(recommendations)
```

#### 3. Load Pre-trained Model

```python
import pickle

def load_model_artifacts(filepath='renty_lightfm_model_artifacts.pkl'):
    with open(filepath, 'rb') as f:
        artifacts = pickle.load(f)
    return (artifacts['model'], 
            artifacts['dataset'], 
            artifacts['user_features'], 
            artifacts['item_features'])

# Load saved model
model, dataset, user_features, item_features = load_model_artifacts()
```

### Interactive Recommendation Interface

```python
# Get user input and generate recommendations
sample_user_id = int(input("Please enter a User ID: "))

get_recommendations_for_input_user(
    user_id_input=sample_user_id,
    model=loaded_model,
    dataset=loaded_dataset,
    user_features=loaded_user_features,
    item_features=loaded_item_features,
    df=df,
    n_recommendations=10
)
```

### Running the Jupyter Notebook

1. Open the notebook in Google Colab or Jupyter:

```bash
jupyter notebook Ecommerce-Recommendation-System_Renty_Ecommerce_v1.ipynb
```

2. Mount Google Drive (if using Colab):

```python
from google.colab import drive
drive.mount('/content/drive')
```

3. Update the filepath to your dataset location
4. Run all cells sequentially

## 📈 Model Performance

### Final Model Metrics

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **Test AUC** | 0.8409 | Good ranking ability |
| **Precision@5** | 0.1505 | 15% of top-5 recommendations are relevant |
| **Precision@10** | 0.1205 | 12% of top-10 recommendations are relevant |
| **Precision@20** | 0.0868 | 9% of top-20 recommendations are relevant |
| **Recall@5** | 0.2698 | Captures 27% of relevant items in top-5 |
| **Recall@10** | 0.4393 | Captures 44% of relevant items in top-10 |
| **Recall@20** | 0.6402 | Captures 64% of relevant items in top-20 |
| **Overfitting Gap** | 0.0493 | Low overfitting, good generalization |

### Optimal Hyperparameters

```python
{
    'no_components': 50,
    'learning_rate': 0.05,
    'loss': 'warp',
    'item_alpha': 1e-05,  # L2 regularization for items
    'user_alpha': 1e-05,  # L2 regularization for users
    'epochs': 60
}
```

### Performance Comparison

The improved model significantly outperforms the baseline:

- **AUC improvement**: +12% (0.75 → 0.84)
- **Overfitting reduction**: -71% (0.17 → 0.05)
- **Precision@10**: Maintained competitive performance with better generalization

## 📁 Project Structure

```
renty-recommendation-system/
│
├── data/
│   └── GradProject_final_1.xlsx          # Main dataset
│
├── notebooks/
│   └── Ecommerce-Recommendation-System_Renty_Ecommerce_v1.ipynb
│
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py             # Data loading and preprocessing
│   ├── feature_engineering.py            # Advanced feature creation
│   ├── text_features.py                  # TF-IDF extraction
│   ├── model_training.py                 # LightFM model training
│   ├── evaluation.py                     # Model evaluation metrics
│   └── recommendation_utils.py           # Recommendation generation
│
├── models/
│   └── renty_lightfm_model_artifacts.pkl # Saved model
│
├── visualizations/
│   ├── eda_plots/                        # EDA visualizations
│   └── model_performance/                # Performance charts
│
├── requirements.txt                      # Python dependencies
├── README.md                             # This file
├── LICENSE                               # MIT License
└── .gitignore                            # Git ignore file
```

## 🔧 Technical Details

### Feature Engineering Pipeline

#### 1. Temporal Features
- **OrderYear, OrderMonth, OrderQuarter**: Capture temporal trends
- **DayOfWeek, IsWeekend**: Identify purchasing patterns
- **Season**: Spring, Summer, Fall, Winter categorization

#### 2. User Segmentation
```python
# RFM-inspired Customer Value Score
CustomerValueScore = (
    TotalOrders * 0.3 +
    UniqueProducts * 0.3 +
    (TotalQuantity / TotalQuantity.max()) * 100 * 0.4
)

# Segment into tiers
CustomerSegment = pd.qcut(CustomerValueScore, q=4, 
                          labels=['Bronze', 'Silver', 'Gold', 'Platinum'])
```

#### 3. Text Feature Extraction
```python
# TF-IDF Vectorization
tfidf = TfidfVectorizer(
    max_features=50,
    stop_words='english',
    ngram_range=(1, 2),
    min_df=2
)
```

### LightFM Model Configuration

#### Loss Function: WARP (Weighted Approximate-Rank Pairwise)
- Optimized for implicit feedback scenarios
- Focuses on ranking relevant items higher than irrelevant ones
- Particularly effective for top-N recommendations

#### Regularization Strategy
- **L2 Regularization**: Prevents overfitting on sparse data
- **Item Alpha (1e-05)**: Regularizes item feature embeddings
- **User Alpha (1e-05)**: Regularizes user feature embeddings

### Train/Test Split Strategy

**Temporal Split (80/20)**
- Training: First 80% of data chronologically (44,532 samples)
- Testing: Most recent 20% of data (11,134 samples)
- Simulates real-world scenario where model predicts future purchases

### Evaluation Methodology

```python
# Precision@K: Fraction of recommended items that are relevant
Precision@K = (Relevant items in top-K) / K

# Recall@K: Fraction of relevant items that are recommended
Recall@K = (Relevant items in top-K) / (Total relevant items)

# AUC: Area Under ROC Curve
# Measures the model's ability to rank relevant items higher
```

## 🔮 Future Improvements

### Short-term Enhancements
- [ ] Implement real-time recommendation API with Flask/FastAPI
- [ ] Add A/B testing framework for model evaluation
- [ ] Create recommendation diversity metrics
- [ ] Develop cold-start strategy for new users/items
- [ ] Add explainability features (why this recommendation?)

### Medium-term Goals
- [ ] Integrate deep learning approaches (Neural Collaborative Filtering)
- [ ] Implement session-based recommendations (RNN/Transformer)
- [ ] Add multi-objective optimization (relevance + diversity + serendipity)
- [ ] Create recommendation confidence scores
- [ ] Develop automated retraining pipeline

### Long-term Vision
- [ ] Build real-time streaming recommendation system
- [ ] Implement contextual bandits for online learning
- [ ] Add multi-modal features (images, reviews, ratings)
- [ ] Create recommendation explanation dashboard
- [ ] Develop cross-domain recommendation capabilities

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Contribution Guidelines

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Code Style
- Follow PEP 8 style guide for Python code
- Add docstrings to all functions and classes
- Include type hints where appropriate
- Write unit tests for new features

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **LightFM Library**: For providing an excellent hybrid recommendation framework
- **Renty E-commerce**: For the dataset and use case
- **Team Members**: 
  - Abdulrahman (Data Analysis & EDA)
  - Yasser Ashraf (Model Development)
  - Moamen Ahmed (Model Evaluation)

## 📞 Contact

For questions or feedback, please reach out:

- **Email**: yasserashraf3142@gmail.com
- **LinkedIn**: [Yasser Ashraf](https://www.linkedin.com/in/yasserashraf/)

**⭐ If you find this project helpful, please consider giving it a star!**

Last Updated: November 2025

