# Fundamentals of Machine Learning

> "Machine learning is the field of study that gives computers the ability to learn without being explicitly programmed." -- Arthur Samuel (1959)

## What You Will Learn in This Chapter

- [ ] Distinguish the overall landscape of machine learning and its three learning paradigms
- [ ] Understand and implement major supervised learning algorithms (linear regression, logistic regression, decision trees, SVM)
- [ ] Explain the principles and application scenarios of unsupervised learning (clustering, dimensionality reduction, anomaly detection)
- [ ] Understand neural network structure, activation functions, and backpropagation from both mathematical and coding perspectives
- [ ] Distinguish the differences among major deep learning architectures (CNN, RNN/LSTM, Transformer)
- [ ] Properly use evaluation metrics (Accuracy, Precision, Recall, F1, AUC-ROC, MSE, R-squared)
- [ ] Diagnose and address overfitting and underfitting in practice
- [ ] Build a machine learning pipeline using scikit-learn / NumPy


## Prerequisites

Having the following knowledge will deepen your understanding before reading this guide:

- Basic programming knowledge
- Understanding of related foundational concepts
- Familiarity with the content of [Distributed Systems](./00-distributed-systems.md)

---

## Table of Contents

1. [What Is Machine Learning](#1-what-is-machine-learning)
2. [Supervised Learning](#2-supervised-learning)
3. [Unsupervised Learning](#3-unsupervised-learning)
4. [Overview of Reinforcement Learning](#4-overview-of-reinforcement-learning)
5. [Fundamentals of Neural Networks](#5-fundamentals-of-neural-networks)
6. [Deep Learning and Architectures](#6-deep-learning-and-architectures)
7. [System of Evaluation Metrics](#7-system-of-evaluation-metrics)
8. [Overfitting and Regularization](#8-overfitting-and-regularization)
9. [Feature Engineering](#9-feature-engineering)
10. [Practical Machine Learning Pipeline](#10-practical-machine-learning-pipeline)
11. [Generative AI and Large Language Models](#11-generative-ai-and-large-language-models)
12. [Anti-Pattern Collection](#12-anti-pattern-collection)
13. [Practical Exercises (3 Levels)](#13-practical-exercises-3-levels)
14. [FAQ (Frequently Asked Questions)](#14-faq-frequently-asked-questions)
15. [Summary and Comparison Tables](#15-summary-and-comparison-tables)
16. [References](#16-references)

---

## 1. What Is Machine Learning

### 1.1 Comparison with Traditional Programming

In software development, the traditional approach and the ML approach are fundamentally different.

```
Traditional Programming vs Machine Learning:

  Traditional Programming:
  ┌──────────┐   ┌──────────┐   ┌──────────┐
  │  Rules    │ + │  Data    │ → │  Results  │
  │(written  │   │ (input)  │   │ (output)  │
  │ by human)│   │          │   │           │
  └──────────┘   └──────────┘   └──────────┘

  Machine Learning:
  ┌──────────┐   ┌──────────┐   ┌──────────┐
  │  Data    │ + │  Results  │ → │  Rules    │
  │ (input)  │   │ (labels)  │   │ (model)   │
  └──────────┘   └──────────┘   └──────────┘
```

This difference is fundamental. In traditional programming, developers manually write all conditional branches and logic. In machine learning, the algorithm automatically acquires rules (a model) from "data and correct answer pairs."

**Example: Spam Filter**

| Approach | Method | Challenge |
|----------|--------|-----------|
| Traditional | Manually write rules like `if "winner" in email → spam` | Cannot handle new types of spam |
| ML-based | Automatically learn patterns from large volumes of spam/non-spam emails | Requires data |

### 1.2 Four Scenarios Where Machine Learning Is Effective

1. **Rules are too complex to write manually**: Image recognition (can you articulate the rules for distinguishing cats from dogs?)
2. **Rules constantly change**: Spam detection (attackers constantly develop new techniques)
3. **Pattern discovery from large data**: Recommendations, demand forecasting
4. **Domains beyond human intuition**: Protein structure prediction (AlphaFold), materials design

### 1.3 Mathematical Definition of Machine Learning

Tom Mitchell's formal definition (1997):

> A computer program is said to learn from experience E with respect to some class of tasks T and performance measure P, if its performance at tasks in T, as measured by P, improves with experience E.

Understanding through a concrete example:

- **Task T**: Email spam classification
- **Performance measure P**: Classification accuracy
- **Experience E**: Labeled email dataset (spam/non-spam)

### 1.4 Three Major Categories of Machine Learning

```
┌─────────────────────────────────────────────────────────┐
│                   Machine Learning                       │
├──────────────────┬──────────────────┬──────────────────┤
│   Supervised     │   Unsupervised   │  Reinforcement   │
│   Learning       │   Learning       │  Learning        │
├──────────────────┼──────────────────┼──────────────────┤
│ Input + Labels   │ Input only       │ State + Reward   │
│                  │                  │                  │
│ ■ Classification:│ ■ Clustering:    │ ■ Policy         │
│   Spam detection │   Customer       │   Learning:      │
│   Image classif. │   segmentation   │   Game AI        │
│   Disease diagn. │   Gene classif.  │   Robot control  │
│                  │                  │                  │
│ ■ Regression:    │ ■ Dimensionality │ ■ Exploration    │
│   Price predict. │   Reduction:     │   & Exploitation:│
│   Temperature    │   PCA            │   AlphaGo        │
│   Sales forecast │   t-SNE          │   Self-driving   │
│                  │   UMAP           │   Ad optimization│
│                  │                  │                  │
│                  │ ■ Anomaly Det.:  │ ■ Multi-Armed    │
│                  │   Fraud detect.  │   Bandit:        │
│                  │   Manufacturing  │   A/B test       │
│                  │   quality control│   optimization   │
└──────────────────┴──────────────────┴──────────────────┘
```

---

## 2. Supervised Learning

Supervised learning is an approach that learns a prediction function f: X → y from pairs of input X and correct labels y, to make predictions on unseen data. It is the most widely used type of machine learning.

### 2.1 Classification

Classification is the task of assigning input data to one of several discrete categories.

**Binary classification**: Output is 2 classes (e.g., spam/non-spam, positive/negative)
**Multi-class classification**: Output is 3 or more classes (e.g., classifying images as cat/dog/bird/fish)

#### 2.1.1 Logistic Regression

Despite the name "regression," this is actually a classification algorithm. It transforms the output of a linear combination into a probability between 0 and 1 using the sigmoid function.

```
Model: P(y=1|x) = σ(wᵀx + b) = 1 / (1 + e^(-(wᵀx + b)))

Shape of the sigmoid function:

  P(y=1)
  1.0 │                    ●●●●●
      │                 ●●●
  0.5 │- - - - - - - ●- - - - - -
      │            ●●●
  0.0 │●●●●●●●●●
      └──────────────────────── z = wᵀx + b
     -5                0                5

  Large z → P(y=1) ≈ 1 (predicted positive)
  Small z → P(y=1) ≈ 0 (predicted negative)
  z = 0   → P(y=1) = 0.5 (decision boundary)
```

The **Cross-Entropy Loss** is used as the loss function:

```
L = -1/n Σ [yᵢ log(ŷᵢ) + (1-yᵢ) log(1-ŷᵢ)]

  When yᵢ=1: L = -log(ŷᵢ)  → Loss decreases as ŷᵢ approaches 1
  When yᵢ=0: L = -log(1-ŷᵢ) → Loss decreases as ŷᵢ approaches 0
```

**Code Example 1: Logistic Regression with scikit-learn**

```python
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.datasets import load_breast_cancer

# Load the breast cancer diagnosis dataset
data = load_breast_cancer()
X, y = data.data, data.target  # 569 samples, 30 features, 2 classes

# Split into training and test data (80:20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train the model
model = LogisticRegression(max_iter=10000, random_state=42)
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# Evaluate
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred,
                            target_names=data.target_names))

# Example output:
# Confusion Matrix:
# [[39  4]
#  [ 1 70]]
#
# Classification Report:
#               precision    recall  f1-score   support
#    malignant       0.97      0.91      0.94        43
#       benign       0.95      0.99      0.97        71
#     accuracy                           0.96       114
```

#### 2.1.2 Support Vector Machine (SVM)

SVM is an algorithm that finds the hyperplane that **maximizes the margin (distance to the boundary)** between classes.

```
Intuition behind margin maximization:

    ●                    ●
      ●    Margin           ●   Margin
        ● |←──→|            ● |←──────→|
  --------+--------    ------+--------+------
          |              Support Vectors
        ○ |←──→|            ○ |←──────→|
      ○                   ○
    ○                    ○

    Narrow margin           Wide margin (what SVM aims for)
    → Low generalization    → High generalization
```

**Kernel Trick**: For data that is not linearly separable, a kernel function maps the data to a higher-dimensional space where linear separation becomes possible.

| Kernel | Formula | Use Case |
|--------|---------|----------|
| Linear | K(x,z) = xᵀz | When linearly separable |
| RBF (Gaussian) | K(x,z) = exp(-γ‖x-z‖²) | General purpose (recommended default) |
| Polynomial | K(x,z) = (γxᵀz + r)^d | Considering feature interactions |

### 2.2 Regression

Regression is the task of predicting continuous values.

#### 2.2.1 Linear Regression

The simplest and most fundamental regression algorithm.

```
Simple linear regression:
  Model: y = wx + b
  w: weight (slope), b: bias (intercept)

  Goal: Minimize the difference (loss) between predicted and actual values
  Loss function: MSE = (1/n) Σ(yᵢ - ŷᵢ)²

  Parameter update via gradient descent:
  w ← w - α × ∂L/∂w = w - α × (-2/n) Σ xᵢ(yᵢ - ŷᵢ)
  b ← b - α × ∂L/∂b = b - α × (-2/n) Σ (yᵢ - ŷᵢ)
  α: learning rate (step size)

  y │       ●
    │     ●   ━━━━━━ Regression line (y = wx + b)
    │   ●  ●╱
    │    ╱●
    │  ╱●
    │╱●
    └──────────── x

  Multiple linear regression:
  y = w₁x₁ + w₂x₂ + ... + wₙxₙ + b
  Matrix notation: y = Xw + b
  Analytical solution via normal equation: w = (XᵀX)⁻¹Xᵀy
```

**Code Example 2: Implementing Gradient Descent from Scratch with NumPy**

```python
import numpy as np

# === Linear regression implementation with gradient descent ===

# Generate sample data
np.random.seed(42)
X = 2 * np.random.rand(100, 1)        # Uniform distribution from 0 to 2
y = 4 + 3 * X + np.random.randn(100, 1) * 0.5  # y = 4 + 3x + noise

# Initialize parameters
w = np.random.randn(1, 1)  # Weight
b = np.zeros((1, 1))       # Bias

# Hyperparameters
learning_rate = 0.1
n_epochs = 100
m = len(X)  # Number of samples

# Training loop
losses = []
for epoch in range(n_epochs):
    # Forward pass: compute predictions
    y_pred = X @ w + b

    # Compute loss (MSE)
    loss = np.mean((y - y_pred) ** 2)
    losses.append(loss)

    # Compute gradients
    dw = (-2 / m) * (X.T @ (y - y_pred))   # ∂L/∂w
    db = (-2 / m) * np.sum(y - y_pred)      # ∂L/∂b

    # Update parameters
    w -= learning_rate * dw
    b -= learning_rate * db

    if epoch % 20 == 0:
        print(f"Epoch {epoch:3d}: Loss = {loss:.4f}, "
              f"w = {w[0,0]:.4f}, b = {b[0,0]:.4f}")

# Display final results
print(f"\nLearned model: y = {w[0,0]:.4f}x + {b[0,0]:.4f}")
print(f"True model: y = 3.0000x + 4.0000")

# Example output:
# Epoch   0: Loss = 18.2341, w = 1.2345, b = 0.3456
# Epoch  20: Loss = 0.3012, w = 2.8901, b = 3.9123
# Epoch  40: Loss = 0.2534, w = 2.9678, b = 4.0234
# Epoch  60: Loss = 0.2498, w = 2.9890, b = 4.0312
# Epoch  80: Loss = 0.2495, w = 2.9956, b = 4.0345
#
# Learned model: y = 2.9956x + 4.0345
# True model: y = 3.0000x + 4.0000
```

#### 2.2.2 Polynomial Regression and Regularization

Linear regression has limited expressiveness. When data has a nonlinear relationship, polynomial features can be introduced.

```
Polynomial regression:
  Degree 1: y = w₁x + b
  Degree 2: y = w₁x + w₂x² + b
  Degree 3: y = w₁x + w₂x² + w₃x³ + b

  However, increasing the degree too much causes overfitting:

  Degree 1 (underfitting)  Degree 3 (appropriate)  Degree 15 (overfitting)
  y │  ──────            y │  ～～～             y │  ∿∿∿∿∿∿
    │ ●   ●               │ ●～～●               │ ●∿∿∿∿●
    │●  ●                  │●  ●                  │●∿   ∿
    │  ●                   │  ●                   │  ●
    └──── x                └──── x                └──── x
  High bias              Bias/Variance           High variance
  Low variance           balanced                Low bias
```

**Regularization** prevents overfitting by adding a penalty on parameter magnitude to the loss function.

| Regularization | Loss Function | Effect |
|---------------|--------------|--------|
| L1 (Lasso) | MSE + λΣ\|wᵢ\| | Sparse solutions (some weights become 0) → Feature selection |
| L2 (Ridge) | MSE + λΣwᵢ² | Weights shrink overall → Smoother predictions |
| ElasticNet | MSE + λ₁Σ\|wᵢ\| + λ₂Σwᵢ² | Combination of L1 and L2 |

### 2.3 Decision Tree

Decision trees classify or regress data through a chain of conditional splits. They are among the most interpretable algorithms for humans.

```
Example: Loan approval decision tree

              ┌──────────────────┐
              │ Income > 50K?     │
              └───┬──────┬───────┘
              Yes │      │ No
          ┌───────┘      └───────┐
     ┌────┴──────┐         ┌─────┴────┐
     │Tenure > 3y?│         │ Reject ✗  │
     └──┬────┬───┘         └──────────┘
     Yes│    │No
    ┌───┘    └───┐
┌───┴───┐  ┌────┴─────┐
│Approve ✓│  │Debt > 20K?│
└────────┘  └──┬───┬───┘
            Yes│   │No
           ┌──┘   └──┐
      ┌────┴───┐ ┌───┴────┐
      │ Reject ✗│ │Approve ✓│
      └────────┘ └────────┘
```

**Details on Split Criteria**:

**Gini Impurity**:

```
Gini(t) = 1 - Σ p(i|t)²

Example: A node has 100 samples [Cat: 40, Dog: 60]
Gini = 1 - (0.4² + 0.6²) = 1 - (0.16 + 0.36) = 0.48

Completely pure node (all cats):
Gini = 1 - 1² = 0

Maximum impurity (50% cat, 50% dog):
Gini = 1 - (0.5² + 0.5²) = 0.5
```

**Information Gain**:

```
Entropy: H(t) = -Σ p(i|t) log₂ p(i|t)
Information Gain: IG = H(parent) - Σ (|child_k| / |parent|) × H(child_k)

Split on the feature and threshold that reduces entropy the most.
```

### 2.4 Ensemble Learning

A family of methods that combine multiple weak learners to achieve higher performance than any single learner.

#### Random Forest

```
Bagging (Bootstrap Aggregating) + Random feature selection

  Original data
  ┌────────┐
  │ Full D  │
  └──┬─┬─┬─┘
     │ │ │   Bootstrap sampling
     ▼ ▼ ▼   (with replacement)
  ┌──┐┌──┐┌──┐
  │D₁││D₂││D₃│  Each subset is trained independently
  └┬─┘└┬─┘└┬─┘
   ▼   ▼   ▼
  Tree₁ Tree₂ Tree₃  Each tree splits on a feature subset
   │   │   │
   ▼   ▼   ▼
  Cat  Dog  Cat  → Majority vote → Cat (final prediction)
```

Characteristics:
- Each tree trains on about 63.2% of data (remaining 36.8% is OOB: Out-of-Bag)
- Features are also randomly selected (typically √p, where p = total features)
- Parallelizable for fast training
- Resistant to overfitting

#### Gradient Boosting

```
Sequentially adds weak learners that correct the errors (residuals) of the previous ones

  f(x) = f₁(x) + η・f₂(x) + η・f₃(x) + ...
  η: learning rate (shrinkage)

  Step 1: Train f₁ (initial model)
  Step 2: Compute residuals r₁ = y - f₁(x)
  Step 3: Train f₂ on r₁
  Step 4: Compute residuals r₂ = y - (f₁ + η・f₂)
  Step 5: Train f₃ on r₂
  ...repeat
```

Major implementations:

| Library | Features | Primary Use |
|---------|----------|-------------|
| XGBoost | Regularized gradient boosting, parallelization | Kaggle competition standard |
| LightGBM | Histogram-based, fast and memory-efficient | Large-scale data |
| CatBoost | Automatic categorical variable handling, ordered boosting | Data with many categorical variables |

**Code Example 3: Comparing Random Forest and Gradient Boosting with scikit-learn**

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier
)
from sklearn.tree import DecisionTreeClassifier

# Generate synthetic data (1000 samples, 20 features, 2 classes)
X, y = make_classification(
    n_samples=1000, n_features=20,
    n_informative=10, n_redundant=5,
    random_state=42
)

# Compare three models
models = {
    "Decision Tree (single)": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(
        n_estimators=100, random_state=42
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=100, learning_rate=0.1, random_state=42
    ),
}

print("Accuracy comparison with 5-fold cross-validation:")
print("-" * 50)
for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    print(f"{name:20s}: {scores.mean():.4f} (+/- {scores.std():.4f})")

# Example output:
# Accuracy comparison with 5-fold cross-validation:
# --------------------------------------------------
# Decision Tree (single) : 0.8560 (+/- 0.0234)
# Random Forest           : 0.9320 (+/- 0.0187)
# Gradient Boosting       : 0.9420 (+/- 0.0156)
```

### 2.5 k-Nearest Neighbors (kNN)

kNN is a representative example of "lazy learning," where there is virtually no training phase. At prediction time, it finds the k closest training samples to the input and predicts by majority vote (classification) or average (regression).

```
When k=3:

      ○ ○
    ○   ★ ← New data point
      ● ○
    ●
  ●   ●

  3 closest points to ★: ○, ○, ○ → Predicted as class ○

Choice of k:
  k too small → Sensitive to noise (overfitting)
  k too large → Decision boundary becomes overly smooth (underfitting)
  → Select optimal k using cross-validation
```

---

## 3. Unsupervised Learning

Unsupervised learning is a set of methods for discovering structure and patterns in data without correct labels.

### 3.1 Clustering

#### K-Means

The most widely used clustering algorithm.

```
K-Means algorithm:

  Step 1: Randomly place K centroids
  Step 2: Assign each data point to the nearest centroid's cluster
  Step 3: Recompute each centroid (mean of the cluster)
  Step 4: Repeat Steps 2-3 until convergence

  Iteration progression (K=3):

  Initial state        Iteration 1        Iteration 2 (converged)
  ・・ ★₁              ●● ★₁              ●● ★₁
  ・ ・                ● ●                ● ●
  ・  ★₂・            ○  ★₂○             ○  ★₂○
   ・・                 ○○                  ○○
  ・ ★₃ ・            △ ★₃ △             △ ★₃ △
   ・ ・                △ △                 △ △

  ★: Centroid position
  Each symbol: Data point assigned to that cluster
```

**Determining K: The Elbow Method**

```
Plot SSE (Sum of Squared Errors) for each value of K:

  SSE
   │╲
   │  ╲
   │   ╲___       ← Elbow point (K=3)
   │       ╲___
   │           ╲___
   └──────────────── K
   1  2  3  4  5  6

  The point where SSE stops decreasing sharply (the elbow) is the optimal K
```

Limitations of K-Means:
- The number of clusters K must be specified in advance
- Assumes spherical clusters (struggles with non-spherical shapes)
- Sensitive to initialization (improved with K-Means++)
- Sensitive to outliers

#### DBSCAN (Density-Based Spatial Clustering)

A density-based clustering method that addresses K-Means' weaknesses.

```
DBSCAN parameters:
  - ε (eps): Neighborhood radius
  - MinPts: Minimum number of samples to be classified as a core point

  Point classification:
  ● Core point: Has MinPts or more points within ε
  ◐ Border point: Within ε of a core point but fewer than MinPts
  ○ Noise: Does not belong to any cluster

  Advantages:
  - Automatically determines the number of clusters
  - Detects clusters of arbitrary shape
  - Automatically excludes outliers

  Disadvantages:
  - Struggles with clusters of varying density
  - Depends on the ε and MinPts settings
```

### 3.2 Dimensionality Reduction

Compresses high-dimensional data to lower dimensions for visualization and computational efficiency.

#### Principal Component Analysis (PCA)

```
Intuition behind PCA:

  Find the direction (principal component) that maximizes data variance

  2D → 1D example:
  y │    ●  ●
    │  ●  ●      First principal component (max variance direction)
    │ ● ●    ╱
    │●  ●  ╱
    │ ●  ╱
    │  ╱
    └──────── x

  Algorithm:
  1. Standardize data (mean 0, variance 1)
  2. Compute covariance matrix
  3. Eigenvalue decomposition (largest eigenvalues = principal components)
  4. Project onto top k eigenvectors

  Explained variance ratio:
  First principal component explains 70% of total variance
  Second principal component explains 20% of total variance
  → 90% of information retained in 2 dimensions
```

#### t-SNE and UMAP

| Method | Principle | Speed | Global Structure | Primary Use |
|--------|-----------|-------|-----------------|-------------|
| PCA | Linear projection (variance maximization) | Fast | Preserved | Preprocessing, feature extraction |
| t-SNE | KL divergence minimization between distributions | Slow | Weak | Visualization (2D/3D) |
| UMAP | Topological data analysis | Medium | Strong | Visualization + downstream tasks |

### 3.3 Anomaly Detection

Learns patterns of normal data and detects data that deviates from them.

Main methods:
- **Isolation Forest**: Identifies points that are "quickly isolated" by random splits as anomalies
- **One-Class SVM**: Learns the boundary of normal data
- **Autoencoder**: Identifies data with high reconstruction error as anomalies
- **Statistical methods**: Z-score, IQR (Interquartile Range)

Applications:
- Credit card fraud detection
- Network intrusion detection
- Manufacturing quality control
- System monitoring (CPU/memory anomaly detection)

---

## 4. Overview of Reinforcement Learning

Reinforcement learning is a framework where an agent learns a behavioral policy that maximizes cumulative reward through interaction with an environment.

```
Reinforcement learning framework:

  ┌────────────┐         Action aₜ          ┌────────────┐
  │            │───────────────────────────→│            │
  │   Agent    │                             │ Environment│
  │  (Policy π)│←───────────────────────────│ (State      │
  │            │    State sₜ₊₁ + Reward rₜ  │ transition) │
  └────────────┘                             └────────────┘

  Markov Decision Process (MDP):
  - S: Set of states
  - A: Set of actions
  - P(s'|s,a): State transition probability
  - R(s,a): Reward function
  - γ: Discount factor (0 < γ ≤ 1, discounts future rewards)

  Goal: Learn the policy π* that maximizes cumulative discounted reward G = Σ γᵗ rₜ
```

**Exploration-Exploitation Trade-off**:

- **Exploration**: Try unknown actions → Possibility of discovering better policies in the long run
- **Exploitation**: Take the action believed to be best → Maximize short-term reward

Representative strategies:
- **ε-greedy**: Explore (random action) with probability ε, exploit (best action) with probability 1-ε
- **UCB (Upper Confidence Bound)**: Preferentially explore actions with high uncertainty
- **Boltzmann exploration**: Select actions with probability proportional to action values

**Major Algorithms**:

| Algorithm | Type | Features |
|-----------|------|----------|
| Q-Learning | Value-based | Learns Q(s,a) table. Off-policy |
| SARSA | Value-based | Learns Q(s,a) table. On-policy |
| DQN | Value-based + Deep Learning | Approximates Q function with NN. Atari games |
| Policy Gradient | Policy-based | Directly learns policy π(a\|s) |
| PPO | Actor-Critic | Stable training. Used in ChatGPT's RLHF |
| SAC | Actor-Critic | Maximum entropy. Robot control |

---

## 5. Fundamentals of Neural Networks

### 5.1 Perceptron (Single Neuron)

```
Input signals are weighted, summed, and passed through an activation function:

  x₁ ──w₁──┐
             │
  x₂ ──w₂──┼──→ Σ ──→ σ(z) ──→ y
             │        ↑
  x₃ ──w₃──┘        b (bias)

  z = w₁x₁ + w₂x₂ + w₃x₃ + b  (weighted sum)
  y = σ(z)                       (activation function applied)
```

### 5.2 Activation Functions in Detail

```
┌──────────────────────────────────────────────────────────┐
│ Name        │ Definition            │ Range     │ Feature │
├──────────────────────────────────────────────────────────┤
│ Sigmoid     │ 1/(1+e⁻ᶻ)            │ (0, 1)   │ Prob.   │
│             │                       │          │ interp. │
│ tanh        │ (eᶻ-e⁻ᶻ)/(eᶻ+e⁻ᶻ)   │ (-1, 1)  │ Zero-   │
│             │                       │          │ centered│
│ ReLU        │ max(0, z)             │ [0, ∞)   │ Modern  │
│             │                       │          │ standard│
│ Leaky ReLU  │ max(αz, z), α=0.01   │ (-∞, ∞)  │ Avoids  │
│             │                       │          │ dead    │
│             │                       │          │ neurons │
│ GELU        │ z・Φ(z)               │ (-∞, ∞)  │Transformer│
│ Softmax     │ eᶻⁱ/Σeᶻʲ             │ (0, 1)   │ Multi-  │
│             │                       │ sum=1    │ class   │
│             │                       │          │ output  │
└──────────────────────────────────────────────────────────┘

Shape of each activation function:

  Sigmoid:               ReLU:                Leaky ReLU:
  1 │      ●●●●          │    ╱              │    ╱
    │    ●●               │   ╱               │   ╱
 .5 │  ●                  │  ╱                │  ╱
    │●●                   │ ╱                 │╱
  0 │●●●●                 │╱                ╱╱│
    └──────── z           └──────── z      ╱  └──── z
```

**Why Are Activation Functions Necessary?**

Without activation functions, a multi-layer network is merely a composition of linear transformations, equivalent to a single-layer linear model:

```
Without activation functions:
  y = W₂(W₁x + b₁) + b₂ = (W₂W₁)x + (W₂b₁ + b₂) = W'x + b'
  → No matter how many layers, it remains a linear model

With activation functions:
  y = W₂ σ(W₁x + b₁) + b₂
  → Nonlinear transformations enable approximation of complex functions
  → Universal Approximation Theorem
```

### 5.3 Multi-Layer Neural Networks

```
Fully Connected Neural Network:

  Input Layer   Hidden Layer 1  Hidden Layer 2   Output Layer
  (Features)    (128 units)     (64 units)       (# classes)

  ○──┐    ┌──○──┐    ┌──○──┐    ┌──○
  ○──┼────┼──○──┼────┼──○──┼────┼──○
  ○──┼────┼──○──┼────┼──○──┼────┤
  ○──┘    └──○──┘    └──○──┘    └──○

  Each connection has a weight w, and each unit has a bias b
  → Number of parameters: (input×128) + (128×64) + (64×output) + biases
```

### 5.4 Backpropagation

Neural network training is achieved through **backpropagation**. Using the chain rule, gradients are efficiently computed from the output layer back to the input layer.

```
Backpropagation procedure:

  1. Forward Pass:
     Input x → Hidden h = σ(W₁x + b₁) → Output ŷ = σ(W₂h + b₂)

  2. Loss computation:
     L = Loss(y, ŷ)

  3. Backward Pass:
     ∂L/∂W₂ = ∂L/∂ŷ × ∂ŷ/∂z₂ × ∂z₂/∂W₂   (output layer gradient)
     ∂L/∂W₁ = ∂L/∂ŷ × ∂ŷ/∂z₂ × ∂z₂/∂h × ∂h/∂z₁ × ∂z₁/∂W₁
                                               (hidden layer gradient)
     ↑ Decomposed and computed using the Chain Rule

  4. Parameter update:
     W₁ ← W₁ - α × ∂L/∂W₁
     W₂ ← W₂ - α × ∂L/∂W₂
     b₁ ← b₁ - α × ∂L/∂b₁
     b₂ ← b₂ - α × ∂L/∂b₂
```

### 5.5 Optimization Algorithms

Simple gradient descent (SGD) has challenges: setting the learning rate is difficult and it can get stuck in local optima. Various optimization algorithms have been developed to address this.

| Optimizer | Features | When to Use |
|-----------|----------|-------------|
| SGD | Basic gradient descent | Theoretically converges to best solution |
| SGD + Momentum | Uses inertia to reduce oscillation | Improved SGD |
| AdaGrad | Adapts learning rate per parameter | Sparse data |
| RMSprop | Fixes AdaGrad's learning rate decay issue | Effective for RNN training |
| Adam | Momentum + RMSprop | Most widely used default |
| AdamW | Adam + Weight Decay | Standard for Transformer training |

**Code Example 4: Implementing a Neural Network from Scratch with NumPy**

```python
import numpy as np

class SimpleNeuralNetwork:
    """2-layer neural network (1 hidden layer)"""

    def __init__(self, input_size, hidden_size, output_size):
        # Xavier initialization
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / hidden_size)
        self.b2 = np.zeros((1, output_size))

    def relu(self, z):
        return np.maximum(0, z)

    def relu_derivative(self, z):
        return (z > 0).astype(float)

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    def forward(self, X):
        """Forward pass"""
        self.z1 = X @ self.W1 + self.b1
        self.a1 = self.relu(self.z1)           # Hidden layer: ReLU
        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = self.sigmoid(self.z2)         # Output layer: Sigmoid
        return self.a2

    def compute_loss(self, y, y_pred):
        """Binary cross-entropy loss"""
        m = y.shape[0]
        epsilon = 1e-8  # Prevent log(0)
        loss = -np.mean(
            y * np.log(y_pred + epsilon) +
            (1 - y) * np.log(1 - y_pred + epsilon)
        )
        return loss

    def backward(self, X, y, y_pred, learning_rate=0.01):
        """Backward pass"""
        m = X.shape[0]

        # Output layer gradients
        dz2 = y_pred - y                        # (m, output_size)
        dW2 = (self.a1.T @ dz2) / m
        db2 = np.sum(dz2, axis=0, keepdims=True) / m

        # Hidden layer gradients (chain rule)
        da1 = dz2 @ self.W2.T
        dz1 = da1 * self.relu_derivative(self.z1)
        dW1 = (X.T @ dz1) / m
        db1 = np.sum(dz1, axis=0, keepdims=True) / m

        # Parameter update
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1

    def train(self, X, y, epochs=1000, learning_rate=0.01):
        """Training loop"""
        for epoch in range(epochs):
            y_pred = self.forward(X)
            loss = self.compute_loss(y, y_pred)
            self.backward(X, y, y_pred, learning_rate)

            if epoch % 200 == 0:
                accuracy = np.mean((y_pred > 0.5).astype(int) == y)
                print(f"Epoch {epoch:4d}: Loss={loss:.4f}, "
                      f"Accuracy={accuracy:.4f}")


# --- Learning the XOR problem (not linearly separable) ---
X = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([[0], [1], [1], [0]])  # XOR

nn = SimpleNeuralNetwork(input_size=2, hidden_size=8, output_size=1)
nn.train(X, y, epochs=2000, learning_rate=0.5)

# Predictions
predictions = nn.forward(X)
print("\nPrediction results:")
for i in range(len(X)):
    print(f"  Input: {X[i]} → Prediction: {predictions[i,0]:.4f} "
          f"(True value: {y[i,0]})")

# Example output:
# Epoch    0: Loss=0.6931, Accuracy=0.5000
# Epoch  200: Loss=0.5234, Accuracy=0.7500
# Epoch  400: Loss=0.1456, Accuracy=1.0000
# Epoch  600: Loss=0.0423, Accuracy=1.0000
# ...
# Prediction results:
#   Input: [0 0] → Prediction: 0.0312 (True value: 0)
#   Input: [0 1] → Prediction: 0.9678 (True value: 1)
#   Input: [1 0] → Prediction: 0.9701 (True value: 1)
#   Input: [1 1] → Prediction: 0.0298 (True value: 0)
```

---

## 6. Deep Learning and Architectures

Deep learning uses neural networks with many hidden layers. It saw explosive growth after AlexNet's overwhelming victory in the ImageNet competition in 2012.

### 6.1 Three Key Elements of Deep Learning

```
Why did deep learning explode in 2012:

  ┌──────────────────────────────────────────────┐
  │ 1. Massive amounts of data                    │
  │    ImageNet: 14 million labeled images        │
  │    Internet proliferation made data collection │
  │    easy                                       │
  │                                               │
  │ 2. Improved computational power               │
  │    GPU: Optimal for parallel matrix operations│
  │    NVIDIA CUDA → 10-100x faster training      │
  │                                               │
  │ 3. Algorithm improvements                     │
  │    ReLU: Mitigated vanishing gradient problem │
  │    Dropout: Prevented overfitting             │
  │    BatchNorm: Stabilized and sped up training │
  │    Residual connections: Enabled very deep     │
  │    networks                                   │
  └──────────────────────────────────────────────┘
```

### 6.2 CNN (Convolutional Neural Network)

An architecture specialized for image recognition. Extracts local features hierarchically.

```
CNN architecture:

  ┌────────┐  ┌────────┐  ┌────────┐  ┌──────┐  ┌────┐
  │Input   │→│Convolu- │→│Pooling  │→│Convo- │→│Fully│→ Classification
  │Image   │  │tion     │  │(shrink) │  │lution │  │Con- │
  │(HxWxC) │  │+ReLU    │  │         │  │+ReLU  │  │nect │
  │        │  │         │  │         │  │       │  │+Soft│
  │        │  │         │  │         │  │       │  │max  │
  └────────┘  └────────┘  └────────┘  └──────┘  └────┘

  Convolution operation:
  Input image (5x5)    Filter (3x3)       Output (3x3)
  ┌─┬─┬─┬─┬─┐     ┌─┬─┬─┐           ┌─┬─┬─┐
  │1│0│1│0│1│     │1│0│1│           │4│3│4│
  ├─┼─┼─┼─┼─┤     ├─┼─┼─┤    *     ├─┼─┼─┤
  │0│1│0│1│0│  ×  │0│1│0│   =     │2│4│3│
  ├─┼─┼─┼─┼─┤     ├─┼─┼─┤          ├─┼─┼─┤
  │1│0│1│0│1│     │1│0│1│          │4│3│4│
  ├─┼─┼─┼─┼─┤     └─┴─┴─┘          └─┴─┴─┘
  │0│1│0│1│0│
  ├─┼─┼─┼─┼─┤
  │1│0│1│0│1│
  └─┴─┴─┴─┴─┘

  Deeper layers learn more abstract features:
  ┌─────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐
  │ Layer 1  │→│  Layer 2  │→│  Layer 3  │→│  Layer 4  │
  │Edges,    │  │Textures,  │  │Parts     │  │Whole      │
  │colors    │  │patterns   │  │(eyes,    │  │objects    │
  │          │  │           │  │nose,     │  │           │
  │          │  │           │  │wheels)   │  │           │
  └─────────┘  └──────────┘  └──────────┘  └──────────┘
```

Evolution of major CNN architectures:

| Model | Year | Layers | ImageNet Top-5 | Innovation |
|-------|------|--------|---------------|------------|
| AlexNet | 2012 | 8 | 15.3% | GPU training, ReLU, Dropout |
| VGGNet | 2014 | 19 | 7.3% | Stacking small (3x3) filters |
| GoogLeNet | 2014 | 22 | 6.7% | Inception module |
| ResNet | 2015 | 152 | 3.6% | Residual connections (Skip Connection) |
| EfficientNet | 2019 | - | 2.9% | Compound scaling of width, depth, resolution |

### 6.3 RNN / LSTM / GRU

Architectures for processing sequential data (text, audio, time series).

```
RNN (Recurrent Neural Network):

  ┌────┐  ┌────┐  ┌────┐  ┌────┐
  │ h₁ │→│ h₂ │→│ h₃ │→│ h₄ │→ Output
  └──┬─┘  └──┬─┘  └──┬─┘  └──┬─┘
     ↑       ↑       ↑       ↑
    x₁      x₂      x₃      x₄
   (I)    (love)  (machine)(learning)

  hₜ = σ(Wₕhₜ₋₁ + Wₓxₜ + b)

  Problem: Vanishing gradients → Forgets past information in long sequences

LSTM (Long Short-Term Memory):
  Controls long-term and short-term memory through gating mechanisms

  ┌─────────────────────────────────────┐
  │ LSTM Cell                            │
  │                                      │
  │  Forget gate: fₜ = σ(Wf[hₜ₋₁,xₜ]+bf)│ ← What to forget
  │  Input gate:  iₜ = σ(Wi[hₜ₋₁,xₜ]+bi)│ ← What to remember
  │  Output gate: oₜ = σ(Wo[hₜ₋₁,xₜ]+bo)│ ← What to output
  │                                      │
  │  Cell state: Cₜ = fₜ⊙Cₜ₋₁ + iₜ⊙C̃ₜ  │ ← Long-term memory
  │  Hidden state: hₜ = oₜ⊙tanh(Cₜ)     │ ← Short-term memory
  └─────────────────────────────────────┘

GRU (Gated Recurrent Unit):
  Simplified version of LSTM. Fewer parameters and faster training.
  Has 2 gates (reset gate, update gate).
```

### 6.4 Transformer

The foundational architecture of modern AI, proposed in the 2017 paper "Attention Is All You Need."

```
Transformer architecture:

  ┌──────────────────────────────────────┐
  │           Transformer                 │
  │                                       │
  │  ┌─────────────┐  ┌─────────────┐    │
  │  │  Encoder     │  │   Decoder    │   │
  │  │              │  │              │   │
  │  │ Self-Attn    │  │ Masked       │   │
  │  │     ↓        │  │ Self-Attn    │   │
  │  │ Feed Forward │  │     ↓        │   │
  │  │     ↓        │  │ Cross-Attn   │   │
  │  │ (×N layers)  │  │     ↓        │   │
  │  │              │  │ Feed Forward │   │
  │  └─────────────┘  │     ↓        │   │
  │                    │ (×N layers)  │   │
  │                    └─────────────┘    │
  └──────────────────────────────────────┘

Self-Attention mechanism:

  Input: "The cat sat on the mat"

  For each token, compute Query(Q), Key(K), Value(V):
  Q = XWq, K = XWk, V = XWv

  Attention(Q,K,V) = softmax(QKᵀ / √dₖ) V

  Attention weights for "cat":
   The  cat  sat  on  the  mat
  [0.05 0.40 0.15 0.05 0.05 0.30]
        ↑                    ↑
   Attends to itself     Also attends to "mat"

  → Unlike RNN, all tokens can be processed in parallel
  → Long-range dependencies are captured directly
```

**Multi-Head Attention**:

Computes multiple attention heads in parallel to capture attention from different perspectives.

```
MultiHead(Q,K,V) = Concat(head₁, ..., headₕ) Wᴼ
  where headᵢ = Attention(QWᵢQ, KWᵢK, VWᵢV)

  Head 1: Focuses on syntactic relations (subject-verb)
  Head 2: Focuses on semantic relations (synonyms)
  Head 3: Focuses on positional relations (adjacent words)
  → Multiple perspectives are integrated
```

Model families derived from Transformer:

| Model | Structure | Training Method | Examples |
|-------|-----------|-----------------|----------|
| Encoder-only | Encoder only | Masked language modeling | BERT, RoBERTa |
| Decoder-only | Decoder only | Next-token prediction | GPT, Claude, LLaMA |
| Encoder-Decoder | Both | Seq2Seq | T5, BART |

---

## 7. System of Evaluation Metrics

Correctly evaluating model performance is one of the most critical skills that determines the success or failure of a machine learning project.

### 7.1 Classification Metrics

#### Confusion Matrix

```
                    Predicted
                 Positive  Negative
           ┌────────┬────────┐
  Actual P │  TP     │  FN     │  ← False Negative (miss)
           ├────────┼────────┤
       N   │  FP     │  TN     │  ← False Positive (false alarm)
           └────────┴────────┘
              ↑
        False Positive (false accusation)

  TP (True Positive):  Correctly predicted positive as positive
  FP (False Positive): Incorrectly predicted negative as positive (Type I error)
  FN (False Negative): Incorrectly predicted positive as negative (Type II error)
  TN (True Negative):  Correctly predicted negative as negative
```

#### Key Metrics

```
Accuracy: (TP + TN) / (TP + FP + FN + TN)
  → Proportion correctly classified out of the total
  → Dangerous with imbalanced data (if 99% is negative, always predicting negative gives Acc=99%)

Precision: TP / (TP + FP)
  → Of those predicted as positive, what proportion was actually positive
  → Important when you want to "reduce false accusations"
  → Spam filter: Don't want to misclassify legitimate emails as spam

Recall (Sensitivity): TP / (TP + FN)
  → Of actual positives, what proportion was correctly detected
  → Important when you want to "reduce misses"
  → Cancer screening: Don't want to miss cancer patients

F1 Score: 2 × (Precision × Recall) / (Precision + Recall)
  → Harmonic mean of Precision and Recall
  → Used instead of Accuracy for imbalanced data
```

**Precision-Recall Trade-off**:

```
Changing the threshold changes the balance between Precision and Recall:

  Precision
  1.0│╲
     │  ╲
     │    ╲
  0.5│     ╲
     │       ╲
  0.0│        ╲
     └──────────
    0.0  0.5  1.0
         Recall

  Higher threshold → Precision↑, Recall↓ (detect only high-confidence cases)
  Lower threshold  → Precision↓, Recall↑ (detect more but increase false positives)
```

#### AUC-ROC Curve

```
ROC Curve: Plot of TPR vs FPR as the threshold varies

  TPR (True Positive Rate = Recall)
  1.0│    ●●●●●●●
     │  ●●
     │ ●         ← Good model (AUC ≈ 0.9)
     │●
  0.5│─ ─ ─ ─ ─ ← Random classifier (AUC = 0.5)
     │╱
     │
  0.0│
     └──────────
    0.0  0.5  1.0
     FPR (False Positive Rate)

  AUC (Area Under the Curve): Area under the ROC curve
  - AUC = 1.0: Perfect classification
  - AUC = 0.5: Random classification (model has no learning ability)
  - AUC < 0.5: Worse than random (labels may be inverted)
```

### 7.2 Regression Metrics

| Metric | Definition | Characteristics |
|--------|-----------|-----------------|
| MSE | (1/n)Σ(yᵢ-ŷᵢ)² | Penalizes large errors heavily |
| RMSE | √MSE | Same scale as the original units |
| MAE | (1/n)Σ\|yᵢ-ŷᵢ\| | Robust to outliers |
| R² | 1 - Σ(yᵢ-ŷᵢ)² / Σ(yᵢ-ȳ)² | Proportion of variance explained by the model (closer to 1 is better) |
| MAPE | (100/n)Σ\|yᵢ-ŷᵢ\|/\|yᵢ\| | Percentage error (easy to interpret) |

### 7.3 Choosing the Right Metric

```
Selecting metrics based on the task:

  ┌──────────────────────────────────────────────────┐
  │ Task                │ Recommended   │ Reason      │
  ├──────────────────────────────────────────────────┤
  │ Balanced binary     │ Accuracy, F1  │ Straightfor-│
  │ classification      │               │ ward eval   │
  │ Imbalanced binary   │ F1, AUC-ROC   │ Acc is      │
  │ classification      │               │ unsuitable  │
  │ Medical diagnosis   │ Recall-focus  │ Prevent     │
  │                     │               │ misses      │
  │ Spam detection      │ Precision-    │ Prevent     │
  │                     │ focus         │ false alarms│
  │ Multi-class classif.│ Macro-F1      │ Equal across│
  │                     │               │ classes     │
  │ Ranking             │ NDCG, MAP     │ Evaluates   │
  │                     │               │ ordering    │
  │ Regression (general)│ RMSE, R²      │ Standard    │
  │                     │               │ metrics     │
  │ Regression (with    │ MAE           │ Robustness- │
  │ outliers)           │               │ focused     │
  └──────────────────────────────────────────────────┘
```

---

## 8. Overfitting and Regularization

### 8.1 Bias-Variance Trade-off

```
Model prediction error = Bias² + Variance + Irreducible Error

  Error│
      │╲ Variance              ╱ Bias²
      │  ╲                   ╱
      │    ╲    ╱─────╲    ╱  ← Total error
      │      ╲╱         ╲
      │       │
      └───────┼──────────── Model complexity
            Optimal point

  Bias: Strength of model assumptions (too simple → high)
  Variance: Sensitivity to data (too complex → high)
  → Choose the model that balances both
```

### 8.2 Diagnosing Overfitting

```
Learning Curve:

  Loss│
     │ ╲  Training loss
     │   ╲─────────────── ← Overfitting: gap between training and test
     │
     │   ╱─────────────── ← Test loss
     │  ╱
     │╱
     └──────────────────── Epoch

  Signs of overfitting:
  - Training loss keeps decreasing, but test loss starts increasing
  - Training accuracy >> Test accuracy (large gap)
```

### 8.3 Systematic Approaches to Preventing Overfitting

```
Overview of overfitting countermeasures:

  ┌─────────────────────────────────────────────┐
  │  1. Increase data volume                     │
  │     - Data collection                        │
  │     - Data augmentation (rotation, flip,     │
  │       crop for images)                       │
  │                                              │
  │  2. Limit model complexity                   │
  │     - L1/L2 regularization                   │
  │     - Dropout                                │
  │     - Reduce model size                      │
  │                                              │
  │  3. Control the training process             │
  │     - Early Stopping                         │
  │     - Learning rate scheduling               │
  │     - Batch Normalization                    │
  │                                              │
  │  4. Rigorous evaluation                      │
  │     - Cross Validation                       │
  │     - Holdout method                         │
  └─────────────────────────────────────────────┘
```

**How Dropout Works**:

```
During training, randomly "deactivate" neurons:

  Normal network:          After Dropout (p=0.5):
  ○──○──○──○              ○──○  ×  ○
  ○──○──○──○              ×  ○──○  ×
  ○──○──○──○              ○  ×  ○──○

  × = Deactivated neuron (output set to 0)

  Effects:
  - Prevents dependence on specific neurons
  - Ensemble effect (learning different sub-networks)
  - At inference, all neurons are used (output multiplied by p)
```

**Cross Validation**:

```
K-Fold Cross Validation:

  For K=5:

  Fold 1: [Test | Train | Train | Train | Train ] → Score₁
  Fold 2: [Train | Test | Train | Train | Train ] → Score₂
  Fold 3: [Train | Train | Test | Train | Train ] → Score₃
  Fold 4: [Train | Train | Train | Test | Train ] → Score₄
  Fold 5: [Train | Train | Train | Train | Test ] → Score₅

  Final score = (Score₁ + ... + Score₅) / 5

  Advantages:
  - All data is used for both training and testing
  - Can estimate performance variance
  - Especially effective when data is limited
```

---

## 9. Feature Engineering

Feature engineering is the process of designing and extracting features from raw data that are easy for a model to learn. It embodies the principle "data quality = model quality (Garbage In, Garbage Out)."

### 9.1 Preprocessing Numerical Features

```
Standardization:
  z = (x - μ) / σ  →  Mean 0, standard deviation 1
  → Effective for gradient descent-based algorithms

Normalization (Min-Max):
  x' = (x - x_min) / (x_max - x_min)  →  Scales to 0-1
  → Effective for neural networks and image data

Log transformation:
  x' = log(1 + x)
  → Makes right-skewed distributions (income, population) closer to normal
```

### 9.2 Encoding Categorical Features

| Method | Description | Use Case |
|--------|-------------|----------|
| Label Encoding | Convert categories to integers (A=0, B=1, C=2) | Tree-based models |
| One-Hot Encoding | Convert categories to binary vectors | Linear models, NNs |
| Target Encoding | Convert categories to target variable mean | High cardinality |
| Ordinal Encoding | Integer encoding preserving ordinal relationships | Ordinal categories (low/medium/high) |

### 9.3 Handling Missing Values

```
Missing value strategies:

  1. Deletion:
     - Listwise deletion (remove rows with missing values)
     - Only effective when missing rate is low (< 5%)

  2. Imputation:
     - Fill with mean/median/mode
     - Forward/backward fill (time series data)
     - KNN imputation (fill with similar data values)
     - Multiple imputation (MICE)

  3. Adding indicator variables:
     - Add "whether missing or not" as a new feature
     - Effective when missingness itself carries information
```

---

## 10. Practical Machine Learning Pipeline

### 10.1 ML Project Lifecycle

```
Overall flow of an ML project:

  ┌──────────────┐
  │ 1. Problem    │ ← Clarify "what to predict" and "what is the KPI"
  │    Definition │
  └──────┬───────┘
         ▼
  ┌──────────────┐
  │ 2. Data       │ ← API, DB, scraping, external datasets
  │    Collection │
  └──────┬───────┘
         ▼
  ┌──────────────┐
  │ 3. EDA        │ ← Exploratory Data Analysis (visualization,
  │ (Exploratory  │    statistics, correlations)
  │  Analysis)    │
  └──────┬───────┘
         ▼
  ┌──────────────┐
  │ 4. Preprocess-│ ← Missing values, encoding, scaling
  │ ing & Feature │    Feature engineering
  │ Engineering   │
  └──────┬───────┘
         ▼
  ┌──────────────┐
  │ 5. Model      │ ← Baseline → progressively complex models
  │ Selection &   │    Hyperparameter tuning
  │ Training      │
  └──────┬───────┘
         ▼
  ┌──────────────┐
  │ 6. Evaluation │ ← Cross-validation, final evaluation on test set
  └──────┬───────┘
         ▼
  ┌──────────────┐
  │ 7. Deployment │ ← REST API, batch inference, edge inference
  │ & Monitoring  │    Data drift detection, model retraining
  └──────────────┘
```

### 10.2 Data Splitting Principles

```
Basic data splitting pattern:

  ┌────────────────────────────────────────────┐
  │             All Data                        │
  ├────────────────────────┬─────────┬─────────┤
  │  Training Data (60%)   │Val (20%)│Test (20%)│
  └────────────────────────┴─────────┴─────────┘

  Training data:  Learns model parameters
  Validation data: Adjusts hyperparameters, model selection
  Test data:      Final performance evaluation (use only once)

  Cautions:
  - Test data is the "last resort." Using it repeatedly causes information leakage
  - Time series data must be split chronologically (don't train on future data)
  - Use stratified sampling (stratify) to maintain class ratios
```

**Code Example 5: Complete ML Pipeline Implementation with scikit-learn**

```python
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import (
    train_test_split, GridSearchCV, cross_val_score
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

# === 1. Load Data ===
data = fetch_california_housing()
X, y = data.data, data.target
feature_names = data.feature_names
print(f"Data size: {X.shape}")
print(f"Features: {feature_names}")

# === 2. Split Data ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Training: {X_train.shape[0]} samples, Test: {X_test.shape[0]} samples")

# === 3. Build Pipeline ===
# Combine preprocessing and model into a single pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),          # Standardization
    ('model', GradientBoostingRegressor(   # Gradient Boosting
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        random_state=42
    ))
])

# === 4. Evaluate Model with Cross-Validation ===
cv_scores = cross_val_score(
    pipeline, X_train, y_train,
    cv=5, scoring='r2'
)
print(f"\nCross-validation R²: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# === 5. Hyperparameter Tuning ===
param_grid = {
    'model__n_estimators': [100, 200, 300],
    'model__max_depth': [3, 5, 7],
    'model__learning_rate': [0.05, 0.1, 0.2],
}

grid_search = GridSearchCV(
    pipeline, param_grid,
    cv=3, scoring='r2',
    n_jobs=-1, verbose=0
)
grid_search.fit(X_train, y_train)
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best CV R²: {grid_search.best_score_:.4f}")

# === 6. Final Evaluation on Test Data ===
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"\n=== Final Evaluation on Test Set ===")
print(f"RMSE: {rmse:.4f}")
print(f"R²:   {r2:.4f}")

# === 7. Feature Importance ===
importances = best_model.named_steps['model'].feature_importances_
sorted_idx = np.argsort(importances)[::-1]

print(f"\nFeature Importance:")
for i in sorted_idx:
    print(f"  {feature_names[i]:15s}: {importances[i]:.4f}")

# Example output:
# Data size: (20640, 8)
# Training: 16512 samples, Test: 4128 samples
#
# Cross-validation R²: 0.7823 (+/- 0.0134)
# Best parameters: {'model__learning_rate': 0.1, ...}
# Best CV R²: 0.7956
#
# === Final Evaluation on Test Set ===
# RMSE: 0.5234
# R²:   0.8012
#
# Feature Importance:
#   MedInc         : 0.4512
#   AveOccup       : 0.1234
#   Latitude       : 0.1123
#   Longitude      : 0.1098
#   ...
```

---

## 11. Generative AI and Large Language Models

### 11.1 Large Language Models (LLM)

```
Basic principle of LLMs: Predict the next token

  Input: "The cat sat on the"
  Output probability distribution:
    "mat"   → 0.35
    "floor" → 0.20
    "bed"   → 0.15
    ...

  Progression of parameter counts:
  ┌─────────────────────────────────────────────┐
  │ Model               │ Parameters            │
  ├─────────────────────────────────────────────┤
  │ GPT-1  (2018)       │        117 million    │
  │ GPT-2  (2019)       │        1.5 billion    │
  │ GPT-3  (2020)       │        175 billion    │
  │ GPT-4  (2023)       │ Undisclosed (est. 1T+)│
  │ Claude Opus (2025)  │ Undisclosed           │
  │ LLaMA 3 (2024)      │    7B to 405B         │
  └─────────────────────────────────────────────┘

  Scaling Laws:
  Model size↑ + Data volume↑ + Compute↑
  → Prediction accuracy improves smoothly (power law)
  → At sufficient scale, "Emergent Abilities" appear
     (CoT reasoning, few-shot learning, code generation, etc.)
```

### 11.2 LLM Training Process

```
3-stage training:

  ┌────────────────────────────────────────────────┐
  │ Stage 1: Pre-training                           │
  │  - Learns "next token prediction" on massive    │
  │    text                                         │
  │  - Thousands of GPUs × several months           │
  │  - Acquires world knowledge, language ability,  │
  │    and reasoning ability                        │
  └─────────────────────┬──────────────────────────┘
                         ▼
  ┌────────────────────────────────────────────────┐
  │ Stage 2: Supervised Fine-Tuning (SFT)           │
  │  - Fine-tuned on high-quality dialogue data     │
  │    created by humans                            │
  │  - Acquires ability to follow instructions      │
  └─────────────────────┬──────────────────────────┘
                         ▼
  ┌────────────────────────────────────────────────┐
  │ Stage 3: RLHF / RLAIF                           │
  │  - Reinforcement learning based on human/AI     │
  │    feedback                                     │
  │  - A reward model evaluates "good responses"    │
  │  - Policy optimized with PPO                    │
  │  → Learns to produce helpful, safe, and honest  │
  │    responses                                    │
  └────────────────────────────────────────────────┘
```

### 11.3 Image Generation and Diffusion Models

```
How Diffusion Models work:

  Forward Process: Gradually add noise to an image
  Image → Noisy image → ... → Pure noise

  Reverse Process: Gradually remove noise to generate an image
  Noise → [denoise] → [denoise] → ... → Image
  (Gaussian)                          (sharp)

  Text-conditioned generation:
  "A cat sitting on a rainbow" → Text encoder → Condition vector
  Noise + Condition vector → Reverse diffusion → Image of a cat on a rainbow
```

### 11.4 RAG (Retrieval-Augmented Generation)

A method that retrieves external knowledge to enhance LLM responses.

```
RAG architecture:

  Query → [Embedding] → [Vector search] → Relevant docs Top-K
                                                   │
                           ┌───────────────────────┘
                           ▼
  Prompt: "Answer the question using the following documents
           as reference:
               {relevant documents}
               Question: {user's question}"
                           │
                           ▼
                      [LLM] → Answer

  Advantages:
  - Reduces hallucination (fabrication)
  - Reflects up-to-date information (beyond LLM training data)
  - Can cite sources (improved reliability)
  - Inject domain-specific knowledge
```

---

## 12. Anti-Pattern Collection

### Anti-Pattern 1: Data Leakage

**Problem**: Information from the test data leaks into the training process, producing overly optimistic evaluation results.

```
Typical leakage examples:

  BAD: Scale all data → Split
  ┌──────────────────────────┐
  │    Standardize all data   │  ← Test statistics leak into training!
  │    ↓                      │
  │    Split into train/test  │
  └──────────────────────────┘

  OK: Split → Scale using training data only
  ┌──────────────────────────┐
  │    Split into train/test  │
  │    ↓                      │
  │    Fit on training data   │
  │    Transform test data    │  ← Only training statistics are used
  └──────────────────────────┘
```

**Other leakage examples**:
- Including future information as features (using next day's values in time series data)
- IDs or row numbers correlating with the target
- Duplicate data appearing in both training and test sets

**Countermeasures**:
- Use `sklearn.pipeline.Pipeline` to integrate preprocessing and model
- Always ask "Is this information available at prediction time?" when creating features
- Don't touch test data until the end (not even for visual inspection)

### Anti-Pattern 2: Ignoring Imbalanced Data

**Problem**: Training a model without any countermeasures when there is a large imbalance in sample counts between classes.

```
Example: Credit card fraud detection
  Normal transactions:  99.7% (99,700 cases)
  Fraudulent:            0.3% (   300 cases)

  Model without countermeasures:
  → Predicting everything as "normal" gives Accuracy = 99.7%
  → But detects zero fraud (Recall = 0%)
```

**Systematic countermeasures**:

| Level | Method | Description |
|-------|--------|-------------|
| Data level | Oversampling (SMOTE) | Generate synthetic data for minority class |
| Data level | Undersampling | Reduce majority class data |
| Algorithm level | Class weighting | `class_weight='balanced'` |
| Evaluation level | Appropriate metrics | F1 / AUC-ROC instead of Accuracy |
| Threshold level | Threshold adjustment | Change the default 0.5 |

```python
# Example of handling imbalanced data
from sklearn.ensemble import RandomForestClassifier

# Weight minority class more heavily using class weights
model = RandomForestClassifier(
    class_weight='balanced',  # Automatically weights by inverse class ratio
    random_state=42
)
```

### Anti-Pattern 3: Manual Hyperparameter Tuning

**Problem**: Setting hyperparameters by intuition without systematic exploration.

**Countermeasures**:

```
Comparison of search methods:

  Grid Search:
  Exhaustively try all combinations
  → Guaranteed but computationally expensive (curse of dimensionality)

  Random Search:
  Randomly sample from parameter space
  → Explores a wider area with the same computational budget

  Bayesian Optimization (Optuna, HyperOpt):
  Estimates promising regions from past results
  → Most efficient. Recommended for large-scale exploration
```

---

## 13. Practical Exercises (3 Levels)

### Exercise 1: [Beginner] -- Classification Intuition and Manual Calculation

```
Manually construct a decision tree to classify "Pass/Fail" using the following data:

| Study Hours | Sleep Hours | Attendance(%) | Result |
|-------------|-------------|--------------|--------|
| 10          | 8           | 90           | Pass   |
| 3           | 5           | 50           | Fail   |
| 8           | 7           | 85           | Pass   |
| 2           | 4           | 40           | Fail   |
| 6           | 6           | 70           | Pass   |
| 1           | 3           | 30           | Fail   |

Tasks:
1. Calculate the Gini impurity for appropriate thresholds of each feature
   and determine the most effective first split criterion.

   Hint: Splitting at study hours = 4.5h:
   Left (≤4.5h): [Fail, Fail, Fail] → Gini = 0
   Right (>4.5h): [Pass, Pass, Pass] → Gini = 0
   Information gain = Original Gini(0.5) - Weighted average(0) = 0.5

2. State the depth and number of leaf nodes of the completed decision tree.

3. Classify new data (study 5h, sleep 6h, attendance 60%).
   Explain why that prediction is made by tracing the decision tree path.

4. What are the limitations of this decision tree? (Discuss from the
   perspectives of data size, feature relationships, and overfitting risk)
```

### Exercise 2: [Intermediate] -- Manual Gradient Descent Calculation and Code Implementation

```
Part A: Manual Calculation
Learn the following data using linear regression y = wx + b:

Data: (1, 2), (2, 4), (3, 6)
Initial values: w=0, b=0, learning rate α=0.1

1. Calculate initial predictions ŷ₁, ŷ₂, ŷ₃
2. Calculate MSE loss
3. Calculate ∂L/∂w and ∂L/∂b
   ∂L/∂w = (-2/3) × [x₁(y₁-ŷ₁) + x₂(y₂-ŷ₂) + x₃(y₃-ŷ₃)]
   ∂L/∂b = (-2/3) × [(y₁-ŷ₁) + (y₂-ŷ₂) + (y₃-ŷ₃)]
4. Update parameters once (w_new, b_new)
5. Track w, b, and loss after 3 updates, and summarize in a table

Part B: Code Implementation
Verify the manual calculations above in Python. Additionally implement:
- Compare learning rates 0.01, 0.1, 0.5 and observe convergence speed differences
- Confirm divergence behavior with learning rate 1.0
- Compare w, b after 100 epochs with the "true values" w=2, b=0

Part C: Advanced
- Derive the gradient with L2 regularization (Ridge) added
  L = MSE + λ||w||²
  ∂L/∂w = ∂MSE/∂w + 2λw
- Compare results with and without regularization at λ=0.1
```

### Exercise 3: [Advanced] -- End-to-End ML System Design

```
Theme: Product Recommendation System for an E-Commerce Site

Requirements:
- DAU (Daily Active Users): 1 million
- Number of products: 10 million
- Response time: Under 100ms
- Personalization: Recommendations based on user behavior history
- Cold start: Handling new users and new products

Design items (provide specific choices and reasoning for each):

1. Algorithm Selection
   - Collaborative filtering vs content-based vs hybrid
   - Consideration of deep learning-based approaches (Two-Tower Model, DIN)
   - Real-time vs batch recommendation architecture

2. Feature Design
   - User features: Age, gender, browsing history, purchase history, ...
   - Product features: Category, price range, review score, ...
   - Context features: Time of day, day of week, device, ...
   - Cross features: User × Product interactions

3. System Architecture
   - Two-stage design: Candidate generation (Recall) → Ranking (Precision)
   - ANN (Approximate Nearest Neighbor) for fast candidate generation
   - Feature Store design

4. Evaluation and Testing
   - Offline evaluation: Hit Rate, NDCG, MAP
   - Online evaluation: A/B test design (statistical significance testing)
   - Guardrail metrics: Not just CTR but also purchase rate, diversity

5. Operations and Monitoring
   - Data drift detection (changes in input distribution)
   - Model retraining frequency and triggers
   - Feedback loop management (suppressing popularity bias)

6. Ethical Considerations
   - Preventing filter bubbles
   - Fairness (preventing discriminatory recommendations by specific attributes)
   - Transparency (explaining why a product was recommended)
```

---

## 14. FAQ (Frequently Asked Questions)

### Q1: Why is machine learning knowledge necessary for programmers?

In modern software development, integrating ML features has become routine:

- Search ranking optimization
- Recommendation engines
- Fraud detection and anomaly detection
- Natural language processing (chatbots, translation, summarization)
- Code completion (GitHub Copilot, Claude)
- Image and speech recognition and generation

Even if you are not an ML engineer, **the skill to understand ML concepts, call APIs appropriately, and correctly interpret results** is required. Particularly important are:

1. **Being able to judge which tasks ML is effective for** (not everything needs ML)
2. **Understanding the limitations of models** (100% accuracy does not exist)
3. **Being able to correctly interpret evaluation metrics** (Accuracy 99% is not necessarily a good model)
4. **Understanding that data quality determines model quality**

### Q2: How much math is needed to get started with machine learning?

It is effective to gradually deepen the required math:

**Level 1 (Just get started)**: Use scikit-learn to run things
- Arithmetic, basic statistics (mean, variance, correlation)
- Concept of matrices (sufficient if you can think of data as tables)

**Level 2 (Understand the mechanisms)**: Gain intuition about algorithms
- **Linear algebra**: Vectors, matrix multiplication, eigenvalues and eigenvectors
- **Calculus**: Partial derivatives, concept of gradients
- **Probability & Statistics**: Probability distributions (normal distribution), Bayes' theorem, maximum likelihood estimation

**Level 3 (Read research papers)**: Understand and implement new methods
- Optimization theory (convex optimization, Lagrange multipliers)
- Information theory (entropy, KL divergence)
- Probability theory (stochastic processes, measure theory)

In practice, libraries like scikit-learn, PyTorch, and TensorFlow abstract away mathematical details, so **an intuitive understanding at Level 2 is sufficient for many tasks**.

### Q3: Do optimal algorithms differ between tabular data and image/text?

They differ significantly. Here are selection guidelines based on data type:

| Data Type | Recommended Algorithm | Reason |
|-----------|----------------------|--------|
| Tabular (structured) | Gradient boosting (XGBoost, LightGBM) | Efficiently learns feature interactions. Requires minimal preprocessing |
| Images | CNN (ResNet, EfficientNet) | Specialized for hierarchical extraction of local patterns |
| Text | Transformer (BERT, GPT) | Self-Attention effectively captures context |
| Time series | LSTM / Transformer / Prophet | Modeling temporal dependencies |
| Graphs | GNN (Graph Neural Network) | Considers relationships between nodes |

For tabular data, deep learning has difficulty outperforming gradient boosting. A 2022 benchmark study (Grinsztajn et al.) confirmed that tree-based models remain superior to NNs for tabular data.

### Q4: How are Large Language Models (LLMs) trained?

They are trained through a 3-stage process:

1. **Pre-training**: Learns the "next word prediction" task on massive text (trillions of tokens) from the internet. Runs thousands of GPUs for months. At this stage, the foundation for world knowledge, language ability, and logical reasoning is acquired.

2. **Supervised Fine-Tuning (SFT)**: Fine-tuned on high-quality instruction-response pair data created by human annotators. At this stage, the ability to "generate responses following instructions" is acquired.

3. **RLHF / RLAIF**: Reinforcement learning based on human or AI feedback. Generates multiple responses, trains a reward model that evaluates which is "better," and optimizes the policy using algorithms like PPO. At this stage, the balance of helpfulness, safety, and honesty is adjusted.

### Q5: What is the most important thing in operating machine learning models?

**Monitoring data drift**. Models make predictions based on the assumption of the data distribution at training time. However, real-world data distributions change over time (concept drift).

Examples:
- COVID-19's impact drastically changed purchasing behavior, causing recommendation models to fail
- Seasonal variation caused a model trained in summer to degrade in winter
- The appearance of new products made existing feature spaces inadequate

Countermeasures:
1. **Monitor input data distribution**: Detect distribution changes with KS tests or PSI (Population Stability Index)
2. **Periodic performance evaluation**: Re-evaluate performance when actual labels become available
3. **Automated retraining pipeline**: Automatically retrain the model when performance falls below a threshold
4. **Continuous A/B testing**: Continuously compare old and new models

---


## FAQ

### Q1: What is the most important point in learning this topic?

Gaining practical experience is the most important thing. Understanding deepens not just through theory but by actually writing code and verifying behavior.

### Q2: What are common mistakes beginners make?

Skipping the fundamentals and jumping to applications. We recommend thoroughly understanding the basic concepts explained in this guide before moving to the next step.

### Q3: How is this used in practice?

Knowledge of this topic is frequently used in day-to-day development work. It becomes especially important during code reviews and architecture design.

---

## 15. Summary and Comparison Tables

### Machine Learning Algorithm Comparison Table

| Algorithm | Type | Interpretability | Accuracy | Training Speed | Primary Use |
|-----------|------|------------------|----------|---------------|-------------|
| Linear Regression | Regression | High | Low-Med | Very fast | Baseline, interpretability focus |
| Logistic Regression | Classification | High | Low-Med | Very fast | Binary classification, probability estimation |
| Decision Tree | Classif./Regr. | Very high | Low-Med | Fast | When interpretability is needed |
| Random Forest | Classif./Regr. | Medium | Med-High | Medium | General-purpose first choice |
| Gradient Boosting | Classif./Regr. | Low-Med | High | Medium | Best for tabular data |
| SVM | Classif./Regr. | Low | Med-High | Slow | High-dimensional, small-medium data |
| kNN | Classif./Regr. | Medium | Medium | Slow at inference | Small data, nonparametric |
| Neural Network | Classif./Regr. | Low | High | Slow | Large data, complex patterns |
| CNN | Image classif. etc. | Low | Very high | Slow | Image recognition |
| RNN/LSTM | Sequence classif. | Low | High | Slow | Time series, text |
| Transformer | Multi-purpose | Low | Very high | Very slow | NLP, CV, multimodal |

### Learning Paradigm Comparison Table

| Item | Supervised Learning | Unsupervised Learning | Reinforcement Learning |
|------|--------------------|-----------------------|----------------------|
| Input | X + y (labeled) | X only | State s |
| Output | Predicted value ŷ | Clusters, latent repr. | Action a |
| Objective | Minimize prediction error | Discover data structure | Maximize cumulative reward |
| Data volume | Med-Large (labels needed) | Large (no labels needed) | Environment interaction |
| Representative methods | Regression, Classif., SVM, RF | K-Means, PCA, AE | Q-Learning, PPO |
| Typical applications | Spam detection, prediction | Customer segmentation, visualization | Game AI, robotics |
| Difficult aspects | Label acquisition cost | Evaluation is subjective | Reward design, training instability |

### Core Concepts Summary

| Concept | Key Point |
|---------|-----------|
| Bias-Variance | Trade-off between model simplicity and complexity |
| Overfitting | The phenomenon where a model over-specializes on training data, reducing generalization |
| Regularization | L1/L2 penalties, Dropout, etc. to prevent overfitting |
| Cross Validation | Repeatedly split data to stably evaluate performance |
| Feature Engineering | The art of transforming data using domain knowledge |
| Data Leakage | A critical mistake where test information leaks into training |
| Evaluation Metrics | Choosing the right metric for the task is paramount |
| Data Drift | The problem of data distribution changing during operation, degrading performance |

---

## Recommended Next Guides


---

## 16. References

### Books

1. **Goodfellow, I., Bengio, Y., & Courville, A.** "Deep Learning." MIT Press, 2016.
   - A textbook that systematically covers the theoretical foundations of deep learning. Available for free online. Covers from mathematical foundations to state-of-the-art architectures.

2. **Bishop, C. M.** "Pattern Recognition and Machine Learning." Springer, 2006.
   - A classic masterpiece of machine learning. Explains machine learning from a probabilistic perspective centered on a Bayesian approach. Recommended for readers seeking theoretical depth.

3. **Hastie, T., Tibshirani, R., & Friedman, J.** "The Elements of Statistical Learning." Springer, 2009.
   - An encyclopedic work on statistical learning theory. Available for free online. An important bridge between statistics and ML.

### Papers

4. **Vaswani, A. et al.** "Attention Is All You Need." NeurIPS, 2017.
   - The groundbreaking paper that proposed the Transformer architecture. The foundation of all modern LLMs (GPT, Claude, LLaMA, etc.).

5. **He, K. et al.** "Deep Residual Learning for Image Recognition." CVPR, 2016.
   - The paper proposing ResNet (residual connections). An important technical breakthrough that enabled training of deep networks.

6. **Grinsztajn, L. et al.** "Why do tree-based models still outperform deep learning on typical tabular data?" NeurIPS, 2022.
   - Performance comparison of gradient boosting and NNs on tabular data. Demonstrates that tree-based models remain superior on tabular data.

### Online Resources

7. **Andrew Ng.** "Machine Learning Yearning." 2018.
   - A practical guide on strategy and decision-making for ML projects. Free publication.

8. **scikit-learn Official Documentation** -- https://scikit-learn.org/stable/
   - Not only explains how to use algorithms but also provides thorough explanations of theoretical background.

9. **Google Machine Learning Crash Course** -- https://developers.google.com/machine-learning/crash-course
   - A free introductory ML course provided by Google. Includes practical exercises.

---

## References

- [MDN Web Docs](https://developer.mozilla.org/) - Web technology reference
- [Wikipedia](https://en.wikipedia.org/) - Overview of technical concepts
