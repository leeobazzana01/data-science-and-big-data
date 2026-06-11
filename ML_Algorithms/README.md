# Machine Learning Pipeline & Hyperparameter Optimization Review

This report presents a theoretical and practical decomposition of an end-to-end Machine Learning classification pipeline. The study is applied to a **Customer Churn** dataset containing **10,000 instances** and is divided into two primary phases:

1. **Data Preprocessing**
2. **Algorithm Optimization via Grid Search**

---

# 1. Phase A: Data Preprocessing Framework

Before machine learning algorithms can effectively learn from data, the dataset must undergo cleaning, transformation, encoding, and scaling.

Proper preprocessing ensures that the feature matrix is mathematically consistent and suitable for statistical learning algorithms.

---

## 1.1 Dimensionality Reduction (Feature Selection)

To reduce noise and prevent the model from learning meaningless relationships, non-predictive identifiers were removed from the dataset.

### Dropped Features

- `RowNumber`
- `CustomerId`
- `Surname`

### Data Science Insight

These variables function as identifiers rather than explanatory predictors.

Including them may lead to:

- Increased dimensionality.
- Poor generalization.
- Spurious correlations.
- Model overfitting.

The resulting feature matrix retains only attributes with potential predictive value for customer churn.

---

## 1.2 Imputation Strategies

Machine learning algorithms in **scikit-learn** generally cannot process missing values directly.

To preserve dataset integrity, two imputation strategies were employed.

### Listwise Deletion

Rows containing missing values in:

```text
IsActiveMember
```

were completely removed from the dataset.

#### Mathematical Consequence

If a row contains critical missing information, it is excluded from the feature matrix:

$$
X' = X - \{\text{rows with missing values}\}
$$

This method preserves data consistency but slightly reduces sample size.

---

### Mean Imputation

Missing values in the continuous feature:

```text
NumOfProducts
```

were replaced using the arithmetic mean of the available observations.

The calculated replacement value was:

$$
\bar{x} = 1.53
$$

#### Mathematical Concept

For a missing observation:

$$
x_i = \text{NaN}
$$

the replacement becomes:

$$
x_i = \bar{x}
$$

This preserves dataset size while maintaining the central tendency of the variable.

---

## 1.3 Categorical Feature Encoding

Machine learning algorithms operate on numerical tensors and matrices.

Categorical features must therefore be transformed into mathematical representations.

---

### Label Encoding

Applied to binary categorical variables such as:

```text
Gender
```

Example transformation:

| Original Value | Encoded Value |
|----------------|---------------|
| Male | 0 |
| Female | 1 |

#### Mathematical Representation

A categorical set:

$$
\{\text{Male}, \text{Female}\}
$$

is transformed into:

$$
\{0,1\}
$$

This preserves binary structure while enabling numerical computation.

---

### One-Hot Encoding (Dummy Variables)

Applied to nominal multi-class features such as:

```text
Geography
```

Examples:

| Geography | France | Germany | Spain |
|------------|---------|----------|--------|
| France | 1 | 0 | 0 |
| Germany | 0 | 1 | 0 |
| Spain | 0 | 0 | 1 |

---

#### Why One-Hot Encoding?

Using simple integer encoding:

$$
\text{France}=1,\;
\text{Germany}=2,\;
\text{Spain}=3
$$

would incorrectly imply an ordinal relationship:

$$
\text{France} < \text{Germany} < \text{Spain}
$$

which has no statistical meaning.

One-hot encoding avoids introducing artificial hierarchy into the model.

---

## 1.4 Feature Scaling (Standardization)

The dataset was divided into:

- 80% Training Set
- 20% Test Set

After splitting, feature scaling was performed using:

```python
StandardScaler()
```

Distance-based algorithms are highly sensitive to variable magnitude.

Examples include:

- K-Nearest Neighbors (KNN)
- Support Vector Machines (SVM)

Without scaling, variables with larger numeric ranges dominate distance calculations.

---

### Mathematical Concept (Z-Score Normalization)

Standardization transforms every feature into a distribution with:

$$
\mu = 0
$$

and

$$
\sigma = 1
$$

using:

$$
z
=
\frac{x-\mu}{\sigma}
$$

where:

- $x$ = original value
- $\mu$ = feature mean
- $\sigma$ = feature standard deviation

---

### Benefits

Standardization:

- Improves optimization convergence.
- Prevents scale dominance.
- Enhances distance calculations.
- Improves model stability.

---

# 2. Phase B: Machine Learning & Optimization

After preprocessing, predictive models were trained and optimized.

Instead of relying on default parameters, the project implemented:

```python
GridSearchCV()
```

combined with:

```python
cv = 5
```

for systematic hyperparameter optimization.

---

## 2.1 Grid Search & Cross-Validation

### Grid Search

Grid Search exhaustively evaluates all parameter combinations defined within a search space.

Example:

```python
{
    "C": [0.01, 0.1, 1, 10]
}
```

Each configuration is trained and evaluated independently.

---

### 5-Fold Cross-Validation

The training dataset is partitioned into five folds.

For each iteration:

1. Four folds are used for training.
2. One fold is used for validation.

This process repeats five times.

#### Mathematical Interpretation

Given a dataset:

$$
D
=
\{F_1,F_2,F_3,F_4,F_5\}
$$

each fold serves once as validation data.

The final score becomes:

$$
\text{CV Score}
=
\frac{1}{5}
\sum_{i=1}^{5}
Accuracy_i
$$

This greatly reduces the risk of selecting parameters that perform well only on a specific subset of the data.

---

# 3. Applied Algorithms

---

## 3.1 Logistic Regression

Logistic Regression is a linear probabilistic classifier.

Rather than predicting classes directly, it estimates probabilities through the sigmoid function:

$$
\sigma(z)
=
\frac{1}{1+e^{-z}}
$$

where:

$$
z
=
\beta_0
+
\beta_1x_1
+
...
+
\beta_nx_n
$$

Predicted probabilities are then converted into class labels.

### Characteristics

- Fast training.
- High interpretability.
- Effective on linearly separable problems.

---

## 3.2 K-Nearest Neighbors (KNN)

KNN is a non-parametric, instance-based learning algorithm.

A new observation is classified according to the majority class among its nearest neighbors.

### Mathematical Concept

For a query point:

$$
x_q
$$

the algorithm computes distances to all training observations.

Most commonly:

$$
d(x_i,x_j)
=
\sqrt{
\sum_{k=1}^{n}
(x_{ik}-x_{jk})^2
}
$$

(Euclidean Distance)

The majority class among the nearest $k$ neighbors becomes the prediction.

---

## 3.3 Support Vector Machine (SVM)

Support Vector Machines seek the optimal separating hyperplane between classes.

### Objective

Maximize the margin:

$$
\text{Margin}
=
\frac{2}{||w||}
$$

between the nearest observations of each class.

---

### Radial Basis Function (RBF) Kernel

The pipeline employed:

```text
kernel = rbf
```

which enables nonlinear decision boundaries.

The kernel transformation is:

$$
K(x_i,x_j)
=
e^{-\gamma ||x_i-x_j||^2}
$$

allowing the model to separate classes in higher-dimensional feature spaces.

---

## 3.4 Random Forest Classifier

Random Forest is an ensemble learning algorithm based on multiple decision trees.

Instead of relying on a single tree:

$$
T_1,T_2,\dots,T_n
$$

the model aggregates predictions from many trees.

### Final Prediction

For classification:

$$
\hat{y}
=
\text{mode}
(T_1,T_2,\dots,T_n)
$$

where the most frequent prediction becomes the final output.

---

### Advantages

- High predictive power.
- Reduced variance.
- Strong resistance to overfitting.
- Ability to model nonlinear relationships.

---

# 4. Optimization Results & Performance Metrics

After exploring the hyperparameter search space, the best configuration for each algorithm was evaluated on the unseen test set.

---

## Best Hyperparameter Configurations

| Algorithm | Optimal Hyperparameters | Test Accuracy |
|------------|-------------------------|---------------:|
| Logistic Regression | $C = 0.01$ | **69.92%** |
| Support Vector Machine | $C = 1$, Kernel = `rbf` | **80.18%** |
| K-Nearest Neighbors | Neighbors = 11, Weights = `uniform` | **83.83%** |
| Random Forest | Estimators = 200, Max Depth = None | **85.94%** |

---

## Performance Ranking

$$
\text{Random Forest}
>
\text{KNN}
>
\text{SVM}
>
\text{Logistic Regression}
$$

---

## Evaluation Metrics

Performance was assessed using:

### Accuracy

The proportion of correctly classified observations.

$$
Accuracy
=
\frac{
TP + TN
}{
TP + TN + FP + FN
}
$$

where:

- $TP$ = True Positives
- $TN$ = True Negatives
- $FP$ = False Positives
- $FN$ = False Negatives

---

### Confusion Matrix

A confusion matrix summarizes prediction outcomes by comparing actual and predicted classes.

| | Predicted Positive | Predicted Negative |
|---|---|---|
| Actual Positive | TP | FN |
| Actual Negative | FP | TN |

This matrix provides deeper insight than accuracy alone by revealing the types of classification errors made by the model.

---

# Conclusion

This project implemented a complete machine learning classification workflow, beginning with data preprocessing and culminating in systematic hyperparameter optimization.

Key findings include:

- Missing values were handled through deletion and mean imputation.
- Categorical features were transformed using Label Encoding and One-Hot Encoding.
- Feature scaling standardized all variables through Z-score normalization.
- Grid Search combined with 5-fold Cross-Validation identified optimal hyperparameter configurations.
- Random Forest delivered the strongest predictive performance, achieving an accuracy of **85.94%**.

Overall, the ensemble-based Random Forest model substantially outperformed the linear and distance-based alternatives, demonstrating superior capability in capturing complex relationships within the customer churn dataset while maintaining strong generalization performance.
