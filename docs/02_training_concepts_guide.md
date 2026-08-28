# Machine Learning Training & Evaluation Concepts

Welcome to Phase 2! In this guide, we break down step-by-step how Machine Learning models learn from hyperspectral water data.

---

## 1. What $X$ and $y$ Represent

### $X$ (Feature Matrix)
- **Definition**: The input variables fed into the model.
- **In our project**: A 2D table where each row is a water sample and each column is a **hyperspectral reflectance band** (`band_400nm` through `band_900nm`).
- **Shape**: `(500 samples, 51 spectral features)`.

### $y$ (Target Vector)
- **Definition**: The continuous output quantity we want our model to learn to predict.
- **In our project**: **Turbidity ($NTU$)**.
- **Shape**: `(500 ground-truth turbidity values)`.

---

## 2. Why We Split the Dataset (Train / Test Split)

- **Ratio**: 80% Training Set ($X_{\text{train}}, y_{\text{train}}$) and 20% Test Set ($X_{\text{test}}, y_{\text{test}}$).
- **Reason**: If we test a model on the same data it learned from, it can memorize the noise and pass with 100% score (like giving a student the exact exam questions beforehand).
- **Evaluating on unseen $X_{\text{test}}$** tells us how the model will perform when deployed to a real hyperspectral camera monitoring a real lake or river.

---

## 3. Why `StandardScaler` is Fitted ONLY on Training Data

### Standard Scaling Formula:
$$z = \frac{x - \mu}{\sigma}$$
Where $\mu$ is the mean reflectance of a band and $\sigma$ is the standard deviation. Scaling puts all features on the same scale ($mean = 0, std = 1$), which is crucial for algorithms like SVR and Linear Regression.

### Preventing Data Leakage:
> **CRITICAL RULE**: 
> 1. Compute $\mu_{\text{train}}$ and $\sigma_{\text{train}}$ using `scaler.fit(X_train)`.
> 2. Transform training data: `X_train_scaled = scaler.transform(X_train)`.
> 3. Transform test data using the **TRAINING** statistics: `X_test_scaled = scaler.transform(X_test)`.

- **What happens if you fit on the whole dataset?**: Information about the test set's mean and variance "leaks" into the scaler before training. The model implicitly gets hints about test set distributions, giving artificially high performance numbers that fail in production.

---

## 4. What `model.fit()` Does

- **Syntax**: `model.fit(X_train_scaled, y_train)`
- **Mechanism**:
  - For **Linear Regression**: Calculates optimal coefficients $\beta$ that minimize the sum of squared errors: $y = \beta_0 + \sum \beta_j x_j$.
  - For **Random Forest**: Builds multiple decision trees by splitting spectral bands at thresholds (e.g., "Is `band_705nm` > 0.015?").
  - For **Support Vector Regressor (SVR)**: Constructs a hyper-plane in high-dimensional feature space bounded by support vectors.
  - For **XGBoost**: Iteratively fits tree after tree, each new tree correcting the residual errors left by previous trees.

---

## 5. What `model.predict()` Does

- **Syntax**: `y_pred = model.predict(X_test_scaled)`
- **Mechanism**: Passes the 51 scaled band reflectance features of new water samples into the trained mathematical model and outputs estimated **Turbidity ($NTU$)** numbers.

---

## 6. Evaluation Metrics Explained

### Mean Absolute Error (MAE)
$$\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$$
- Measures the average magnitude of prediction errors in actual Turbidity NTU units.
- An MAE of `0.85 NTU` means predictions are off by `0.85 NTU` on average.

### Root Mean Squared Error (RMSE)
$$\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}$$
- Gives higher penalty to larger outlier errors because errors are squared before averaging.

### $R^2$ Score (Coefficient of Determination)
$$R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$$
- Indicates what percentage of the variance in Turbidity is explained by the hyperspectral reflectance bands.
- $R^2 = 1.0$ is perfect; $R^2 = 0.90$ means 90% of turbidity variation is explained by the spectral signature.
