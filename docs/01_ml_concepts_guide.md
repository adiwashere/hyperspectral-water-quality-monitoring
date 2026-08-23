# Machine Learning Concepts — Hyperspectral Water-Quality Guide

Welcome to Machine Learning! This guide breaks down every core concept you need using our **Hyperspectral Water-Quality Monitoring** project as a concrete example.

---

## 1. Features vs. Target ($X$ and $y$)

In Machine Learning, we learn patterns from data to make predictions. We split our data into two main parts:

### Features ($X$)
- **What it is**: The input measurements used to make a prediction.
- **In our project**: The **51 hyperspectral reflectance bands** (`band_400nm`, `band_410nm`, ..., `band_900nm`). Each band measures how much light reflects off the water surface at a specific wavelength.
- **Symbol**: Represented as a matrix capital **$X$** because it has multiple columns (samples × 51 spectral features).

### Target ($y$)
- **What it is**: The value we want the ML model to predict.
- **In our project**: **Turbidity ($NTU$)** (a measure of water cloudiness caused by suspended sediments).
- **Symbol**: Represented as a vector lowercase **$y$** (length = number of samples).

---

## 2. Regression vs. Classification

- **Regression**: Predicting a **continuous numerical value** (e.g., Turbidity = `4.2 NTU`, `18.7 NTU`).
  - *Our project is a REGRESSION problem.*
- **Classification**: Predicting a **discrete category or class** (e.g., Water Quality = `"Drinkable"` vs `"Polluted"`).

---

## 3. Training Data vs. Test Data (Train/Test Split)

Why can't we test our model on the exact same data it learned from?
Because a model could simply "memorize" the answers without actually understanding the underlying physics!

- **Training Set (80%)**: The data the model studies to learn relationships between spectral curves and turbidity.
- **Test Set (20%)**: Held-out data that the model has **never seen before**. We test the model on this set to evaluate its real-world performance on new water samples.
- **Train/Test Split**: We randomly split our dataset before training ($X_{\text{train}}, X_{\text{test}}, y_{\text{train}}, y_{\text{test}}$).

---

## 4. Overfitting vs. Underfitting

- **Underfitting (High Bias)**: The model is too simple (e.g., trying to fit complex spectral curves with a flat line). It performs poorly on both training and test data.
- **Overfitting (High Variance)**: The model is overly complex and memorizes noisy sensor glitches in the training set. It gets 100% accuracy on training data, but fails miserably on new test data.
- **Goal**: Find a balanced model that generalizes well to unseen spectral signatures.

---

## 5. Data Leakage (A Critical Trap!)

- **What it is**: When information from the test dataset accidentally leaks into the training phase (e.g., scaling features using the mean/standard deviation of the entire dataset instead of *only* the training set).
- **Why it matters**: Data leakage leads to overly optimistic performance numbers during development, which crash when deployed to real-world cameras.
- **Rule**: ALWAYS compute normalization factors (scaler parameters, PCA fits) on **training data only**, then apply those same parameters to transform the test data.

---

## 6. Model Training (`model.fit`) vs. Model Inference (`model.predict`)

- **Training (`model.fit(X_train, y_train)`)**: The process where the algorithm adjusts its internal mathematical weights/trees to minimize error between predicted turbidity and true turbidity.
- **Inference (`model.predict(X_new)`)**: Feeding new, unseen spectral band measurements into the trained model to instantly output predicted turbidity numbers (e.g., `{ "turbidity": 4.2 }`).

---

## 7. Model Evaluation Metrics

How do we measure how good or bad our model predictions are?

### Mean Absolute Error (MAE)
$$\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$$
- **Meaning**: The average error in NTU units.
- **Example**: An MAE of `0.8 NTU` means our model's predictions are off by `0.8 NTU` on average.

### Root Mean Squared Error (RMSE)
$$\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}$$
- **Meaning**: Similar to MAE, but penalizes large errors much more heavily (because differences are squared before taking the square root). Useful for catching occasional large prediction failures.

### $R^2$ Score (Coefficient of Determination)
- **Meaning**: Measures how much of the variance in turbidity is explained by the spectral features.
- **Range**:
  - $R^2 = 1.0$: Perfect predictions.
  - $R^2 = 0.0$: Model is no better than simply predicting the average turbidity for every sample.
  - $R^2 < 0.0$: Model is worse than taking a random average.
- **Goal**: Achieve $R^2 \ge 0.85$ on unseen test data!
