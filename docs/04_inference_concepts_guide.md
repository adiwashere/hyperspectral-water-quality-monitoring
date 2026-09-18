# Machine Learning Inference vs. Model Training

Welcome to Phase 4! In this guide, we break down the fundamental difference between **Model Training** and **Model Inference (Prediction)**.

---

## 1. The Core Difference: Training vs. Inference

| Aspect | Model Training Phase | Model Inference (Prediction) Phase |
|---|---|---|
| **Goal** | Learn rules and mathematical patterns from training data ($X_{\text{train}}, y_{\text{train}}$). | Apply learned rules to predict outcomes on new, unseen samples. |
| **Methods Used** | `scaler.fit()`, `pca.fit()`, `model.fit()` | `scaler.transform()`, `pca.transform()`, `model.predict()` |
| **State** | **MUTABLE**: Updates weights, means, standard deviations, eigenvectors. | **IMMUTABLE**: Preserves saved weights; does NOT modify parameters. |
| **Speed** | Computational heavy (takes seconds/minutes/hours). | Lightning fast (takes milliseconds). |

---

## 2. Training Sequence: `fit()`

During training, we calculate and save all statistical parameters from the **training set**:

1. **`scaler.fit(X_train)`**: Computes the mean $\mu_{\text{train}}$ and standard deviation $\sigma_{\text{train}}$ of each spectral band.
2. **`pca.fit(X_train_scaled)`**: Computes the principal component direction vectors (eigenvectors) of the spectral feature space.
3. **`model.fit(X_train_pca, y_train)`**: Adjusts decision tree split thresholds or regression coefficients to accurately predict Turbidity.

---

## 3. Inference Sequence: `transform()` & `predict()`

When a new hyperspectral camera reading arrives, we do **NOT** re-fit anything! Re-fitting would erase our trained parameters.

```
New 51-Band Spectral Vector
            ↓
  [ scaler.transform() ]   <-- Uses saved μ and σ from scaler.joblib
            ↓
   [ pca.transform() ]     <-- Projects 51 bands into 2 components using pca_transformer.joblib
            ↓
  [ model.predict() ]     <-- Evaluates 2 PCA components using best_model.joblib
            ↓
   Predicted Turbidity
```

---

## 4. Why `.fit()` Must NEVER Be Called During Inference

1. **Parameter Eradication**: Calling `scaler.fit(new_sample)` would overwrite the learned mean and standard deviation of 400 training samples with the single new sample's value ($\sigma = 0$), breaking the model!
2. **Deterministic Consistency**: We want the exact same hyperspectral input to always yield the exact same predicted Turbidity value.
3. **Latency**: Inference must respond in milliseconds in web servers and edge cameras.
