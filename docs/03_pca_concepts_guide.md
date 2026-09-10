# Principal Component Analysis (PCA) & Dimensionality Reduction

Welcome to Phase 3! In this guide, we explore how to handle **high-dimensional hyperspectral data** using **Principal Component Analysis (PCA)**.

---

## 1. What Dimensionality Reduction Means

- **Dimensions**: In ML, "dimensions" refers to the number of feature columns ($X$). In our dataset, we have **51 dimensions** (`band_400nm` to `band_900nm`).
- **Dimensionality Reduction**: The process of compressing a high-dimensional dataset (51 bands) into a smaller number of new features (e.g., 5 to 15 components) while preserving as much of the original information as possible.

---

## 2. Why Hyperspectral Bands Can Be Redundant (Multicollinearity)

Hyperspectral sensors record light reflectance at fine 10nm step intervals. 
- The reflectance at `450nm` is almost identical to the reflectance at `460nm` and `470nm`.
- When adjacent feature columns are strongly correlated with each other, it is called **Multicollinearity**.
- Redundant bands add memory overhead and computational cost without adding new information.

---

## 3. What a Principal Component Represents

- A **Principal Component (PC)** is a new artificial feature created as a linear combination (weighted sum) of all 51 original spectral bands:
  $$\text{PC}_1 = w_1 \cdot \text{band}_{400} + w_2 \cdot \text{band}_{410} + \dots + w_{51} \cdot \text{band}_{900}$$
- **Key Property**: All Principal Components are mathematically orthogonal (uncorrelated) to each other!
- **$\text{PC}_1$** captures the largest possible direction of variance in the spectral data.
- **$\text{PC}_2$** captures the second largest direction of variance, orthogonal to $\text{PC}_1$, and so on.

---

## 4. What Explained Variance Means

- **Explained Variance Ratio**: The percentage of the dataset's total variation captured by each individual principal component.
  - For example, $\text{PC}_1$ might explain $82\%$ of total variance, $\text{PC}_2$ explains $12\%$, $\text{PC}_3$ explains $3\%$.
- **Cumulative Explained Variance**: The sum of explained variance as you add components.
  - $\text{PC}_1 + \text{PC}_2 = 82\% + 12\% = 94\%$ total variance retained.

---

## 5. Why PCA Can Help

1. **Eliminates Multicollinearity**: Removes correlation between adjacent bands.
2. **Speed & Efficiency**: Reduces 51 features down to e.g. 10 features, making training and real-time backend API predictions faster.
3. **Denoising**: Ignores low-variance components that often represent high-frequency sensor noise.

---

## 6. Why PCA Can Sometimes HURT Model Performance

1. **Unsupervised Nature**: PCA looks only at variance in inputs ($X$) — it has NO knowledge of the target variable ($y$: Turbidity!).
   - If a subtle spectral wavelength absorption band contains critical information for predicting Turbidity, but has low overall variance across samples, PCA might discard it as component #45!
2. **Loss of Interpretability**: Original spectral band wavelengths ($705nm$) have direct physical bio-optical meanings. A principal component is an abstract mixture of all 51 bands, making physical interpretation harder.
3. **Tree-Based Models**: Algorithms like Random Forest and XGBoost are invariant to monotonic transformations and naturally handle feature selection, so compressing with PCA may sometimes slightly decrease their $R^2$ score compared to raw features.

---

## 7. CRITICAL: Why PCA Must NOT Be Fitted on Test Data

> **RULES OF ANTI-DATA LEAKAGE IN PCA**:
> 1. Fit PCA strictly on scaled training data: `pca.fit(X_train_scaled)`.
> 2. Transform training data: `X_train_pca = pca.transform(X_train_scaled)`.
> 3. Transform test data using the ALREADY FITTED PCA object: `X_test_pca = pca.transform(X_test_scaled)`.

- **Why?**: If you call `pca.fit()` on the entire dataset or on `X_test`, PCA calculates eigenvector direction vectors using test sample distributions. This leaks test structure into the training pipeline!
