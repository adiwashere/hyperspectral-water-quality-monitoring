# Hyperspectral Water-Quality Monitoring System

## Week 1 — Data Understanding & Exploration

### What did I do this week?

This week focused on understanding the water-quality problem, creating/loading the hyperspectral dataset, exploring the spectral data, and preparing the dataset for machine-learning work.

The main goal is to understand how **hyperspectral reflectance measurements can be used to predict water turbidity**.

---

## 1. Project Problem

The project is designed to estimate **water turbidity** from hyperspectral measurements.

### What is turbidity?

Turbidity describes how cloudy or unclear water is because of suspended particles such as:

- Sediment
- Soil particles
- Organic matter
- Other suspended material

Turbidity is measured in **NTU (Nephelometric Turbidity Units)**.

The machine-learning problem can be represented as:

```text
Hyperspectral measurements
          ↓
      ML Model
          ↓
   Turbidity (NTU)
```

So the model will eventually learn:

```text
Spectral pattern → Turbidity
```

---

## 2. What is Hyperspectral Data?

A normal RGB camera captures only a few broad color channels:

```text
Red
Green
Blue
```

Hyperspectral sensing captures measurements at many different wavelengths.

In this project, wavelengths range from:

```text
400 nm → 900 nm
```

with a spacing of:

```text
10 nm
```

Therefore:

```text
400, 410, 420, 430, ... , 890, 900
```

This gives:

```text
51 spectral bands
```

The 51 bands become the input features for the machine-learning model.

---

## 3. Dataset Structure

The raw dataset is located at:

```text
data/raw/water_quality_hyperspectral_data.csv
```

Each row represents **one water sample**.

The dataset contains:

```text
sample_id
turbidity_ntu
chlorophyll_a_ugL
data_source
band_400nm
band_410nm
band_420nm
...
band_900nm
```

---

## 4. Understanding X and y

For machine learning, we separate the dataset into:

```text
X = Input Features
y = Target
```

### X — Features

The hyperspectral bands are the features:

```text
band_400nm
band_410nm
band_420nm
...
band_900nm
```

There are 51 of them.

Each value represents the measured/simulated reflectance at a particular wavelength.

Therefore:

```text
X = 51 spectral measurements
```

### y — Target

The target is:

```text
turbidity_ntu
```

The eventual ML task is:

```text
X (51 spectral bands)
          ↓
      ML Model
          ↓
y = turbidity_ntu
```

---

## 5. Why are df_target and df_bands separate?

During dataset generation, two different pieces of information are created.

### df_target

Contains information about the water sample:

```text
sample_id
turbidity_ntu
chlorophyll_a_ugL
data_source
```

### df_bands

Contains the 51 hyperspectral measurements:

```text
band_400nm
band_410nm
...
band_900nm
```

These are combined using:

```python
df_full = pd.concat([df_target, df_bands], axis=1)
```

`axis=1` means the columns are combined **horizontally/side-by-side**.

Both DataFrames contain the same samples in the same row order, so row 0 from `df_target` corresponds to row 0 from `df_bands`, row 1 corresponds to row 1, and so on.

---

## 6. What does axis=1 mean?

`axis=1` means we concatenate **columns horizontally**.

Conceptually:

```text
df_target                  df_bands

sample_id                  band_400nm
turbidity                  band_410nm
chlorophyll                band_420nm
data_source                ...
                            band_900nm
```

After concatenation:

```text
sample_id
turbidity
chlorophyll
data_source
band_400nm
band_410nm
...
band_900nm
```

So `df_full` becomes the complete dataset.

`axis=0` would instead concatenate rows vertically.

---

## 7. Bio-Optical Dataset Simulation

The project currently uses a simulated dataset based on simplified bio-optical relationships.

The dataset generator is:

```text
src/data_loader.py
```

The function responsible for generating the dataset is:

```python
generate_demo_dataset()
```

The dataset is labeled:

```text
DEMO DATA - BIO-OPTICAL SIMULATION
```

This is important because the data is simulated rather than collected from a real hyperspectral water sensor.

---

## 8. Turbidity Generation

Turbidity values are generated using a log-normal distribution:

```python
turbidity = np.random.lognormal(
    mean=1.8,
    sigma=0.8,
    size=n_samples
)
```

The values are then restricted using:

```python
np.clip(turbidity, 0.5, 50.0)
```

This keeps the simulated turbidity within the chosen range.

The target variable is:

```text
turbidity_ntu
```

---

## 9. Chlorophyll-a Generation

The dataset also simulates chlorophyll-a:

```python
chl_a = 0.8 * turbidity + np.random.normal(5, 2, size=n_samples)
```

It is then restricted using:

```python
np.clip(chl_a, 0.5, 100.0)
```

Chlorophyll-a is included because phytoplankton can influence the optical properties of water.

---

## 10. Wavelength Array

The wavelength values are generated using:

```python
WAVELENGTHS = np.arange(400, 901, 10)
```

This produces:

```text
400
410
420
...
900
```

The feature names are then created using:

```python
BAND_COLS = [
    f"band_{w}nm"
    for w in WAVELENGTHS
]
```

This produces:

```text
band_400nm
band_410nm
band_420nm
...
band_900nm
```

---

## 11. Pure Water Absorption

The simulation includes a simplified model of water absorption:

```python
a_water = 0.01 * np.exp((w - 400) / 100.0) + 0.02
```

The idea is that water absorbs light differently at different wavelengths.

In the simplified simulation, absorption increases toward longer wavelengths.

---

## 12. Particle Absorption

Particles in water are also represented:

```python
a_particles = 0.05 * np.exp(
    -(w - 400) / 150.0
)
```

This represents a simplified wavelength-dependent particle absorption component.

---

## 13. Phytoplankton Absorption

A Gaussian-shaped absorption feature is simulated around:

```text
675 nm
```

using:

```python
a_phytoplankton = 0.02 * np.exp(
    -((w - 675) ** 2) / (2 * 15**2)
)
```

The negative sign is important.

It creates a bell-shaped curve where the strongest effect occurs around 675 nm.

At:

```text
w = 675
```

the exponent becomes:

```text
0
```

and:

```python
np.exp(0) = 1
```

Therefore the absorption reaches its maximum around 675 nm.

As wavelength moves away from 675 nm, the exponent becomes more negative and the exponential value decreases.

---

## 14. Backscattering

Turbidity is associated with suspended particles that scatter light.

The simulation models particle backscattering using:

```python
b_bp = (
    0.012 * turb
) * ((400.0 / w) ** 0.5)
```

The important relationship is:

```text
Higher turbidity
       ↓
More suspended particles
       ↓
More backscattering
       ↓
Different spectral reflectance
```

This is one of the relationships the eventual ML model can learn.

---

## 15. Phytoplankton Reflectance Feature

The simulation also creates a phytoplankton-related scattering component around 705 nm:

```python
b_phyto = (
    0.001 * chla
) * np.exp(
    -((w - 705) ** 2) / (2 * 20**2)
)
```

This creates another wavelength-dependent spectral feature.

---

## 16. Total Absorption and Backscattering

The simulation combines the different optical components.

Total absorption:

```python
a_total = (
    a_water
    + a_particles
    + (chla * a_phytoplankton * 0.01)
)
```

Total backscattering:

```python
b_b_total = (
    0.002 * (400.0 / w)**4
    + b_bp
    + b_phyto
)
```

The goal is to produce a simplified but structured relationship between water properties and spectral measurements.

---

## 17. Remote Sensing Reflectance

The simulated reflectance is calculated using:

```python
r_rs = 0.05 * (
    b_b_total /
    (a_total + b_b_total)
)
```

These values become the hyperspectral features:

```text
band_400nm
band_410nm
...
band_900nm
```

---

## 18. Sensor Noise

Real sensors are not perfectly accurate.

Therefore the simulation adds random sensor noise:

```python
sensor_noise = np.random.normal(
    0,
    0.0003,
    size=len(w)
)
```

The noise is added to the reflectance:

```python
r_rs_noisy = np.clip(
    r_rs + sensor_noise,
    0.0001,
    0.15
)
```

This makes the simulated data less perfectly clean and more representative of noisy measurements.

---

## 19. Loading the Dataset

The project uses:

```python
load_dataset()
```

to load the CSV.

The function first checks:

```python
os.path.exists(file_path)
```

If the file exists:

```python
pd.read_csv(file_path)
```

loads it.

If the file does not exist, the project automatically calls:

```python
generate_demo_dataset()
```

to create the dataset.

---

## 20. Automatically Finding Spectral Features

The loader identifies spectral columns using:

```python
band_cols = [
    c for c in df.columns
    if c.startswith("band_") or c.endswith("nm")
]
```

The important idea is that we don't manually type all 51 feature names.

The program finds the spectral columns automatically.

---

## 21. Data Exploration

The exploration code is located in:

```text
src/explore_data.py
```

The notebook is:

```text
notebooks/01_data_exploration.ipynb
```

The purpose of exploration is to understand the dataset before applying machine learning.

Before training a model, we should know:

- How many samples exist?
- How many features exist?
- What is the target distribution?
- What do spectral curves look like?
- Are there obvious problems with the data?
- How does the spectral response vary across samples?

---

## 22. Spectral Signature Visualization

One of the Week 1 outputs is:

```text
results/graphs/01_spectral_signatures.png
```

A spectral signature is essentially a plot of reflectance against wavelength.

```text
Reflectance
    ↑
    |
    |\
    | \
    |  \
    |   \____
    |
    +----------------------→ Wavelength
      400                900 nm
```

The X-axis represents:

```text
Wavelength (nm)
```

The Y-axis represents:

```text
Reflectance
```

This allows us to visually understand how water samples behave across different wavelengths.

---

## 23. Turbidity Distribution Visualization

Another Week 1 output is:

```text
results/graphs/02_turbidity_distribution.png
```

This visualization shows how turbidity values are distributed across the dataset.

This is useful because before training a model, we should understand the target variable.

For example, we want to know whether:

- Most samples have low turbidity
- Some samples have high turbidity
- The target distribution is heavily skewed
- There are unusual/extreme values

---

## 24. Week 1 Files

The main files added during Week 1 were:

```text
data/raw/water_quality_hyperspectral_data.csv
docs/01_ml_concepts_guide.md
notebooks/01_data_exploration.ipynb
requirements.txt
results/graphs/01_spectral_signatures.png
results/graphs/02_turbidity_distribution.png
src/data_loader.py
src/explore_data.py
```

---

## 25. Week 1 Workflow

The overall Week 1 workflow was:

```text
Water-quality problem
        ↓
Understand turbidity
        ↓
Understand hyperspectral measurements
        ↓
Generate/load dataset
        ↓
Identify 51 spectral bands
        ↓
Separate features (X) and target (y)
        ↓
Explore the dataset
        ↓
Visualize spectral signatures
        ↓
Visualize turbidity distribution
        ↓
Prepare for machine learning
```

---

## 26. Important Interview Questions

### Q1. What is the target variable?

```text
turbidity_ntu
```

It represents water turbidity measured in NTU.

### Q2. What are your features?

The 51 hyperspectral bands from:

```text
400 nm → 900 nm
```

at 10 nm intervals.

### Q3. Why are there 51 bands?

Because:

```text
(900 - 400) / 10 + 1 = 51
```

### Q4. What does each row represent?

One water sample and its corresponding spectral measurements.

### Q5. What does `axis=1` do in `pd.concat()`?

It combines DataFrames horizontally by adding their columns side-by-side.

### Q6. Is this real-world sensor data?

No. The current dataset is explicitly labeled:

```text
DEMO DATA - BIO-OPTICAL SIMULATION
```

It is simulated using simplified bio-optical relationships.

### Q7. Why explore data before training a model?

Because we need to understand the data distribution, detect possible problems, understand feature behavior, and verify that the dataset is suitable for the next ML steps.

---

## 27. Week 1 Git Commit

Week 1 was committed using:

```bash
git commit -m "Build hyperspectral data exploration pipeline"
```

The commit contains the initial data exploration pipeline and supporting documentation.

---

## 28. What I Should Understand Before Week 2

Before moving to model training, I should be able to explain:

```text
What is turbidity?
What is NTU?
What is hyperspectral data?
What is a wavelength?
What is reflectance?
What are the 51 spectral bands?
What is X?
What is y?
What is a DataFrame?
Why are df_target and df_bands separate?
Why do we use pd.concat(..., axis=1)?
Why do we add sensor noise?
What is the purpose of the data exploration notebook?
Why do we visualize spectral signatures?
Why do we visualize turbidity distribution?
```

Once these concepts are clear, the project can move to the next stage:

```text
WEEK 2
↓
Preprocessing
↓
Feature Engineering
↓
Model Training
↓
Regression
↓
Model Evaluation
```
