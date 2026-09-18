# Web API & FastAPI Concepts

Welcome to Phase 5! In this guide, we transition from standalone Python ML scripts to a web-accessible **FastAPI Application Programming Interface (API)**.

---

## 1. What an API Is

- **API (Application Programming Interface)**: A structured messenger that allows two different software programs to talk to each other over a network.
- **In our project**: The API sits between our **Machine Learning Prediction Pipeline** (written in Python) and our **User Dashboard** (written in React/JavaScript).

---

## 2. What FastAPI Does

- **FastAPI**: A modern, high-performance Python web framework for building APIs based on standard Python type hints.
- **Why we use it**:
  1. **High Speed**: Built on Starlette and Asynchronous I/O (ASGI), making it one of the fastest Python frameworks available.
  2. **Automatic Data Validation**: Uses **Pydantic** to inspect incoming JSON requests and automatically reject malformed inputs before they hit your code.
  3. **Automatic Documentation**: Automatically generates interactive OpenAPI (Swagger UI) documentation at `/docs`.

---

## 3. What an Endpoint Is

- **Endpoint**: A specific URL path on the server dedicated to performing a particular task.
- **Examples in our project**:
  - `http://127.0.0.1:8000/health`: Checks if server is alive.
  - `http://127.0.0.1:8000/model-info`: Returns model architecture metadata.
  - `http://127.0.0.1:8000/predict`: Receives 51 spectral bands and returns predicted Turbidity.

---

## 4. What GET vs. POST Means

HTTP methods define what type of operation the client wants to perform:

### `GET`
- **Purpose**: Fetching or retrieving data from the server.
- **Characteristics**: Read-only, safe, idempotent. Sends parameters in URL.
- **Example**: `GET /health` or `GET /model-info`.

### `POST`
- **Purpose**: Sending data to the server to perform processing or create a record.
- **Characteristics**: Transmits data in the request body (usually JSON format).
- **Example**: `POST /predict` sending `{ "bands": { "band_400nm": 0.005, ... } }`.

---

## 5. How React Will Communicate with This API

1. In React, when a user uploads a hyperspectral sample or clicks "Analyze Water", React uses a library like `Axios` or `fetch()`:
   ```javascript
   const response = await axios.post("http://127.0.0.1:8000/predict", {
       bands: spectralData
   });
   console.log("Predicted Turbidity:", response.data.predicted_turbidity);
   ```
2. FastAPI processes the JSON request in Python and returns JSON:
   ```json
   {
     "predicted_turbidity": 4.17,
     "unit": "NTU",
     "quality_status": "Clear Water (Low Turbidity)"
   }
   ```
3. React renders the `4.17 NTU` number on a chart or card widget.

---

## 6. How the API Loads Saved Models (No Re-training!)

When the FastAPI server starts up (`uvicorn backend.main:app`), it executes a **startup lifecycle handler**:

```python
# Executed ONCE at server startup
pipeline = HyperspectralPredictionPipeline()
```

- `pipeline.__init__()` loads `scaler.joblib`, `pca_transformer.joblib`, and `best_model.joblib` from disk into server RAM.
- When 1,000 users make `POST /predict` requests, the server reuses the **already-loaded** model objects in memory to execute instant `model.predict()` calls in milliseconds!
- **No re-fitting or retraining happens per request.**
