# Real-Time Data Streaming & Camera Simulation Concepts

Welcome to Phase 7! In this guide, we explore **Real-Time Data Streaming** and how we simulate an edge hyperspectral camera feed.

---

## 1. What a Data Stream Is

- **Data Stream**: A continuous, unbroken sequence of data packets transmitted over time from a sensor or source to a processing system.
- **In our project**: Rather than manually uploading a single CSV file, a stream automatically delivers continuous observations over time.

---

## 2. What a "Frame" Means in Hyperspectral Imaging

- **Frame**: A single complete snapshot measurement captured at a specific instant in time.
- **In our project**: A frame consists of **51 continuous reflectance values** across wavelengths $400nm$ to $900nm$ (`band_400nm` through `band_900nm`) recorded at timestamp $t$.

---

## 3. How the Simulator Generates Frames

- `HyperspectralCameraSimulator` in `src/camera_simulator.py` uses bio-optical physics equations ($R_{rs}(\lambda) \propto \frac{b_b}{a + b_b}$) with time-varying sediment/water flow perturbations to generate realistic 51-band spectral signatures.
- Each generated frame contains natural fluctuations (e.g. turbidity drifting between $3.5 \text{ NTU}$ and $22.0 \text{ NTU}$).

---

## 4. How the 2-Second Interval Works

- **Backend**: The simulator updates its simulated optical sensor state every 2 seconds.
- **Frontend**: The React dashboard uses a JavaScript `setInterval(fetchNextFrame, 2000)` timer to request a new frame every 2,000 milliseconds (2 seconds).
- **Clean Teardown**: When the user disables the stream, `clearInterval(timerId)` is called immediately. This stops all background network requests and prevents memory leaks.

---

## 5. How the Existing Trained Model Is Reused

- **Zero Retraining**: The simulator reuses our existing `HyperspectralPredictionPipeline` class from `src/predict.py`.
- **Inference Sequence**:
  ```
  New Frame (51 Bands) -> scaler.transform() -> pca.transform() -> best_model.predict()
  ```
- The in-memory model (`best_model.joblib`) computes predictions in milliseconds without calling `.fit()`.

---

## 6. Why This Is Simulation vs. True Hardware Monitoring

> **IMPORTANT SCIENTIFIC TRANSPARENCY DISTINCTION**:
>
> 1. **Current System (`SIMULATION - HARDWARE CAMERA STREAM EMULATOR`)**:
>    - Generates synthetic bio-optical spectral frames via software.
>    - Demonstrates real-time streaming architecture, API polling, and dynamic UI chart updating.
> 
> 2. **True Hardware-Based Real-Time Monitoring (Future Physical Integration)**:
>    - Requires an actual physical Hyperspectral Imager (e.g. Specim IQ, Resonon, or Headwall sensor) mounted over a river or water pipe.
>    - Requires physical frame grabber drivers, USB/GigE optical camera SDKs, and calibrated radiance-to-reflectance conversion panels.

By clearly tagging all stream outputs as `SIMULATION - HARDWARE CAMERA STREAM EMULATOR`, we preserve complete scientific honesty.
