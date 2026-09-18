# React Frontend Dashboard Concepts

Welcome to Phase 6! In this guide, we break down how our **React Monitoring Dashboard** connects to our **FastAPI Machine Learning Server**.

---

## 1. Terminology & System Description

> **CRITICAL ARCHITECTURAL NOTE**:
> Current system functionality is described as:
> **"Real-time ML inference from submitted hyperspectral samples."**
> Continuous automated monitoring will be connected in future phases when a physical hyperspectral camera stream is attached.

---

## 2. Component Structure

The frontend application is structured into modular components:

```
frontend/src/
├── components/
│   ├── Header.jsx             <- Top banner with DEMO DATA tag & API status
│   ├── PredictionCard.jsx     <- Displays predicted Turbidity (NTU) & Model Info
│   ├── SpectralChart.jsx      <- Recharts 51-band curve (400nm - 900nm)
│   ├── SamplePresets.jsx      <- Quick presets (Clear Water vs. Turbid Water)
│   └── CsvUploader.jsx        <- CSV file upload & 51-band client-side validator
├── App.jsx                    <- Main Dashboard layout & centralized state manager
├── index.css                  <- Global CSS design system & typography
└── main.jsx                   <- React root entry point
```

---

## 3. How React State Works (`useState`)

React components use **State** (`useState`) to store dynamic data that triggers UI updates whenever it changes:

- `activeSpectrum`: Stores the current 51-band reflectance array.
- `prediction`: Stores the JSON response returned by FastAPI (`predicted_turbidity`, `quality_status`, `model_architecture`).
- `isLoading`: Boolean flag indicating when an Axios API request is in-flight (shows a spinner/skeleton loader).
- `errorMessage`: Stores validation or API connection errors to display user-friendly alert badges.

---

## 4. How Axios Communicates with FastAPI

```javascript
import axios from 'axios';

const runPrediction = async (bandData) => {
    setIsLoading(true);
    setErrorMessage(null);
    
    try {
        // HTTP POST request to FastAPI backend
        const response = await axios.post("http://127.0.0.1:8000/predict", {
            bands: bandData
        });
        
        // Update React state with prediction response
        setPrediction(response.data);
    } catch (error) {
        if (error.response) {
            // Server responded with 400 or 422 HTTP Error
            setErrorMessage(error.response.data.detail || "Validation Error");
        } else {
            // Network connection failure
            setErrorMessage("Cannot connect to FastAPI backend at http://127.0.0.1:8000. Ensure server is running.");
        }
    } finally {
        setIsLoading(false);
    }
};
```

---

## 5. How the Spectral Chart Works (Recharts)

- Recharts requires an array of data objects, e.g.:
  `[{ wavelength: 400, reflectance: 0.0048 }, { wavelength: 410, reflectance: 0.0051 }, ...]`
- We convert our 51-band dictionary/array into this format using `Object.entries()` or mapping over wavelengths ($400nm - 900nm$).
- The `<LineChart>` component renders a smooth curve with interactive tooltips displaying wavelength and reflectance values.

---

## 6. How CSV Parsing & Validation Works

When a user uploads a `.csv` file:
1. FileReader reads the raw text.
2. The parser splits lines and extracts header columns.
3. **Strict Validation Check**:
   - Verifies that all 51 required columns (`band_400nm` through `band_900nm`) exist.
   - Verifies that values are valid numeric floats.
   - If columns are missing or values are malformed, it sets `errorMessage` and rejects the upload without sending broken data to the server.
