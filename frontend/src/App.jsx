import React, { useState, useEffect } from 'react';
import axios from 'axios';
import Header from './components/Header';
import PredictionCard from './components/PredictionCard';
import SpectralChart from './components/SpectralChart';
import SamplePresets from './components/SamplePresets';
import CsvUploader from './components/CsvUploader';
import LiveStreamControls from './components/LiveStreamControls';

const API_BASE_URL = 'http://127.0.0.1:8000';

export default function App() {
  const [apiHealthy, setApiHealthy] = useState(false);
  const [activePresetId, setActivePresetId] = useState('clear');
  const [activeSampleName, setActiveSampleName] = useState('Clear Water Sample');
  const [activeBandData, setActiveBandData] = useState(null);
  
  const [prediction, setPrediction] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [errorMsg, setErrorMsg] = useState(null);
  const [isStreaming, setIsStreaming] = useState(false);

  // 1. Check API Health on Mount
  useEffect(() => {
    checkHealth();
    loadInitialDefault();
  }, []);

  const checkHealth = async () => {
    try {
      const res = await axios.get(`${API_BASE_URL}/health`);
      setApiHealthy(res.data.status === 'healthy');
    } catch {
      setApiHealthy(false);
    }
  };

  const loadInitialDefault = () => {
    const defaultBands = {};
    for (let w = 400; w <= 900; w += 10) {
      const base = 0.005 * Math.exp(-(w - 400) / 200) + 0.001;
      defaultBands[`band_${w}nm`] = Number(base.toFixed(6));
    }
    setActiveBandData(defaultBands);
    runInference(defaultBands);
  };

  // 2. Execute Axios POST /predict Inference Call
  const runInference = async (bands) => {
    setIsLoading(true);
    setErrorMsg(null);

    try {
      const response = await axios.post(`${API_BASE_URL}/predict`, { bands });
      setPrediction(response.data);
    } catch (err) {
      if (err.response) {
        setErrorMsg(err.response.data.detail || "FastAPI validation error.");
      } else {
        setErrorMsg(`Cannot connect to FastAPI server at ${API_BASE_URL}. Ensure 'uvicorn backend.main:app' is running.`);
      }
    } finally {
      setIsLoading(false);
    }
  };

  // Handler for Sample Preset Selection
  const handleSelectPreset = (presetId, name, bands) => {
    if (isStreaming) return; // Ignore preset changes during active stream
    setActivePresetId(presetId);
    setActiveSampleName(name);
    setActiveBandData(bands);
    runInference(bands);
  };

  // Handler for CSV File Upload
  const handleCsvUpload = (fileName, bands) => {
    if (isStreaming) return;
    setActivePresetId(null);
    setActiveSampleName(`CSV Upload: ${fileName}`);
    setActiveBandData(bands);
    runInference(bands);
  };

  // Handler for Live Stream Simulation Frame Updates
  const handleLiveFrameReceived = (frameName, bands, predictionData) => {
    setActivePresetId(null);
    setActiveSampleName(frameName);
    setActiveBandData(bands);
    setPrediction(predictionData);
    setIsLoading(false);
    setErrorMsg(null);
  };

  // Handler for Live Stream Toggle
  const handleStreamToggle = (streamingState) => {
    setIsStreaming(streamingState);
    if (!streamingState) {
      // Stream stopped -> Return UI to default state
      loadInitialDefault();
    }
  };

  return (
    <div style={{ maxWidth: '1200px', margin: '0 auto', padding: '24px 16px' }}>
      <Header apiHealthy={apiHealthy} />

      {/* Camera Stream Emulator Component */}
      <LiveStreamControls 
        onFrameReceived={handleLiveFrameReceived} 
        onStreamToggle={handleStreamToggle}
        isStreaming={isStreaming}
      />

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))', gap: '24px', marginBottom: '24px' }}>
        {/* Left Column: Sample Controls & CSV Upload */}
        <div style={{ opacity: isStreaming ? 0.5 : 1, pointerEvents: isStreaming ? 'none' : 'auto', transition: 'opacity 0.3s ease' }}>
          <SamplePresets onSelectPreset={handleSelectPreset} currentPresetId={activePresetId} />
          <CsvUploader onCsvUpload={handleCsvUpload} />
        </div>

        {/* Right Column: Prediction Readout Card */}
        <div>
          <PredictionCard prediction={prediction} isLoading={isLoading} error={errorMsg} />
        </div>
      </div>

      {/* Full Width 51-Band Spectral Signature Chart */}
      <SpectralChart bandData={activeBandData} sampleName={activeSampleName} />

      <footer style={{ textAlign: 'center', padding: '20px 0', fontSize: '0.8rem', color: '#6b7280', borderTop: '1px solid rgba(255,255,255,0.05)', marginTop: '40px' }}>
        Hyperspectral Imaging & ML Water-Quality System | SIMULATION - HARDWARE CAMERA STREAM EMULATOR
      </footer>
    </div>
  );
}
