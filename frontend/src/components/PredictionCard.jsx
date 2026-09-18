import React from 'react';
import { Droplets, Cpu, AlertCircle } from 'lucide-react';

export default function PredictionCard({ prediction, isLoading, error }) {
  if (isLoading) {
    return (
      <div className="glass-card" style={{ padding: '24px', height: '100%', display: 'flex', flexDirection: 'column', justifyContent: 'center', alignItems: 'center', minHeight: '260px' }}>
        <div style={{ width: '40px', height: '40px', border: '3px solid rgba(255,255,255,0.1)', borderTop: '3px solid #3b82f6', borderRadius: '50%', animation: 'spin 1s linear infinite' }} />
        <p style={{ marginTop: '16px', color: '#9ca3af', fontSize: '0.9rem' }}>Executing ML Inference Pipeline...</p>
        <style>{`@keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }`}</style>
      </div>
    );
  }

  if (error) {
    return (
      <div className="glass-card" style={{ padding: '24px', borderLeft: '4px solid #ef4444', minHeight: '260px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px', color: '#ef4444', marginBottom: '12px' }}>
          <AlertCircle size={24} />
          <h3 style={{ fontSize: '1.1rem', fontWeight: 600 }}>API Inference Error</h3>
        </div>
        <p style={{ fontSize: '0.9rem', color: '#d1d5db', lineHeight: 1.6 }}>{error}</p>
        <p style={{ fontSize: '0.8rem', color: '#9ca3af', marginTop: '12px' }}>
          Check that the FastAPI backend server is running at <code>http://127.0.0.1:8000</code>.
        </p>
      </div>
    );
  }

  if (!prediction) {
    return (
      <div className="glass-card" style={{ padding: '24px', minHeight: '260px', display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#9ca3af' }}>
        Select a sample preset or upload a CSV to run inference.
      </div>
    );
  }

  const { predicted_turbidity, unit, quality_status, model_architecture, data_source } = prediction;

  // Determine status color
  let statusColor = '#10b981'; // Green for clear
  if (predicted_turbidity >= 15.0) statusColor = '#ef4444'; // Red for turbid
  else if (predicted_turbidity >= 5.0) statusColor = '#f59e0b'; // Yellow for moderate

  return (
    <div className="glass-card" style={{ padding: '24px', display: 'flex', flexDirection: 'column', justifyContent: 'space-between', minHeight: '260px', position: 'relative' }}>
      <div>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
          <span style={{ fontSize: '0.85rem', fontWeight: 600, color: '#9ca3af', textTransform: 'uppercase', letterSpacing: '1px' }}>
            Predicted Water Quality
          </span>
          <Droplets style={{ color: statusColor }} size={24} />
        </div>

        {/* Large Readout */}
        <div style={{ display: 'flex', alignItems: 'baseline', gap: '8px', marginBottom: '12px' }}>
          <span style={{ fontSize: '3.2rem', fontWeight: 800, letterSpacing: '-1px', color: '#ffffff' }}>
            {predicted_turbidity.toFixed(2)}
          </span>
          <span style={{ fontSize: '1.2rem', fontWeight: 600, color: '#9ca3af' }}>{unit}</span>
        </div>

        {/* Status Badge */}
        <div style={{ display: 'inline-flex', alignItems: 'center', gap: '8px', padding: '6px 14px', borderRadius: '20px', backgroundColor: `${statusColor}20`, border: `1px solid ${statusColor}40`, marginBottom: '16px' }}>
          <span style={{ width: '8px', height: '8px', borderRadius: '50%', backgroundColor: statusColor }} />
          <span style={{ fontSize: '0.9rem', fontWeight: 600, color: statusColor }}>{quality_status}</span>
        </div>
      </div>

      {/* Model Metadata */}
      <div style={{ borderTop: '1px solid rgba(255,255,255,0.08)', paddingTop: '16px', display: 'flex', flexDirection: 'column', gap: '6px', fontSize: '0.8rem', color: '#9ca3af' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <Cpu size={14} style={{ color: '#3b82f6' }} />
          <span>Model: <strong>{model_architecture}</strong></span>
        </div>
        <div style={{ color: '#fbbf24', fontStyle: 'italic', marginTop: '2px' }}>
          Tag: {data_source}
        </div>
      </div>
    </div>
  );
}
