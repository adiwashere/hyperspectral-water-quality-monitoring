import React from 'react';
import { ResponsiveContainer, LineChart, Line, XAxis, YAxis, Tooltip, CartesianGrid } from 'recharts';

export default function SpectralChart({ bandData, sampleName }) {
  if (!bandData || Object.keys(bandData).length === 0) {
    return (
      <div className="glass-card" style={{ padding: '24px', height: '350px', display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#9ca3af' }}>
        No spectral data loaded. Select a preset or upload a CSV file.
      </div>
    );
  }

  // Convert 51-band dictionary into Recharts array format [{ wavelength: 400, reflectance: 0.0048 }, ...]
  const chartData = Object.entries(bandData).map(([key, val]) => {
    const wavelength = parseInt(key.replace('band_', '').replace('nm', ''), 10);
    return {
      wavelength: wavelength,
      reflectance: Number(val)
    };
  }).sort((a, b) => a.wavelength - b.wavelength);

  return (
    <div className="glass-card" style={{ padding: '24px', marginBottom: '24px' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' }}>
        <div>
          <h3 style={{ fontSize: '1.1rem', fontWeight: 700, color: '#f3f4f6' }}>
            Hyperspectral Reflectance Signature (51 Bands)
          </h3>
          <p style={{ fontSize: '0.8rem', color: '#9ca3af', marginTop: '2px' }}>
            Sample ID: <strong>{sampleName || 'Custom Sample'}</strong> | Spectral Range: 400nm (Violet) to 900nm (NIR)
          </p>
        </div>
      </div>

      <div style={{ width: '100%', height: 320 }}>
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={chartData} margin={{ top: 10, right: 30, left: 10, bottom: 10 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255, 255, 255, 0.08)" />
            <XAxis 
              dataKey="wavelength" 
              stroke="#9ca3af" 
              unit="nm" 
              tick={{ fontSize: 12 }}
              label={{ value: 'Wavelength (nm)', position: 'insideBottom', offset: -5, fill: '#9ca3af', fontSize: 12 }}
            />
            <YAxis 
              stroke="#9ca3af" 
              tick={{ fontSize: 12 }}
              tickFormatter={(val) => val.toFixed(4)}
              label={{ value: 'Reflectance R_rs (sr⁻¹)', angle: -90, position: 'insideLeft', fill: '#9ca3af', fontSize: 12 }}
            />
            <Tooltip 
              contentStyle={{ backgroundColor: '#1f2937', border: '1px solid rgba(255,255,255,0.15)', borderRadius: '8px', color: '#fff' }}
              formatter={(value) => [Number(value).toFixed(6), 'Reflectance (R_rs)']}
              labelFormatter={(label) => `Wavelength: ${label} nm`}
            />
            <Line 
              type="monotone" 
              dataKey="reflectance" 
              stroke="#3b82f6" 
              strokeWidth={2.5} 
              dot={{ r: 2, fill: '#60a5fa' }} 
              activeDot={{ r: 6, fill: '#8b5cf6' }} 
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
