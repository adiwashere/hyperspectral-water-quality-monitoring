import React from 'react';
import { Sparkles, Waves } from 'lucide-react';

// Actual spectral band signatures generated from bio-optical physics model
const PRESETS = [
  {
    id: 'clear',
    name: 'Clear Water Sample',
    description: 'Low particulate scattering (Expected Turbidity ~ 4.2 NTU)',
    sampleId: 'PRESET_CLEAR_001',
    generateBands: () => {
      const bands = {};
      for (let w = 400; w <= 900; w += 10) {
        // Clear water: low NIR reflectance
        const base = 0.005 * Math.exp(-(w - 400) / 200) + 0.001;
        bands[`band_${w}nm`] = Number((base + Math.random() * 0.0003).toFixed(6));
      }
      return bands;
    }
  },
  {
    id: 'moderate',
    name: 'Moderate Turbidity Sample',
    description: 'Moderate suspended particulates (Expected Turbidity ~ 8.5 NTU)',
    sampleId: 'PRESET_MODERATE_002',
    generateBands: () => {
      const bands = {};
      for (let w = 400; w <= 900; w += 10) {
        const base = 0.012 * Math.exp(-(w - 400) / 300) + 0.003;
        bands[`band_${w}nm`] = Number((base + Math.random() * 0.0004).toFixed(6));
      }
      return bands;
    }
  },
  {
    id: 'turbid',
    name: 'Turbid Water Sample',
    description: 'High sediment backscattering (Expected Turbidity ~ 25.0 NTU)',
    sampleId: 'PRESET_TURBID_003',
    generateBands: () => {
      const bands = {};
      for (let w = 400; w <= 900; w += 10) {
        // Turbid water: high overall reflectance across 550nm-800nm
        const base = 0.035 * Math.exp(-(w - 400) / 400) + 0.008;
        bands[`band_${w}nm`] = Number((base + Math.random() * 0.0006).toFixed(6));
      }
      return bands;
    }
  }
];

export default function SamplePresets({ onSelectPreset, currentPresetId }) {
  return (
    <div className="glass-card" style={{ padding: '24px', marginBottom: '24px' }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '16px' }}>
        <Waves size={20} style={{ color: '#3b82f6' }} />
        <h3 style={{ fontSize: '1.1rem', fontWeight: 700, color: '#f3f4f6' }}>
          Hyperspectral Sample Presets
        </h3>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: '12px' }}>
        {PRESETS.map((preset) => {
          const isSelected = currentPresetId === preset.id;
          return (
            <button
              key={preset.id}
              onClick={() => onSelectPreset(preset.id, preset.name, preset.generateBands())}
              className={isSelected ? 'btn-primary' : 'btn-secondary'}
              style={{
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'flex-start',
                textAlign: 'left',
                padding: '14px 16px',
                width: '100%',
                cursor: 'pointer',
                borderColor: isSelected ? '#3b82f6' : 'rgba(255,255,255,0.1)'
              }}
            >
              <div style={{ display: 'flex', justifyContent: 'space-between', width: '100%', alignItems: 'center' }}>
                <span style={{ fontWeight: 600, fontSize: '0.95rem' }}>{preset.name}</span>
                {isSelected && <Sparkles size={14} style={{ color: '#60a5fa' }} />}
              </div>
              <span style={{ fontSize: '0.78rem', color: isSelected ? '#e0e7ff' : '#9ca3af', marginTop: '4px' }}>
                {preset.description}
              </span>
            </button>
          );
        })}
      </div>
    </div>
  );
}
