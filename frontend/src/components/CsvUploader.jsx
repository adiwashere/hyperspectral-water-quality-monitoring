import React, { useState } from 'react';
import { Upload, FileText, AlertOctagon, CheckCircle2 } from 'lucide-react';

const EXPECTED_BANDS = Array.from({ length: 51 }, (_, i) => `band_${400 + i * 10}nm`);

export default function CsvUploader({ onCsvUpload }) {
  const [fileName, setFileName] = useState(null);
  const [errorMsg, setErrorMsg] = useState(null);
  const [successMsg, setSuccessMsg] = useState(null);

  const handleFileUpload = (event) => {
    const file = event.target.files[0];
    if (!file) return;

    setFileName(file.name);
    setErrorMsg(null);
    setSuccessMsg(null);

    const reader = new FileReader();
    reader.onload = (e) => {
      try {
        const text = e.target.result;
        const lines = text.split(/\r\n|\n/).filter(line => line.trim() !== '');

        if (lines.length < 2) {
          throw new Error("CSV file must contain a header row and at least one data row.");
        }

        const headers = lines[0].split(',').map(h => h.trim().replace(/^"|"$/g, ''));
        const values = lines[1].split(',').map(v => v.trim().replace(/^"|"$/g, ''));

        // Validate 51 Spectral Bands Presence
        const bandDict = {};
        const missingBands = [];
        const nonNumericBands = [];

        EXPECTED_BANDS.forEach(bandKey => {
          const colIndex = headers.indexOf(bandKey);
          if (colIndex === -1) {
            missingBands.push(bandKey);
          } else {
            const rawVal = values[colIndex];
            const numVal = parseFloat(rawVal);
            if (isNaN(numVal)) {
              nonNumericBands.push(`${bandKey} ('${rawVal}')`);
            } else {
              bandDict[bandKey] = numVal;
            }
          }
        });

        // Error Handling Checks
        if (missingBands.length > 0) {
          throw new Error(`CSV is missing required spectral band columns: ${missingBands.slice(0, 3).join(', ')}... (Total Missing: ${missingBands.length})`);
        }

        if (nonNumericBands.length > 0) {
          throw new Error(`CSV contains non-numeric values in band columns: ${nonNumericBands.slice(0, 3).join(', ')}.`);
        }

        if (Object.keys(bandDict).length !== 51) {
          throw new Error(`Validation failed: Exactly 51 spectral bands required, found ${Object.keys(bandDict).length}.`);
        }

        setSuccessMsg(`Successfully validated 51 spectral bands from '${file.name}'.`);
        onCsvUpload(file.name, bandDict);

      } catch (err) {
        setErrorMsg(err.message);
      }
    };

    reader.readAsText(file);
  };

  return (
    <div className="glass-card" style={{ padding: '24px', marginBottom: '24px' }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '12px' }}>
        <Upload size={20} style={{ color: '#8b5cf6' }} />
        <h3 style={{ fontSize: '1.1rem', fontWeight: 700, color: '#f3f4f6' }}>
          Upload Hyperspectral Sample (.CSV)
        </h3>
      </div>
      <p style={{ fontSize: '0.82rem', color: '#9ca3af', marginBottom: '16px' }}>
        Upload a CSV file containing columns <code>band_400nm</code> to <code>band_900nm</code>.
      </p>

      <div style={{ border: '2px dashed rgba(255, 255, 255, 0.15)', borderRadius: '12px', padding: '20px', textAlign: 'center', backgroundColor: 'rgba(0,0,0,0.2)' }}>
        <input 
          type="file" 
          accept=".csv" 
          onChange={handleFileUpload} 
          id="csv-file-input" 
          style={{ display: 'none' }} 
        />
        <label htmlFor="csv-file-input" className="btn-secondary" style={{ cursor: 'pointer' }}>
          <FileText size={16} />
          {fileName ? `Selected: ${fileName}` : 'Choose CSV File'}
        </label>
      </div>

      {/* Error Badge */}
      {errorMsg && (
        <div style={{ marginTop: '16px', padding: '12px', borderRadius: '8px', backgroundColor: 'rgba(239, 68, 68, 0.15)', border: '1px solid rgba(239, 68, 68, 0.3)', color: '#fca5a5', fontSize: '0.85rem', display: 'flex', alignItems: 'center', gap: '10px' }}>
          <AlertOctagon size={18} style={{ shrink: 0 }} />
          <span>{errorMsg}</span>
        </div>
      )}

      {/* Success Badge */}
      {successMsg && (
        <div style={{ marginTop: '16px', padding: '12px', borderRadius: '8px', backgroundColor: 'rgba(16, 185, 129, 0.15)', border: '1px solid rgba(16, 185, 129, 0.3)', color: '#6ee7b7', fontSize: '0.85rem', display: 'flex', alignItems: 'center', gap: '10px' }}>
          <CheckCircle2 size={18} style={{ shrink: 0 }} />
          <span>{successMsg}</span>
        </div>
      )}
    </div>
  );
}
