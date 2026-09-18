import React from 'react';
import { Activity, AlertTriangle, ShieldCheck } from 'lucide-react';

export default function Header({ apiHealthy }) {
  return (
    <header className="glass-card" style={{ padding: '16px 24px', marginBottom: '24px', borderRadius: '12px' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '12px' }}>
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
            <Activity style={{ color: '#3b82f6' }} size={28} />
            <h1 style={{ fontSize: '1.4rem', fontWeight: 700, background: 'linear-gradient(90deg, #fff, #9ca3af)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent' }}>
              Hyperspectral Water-Quality Monitoring
            </h1>
          </div>
          <p style={{ fontSize: '0.85rem', color: '#9ca3af', marginTop: '4px' }}>
            Real-time ML inference from submitted hyperspectral samples.
          </p>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
          {/* DEMO / SYNTHETIC DATA TAG */}
          <div className="tag-demo">
            <AlertTriangle size={14} />
            DEMO DATA - BIO-OPTICAL SIMULATION
          </div>

          {/* API Health Status */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '0.85rem', color: apiHealthy ? '#10b981' : '#ef4444' }}>
            <ShieldCheck size={18} />
            <span>FastAPI: {apiHealthy ? 'Connected' : 'Offline'}</span>
          </div>
        </div>
      </div>
    </header>
  );
}
