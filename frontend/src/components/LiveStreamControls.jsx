import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import { Play, Square, Video, Radio } from 'lucide-react';

const API_STREAM_URL = 'http://127.0.0.1:8000/stream/next-frame';

export default function LiveStreamControls({ onFrameReceived, onStreamToggle, isStreaming }) {
  const [lastFrameInfo, setLastFrameInfo] = useState(null);
  const [streamError, setStreamError] = useState(null);
  const intervalRef = useRef(null);

  // Clean teardown on unmount
  useEffect(() => {
    return () => {
      stopStream();
    };
  }, []);

  const fetchNextFrame = async () => {
    try {
      const response = await axios.get(API_STREAM_URL);
      const frameData = response.data;
      
      setLastFrameInfo({
        frameId: frameData.frame_id,
        timestamp: frameData.timestamp
      });
      setStreamError(null);

      // Pass updated bands and prediction to main App dashboard
      onFrameReceived(
        `Stream: ${frameData.frame_id} (${frameData.timestamp})`,
        frameData.bands,
        frameData.prediction
      );
    } catch (err) {
      setStreamError("Failed to fetch stream frame from FastAPI. Ensure server is running.");
    }
  };

  const startStream = () => {
    onStreamToggle(true);
    setStreamError(null);
    
    // Immediate first fetch
    fetchNextFrame();

    // Set 2-second interval polling
    intervalRef.current = setInterval(() => {
      fetchNextFrame();
    }, 2000);
  };

  const stopStream = () => {
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
    setLastFrameInfo(null);
    setStreamError(null);
    onStreamToggle(false);
  };

  const toggleStream = () => {
    if (isStreaming) {
      stopStream();
    } else {
      startStream();
    }
  };

  return (
    <div className="glass-card" style={{ padding: '24px', marginBottom: '24px', border: isStreaming ? '1px solid rgba(59, 130, 246, 0.4)' : '1px solid var(--border-card)' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '16px' }}>
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
            <Video size={22} style={{ color: isStreaming ? '#3b82f6' : '#9ca3af' }} />
            <h3 style={{ fontSize: '1.1rem', fontWeight: 700, color: '#f3f4f6' }}>
              Camera Stream Emulator
            </h3>
            {isStreaming && (
              <span style={{ display: 'inline-flex', alignItems: 'center', gap: '6px', backgroundColor: 'rgba(239, 68, 68, 0.2)', color: '#f87171', border: '1px solid rgba(239, 68, 68, 0.4)', padding: '2px 10px', borderRadius: '12px', fontSize: '0.75rem', fontWeight: 700 }}>
                <Radio size={12} style={{ animation: 'pulse 1.5s infinite' }} />
                STREAM ACTIVE (2s INTERVAL)
              </span>
            )}
          </div>
          <p style={{ fontSize: '0.8rem', color: '#9ca3af', marginTop: '4px' }}>
            Tag: <strong>SIMULATION - HARDWARE CAMERA STREAM EMULATOR</strong>
          </p>
        </div>

        {/* Toggle Button */}
        <button
          onClick={toggleStream}
          className={isStreaming ? 'btn-secondary' : 'btn-primary'}
          style={{
            backgroundColor: isStreaming ? 'rgba(239, 68, 68, 0.2)' : undefined,
            color: isStreaming ? '#f87171' : undefined,
            borderColor: isStreaming ? 'rgba(239, 68, 68, 0.4)' : undefined,
            padding: '10px 22px'
          }}
        >
          {isStreaming ? (
            <>
              <Square size={16} /> Stop Camera Stream
            </>
          ) : (
            <>
              <Play size={16} /> Start Live Camera Stream
            </>
          )}
        </button>
      </div>

      {/* Stream Status Readout */}
      {isStreaming && lastFrameInfo && (
        <div style={{ marginTop: '16px', paddingTop: '16px', borderTop: '1px solid rgba(255,255,255,0.08)', display: 'flex', justifyContent: 'space-between', alignItems: 'center', fontSize: '0.85rem', color: '#d1d5db' }}>
          <div>
            Frame ID: <code style={{ color: '#60a5fa' }}>{lastFrameInfo.frameId}</code>
          </div>
          <div>
            Latest Acquisition: <strong style={{ color: '#f3f4f6' }}>{lastFrameInfo.timestamp}</strong>
          </div>
        </div>
      )}

      {streamError && (
        <div style={{ marginTop: '14px', padding: '10px 14px', borderRadius: '8px', backgroundColor: 'rgba(239, 68, 68, 0.15)', border: '1px solid rgba(239, 68, 68, 0.3)', color: '#fca5a5', fontSize: '0.82rem' }}>
          {streamError}
        </div>
      )}

      <style>{`
        @keyframes pulse {
          0% { opacity: 1; }
          50% { opacity: 0.3; }
          100% { opacity: 1; }
        }
      `}</style>
    </div>
  );
}
