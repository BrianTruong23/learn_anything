import React, { useState } from 'react';
import './ToyExample.css';

const ToyExampleCrossAttention = () => {
  const [hoveredDecoderIdx, setHoveredDecoderIdx] = useState(null);

  const encoderTokens = ["The", "cat", "sat"];
  const decoderTokens = ["Le", "chat", "s'est", "assis"];

  // Mock attention weights (Decoder -> Encoder)
  // Rows: Decoder tokens, Cols: Encoder tokens
  const attentionWeights = [
    [0.9, 0.1, 0.0], // Le -> The
    [0.1, 0.8, 0.1], // chat -> cat
    [0.0, 0.2, 0.8], // s'est -> sat (simplified)
    [0.0, 0.1, 0.9]  // assis -> sat (simplified)
  ];

  const getOpacity = (decoderIdx, encoderIdx) => {
    if (decoderIdx === null) return 0.3; // Default opacity
    return attentionWeights[decoderIdx][encoderIdx];
  };

  return (
    <div className="toy-example-container">
      <div className="explanation-box" style={{ marginBottom: '20px', borderLeft: 'none' }}>
        <p>
          <strong>Cross-Attention</strong> connects the Decoder to the Encoder. 
          The Decoder queries the Encoder's memory to focus on relevant source words.
        </p>
        <div style={{ display: 'flex', justifyContent: 'center', gap: '20px', marginTop: '10px', fontSize: '0.9rem' }}>
          <span style={{ color: '#e65100', fontWeight: 'bold' }}>Q (Queries) from Decoder</span>
          <span style={{ color: '#1976d2', fontWeight: 'bold' }}>K (Keys) & V (Values) from Encoder</span>
        </div>
      </div>

      <div className="visualization-flow" style={{ flexDirection: 'row', justifyContent: 'space-around', alignItems: 'flex-start' }}>
        
        {/* Encoder Output (Keys/Values) */}
        <div className="flow-step" style={{ width: '40%' }}>
          <h4>Encoder Output (Source)</h4>
          <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginBottom: '5px' }}>Keys (K) & Values (V)</div>
          <div className="token-list" style={{ flexDirection: 'column', gap: '15px' }}>
            {encoderTokens.map((token, idx) => (
              <div 
                key={idx} 
                className="token-chip"
                style={{ 
                  backgroundColor: `rgba(33, 150, 243, ${hoveredDecoderIdx !== null ? getOpacity(hoveredDecoderIdx, idx) + 0.2 : 0.2})`,
                  color: hoveredDecoderIdx !== null && getOpacity(hoveredDecoderIdx, idx) > 0.5 ? 'white' : '#1976d2',
                  transition: 'background-color 0.3s'
                }}
              >
                {token}
              </div>
            ))}
          </div>
        </div>

        <div className="arrow" style={{ alignSelf: 'center', display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
          <span>⬅ Attention ⬅</span>
          <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>Q searches K</span>
        </div>

        {/* Decoder Input (Queries) */}
        <div className="flow-step" style={{ width: '40%' }}>
          <h4>Decoder (Target)</h4>
          <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginBottom: '5px' }}>Queries (Q)</div>
          <div className="token-list" style={{ flexDirection: 'column', gap: '15px' }}>
            {decoderTokens.map((token, idx) => (
              <div 
                key={idx} 
                className="token-chip"
                style={{ 
                  cursor: 'pointer',
                  backgroundColor: hoveredDecoderIdx === idx ? '#ff9800' : '#fff3e0',
                  color: hoveredDecoderIdx === idx ? 'white' : '#e65100',
                  border: hoveredDecoderIdx === idx ? '2px solid #f57c00' : '2px solid transparent'
                }}
                onMouseEnter={() => setHoveredDecoderIdx(idx)}
                onMouseLeave={() => setHoveredDecoderIdx(null)}
              >
                {token}
              </div>
            ))}
          </div>
        </div>

      </div>
    </div>
  );
};

export default ToyExampleCrossAttention;
