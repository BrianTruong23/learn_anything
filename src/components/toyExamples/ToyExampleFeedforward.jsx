import React, { useState } from 'react';
import './ToyExampleFeedforward.css';

function ToyExampleFeedforward() {
  const [isTransforming, setIsTransforming] = useState(false);
  const [stage, setStage] = useState(0);

  const handleTransform = () => {
    if (isTransforming) return;
    
    setIsTransforming(true);
    setStage(0);
    
    // Animation sequence
    setTimeout(() => setStage(1), 500);
    setTimeout(() => setStage(2), 1000);
    setTimeout(() => setStage(3), 1500);
    setTimeout(() => {
      setIsTransforming(false);
      setStage(0);
    }, 3000);
  };

  return (
    <div className="feedforward-container">
      <div className="vector-visualization">
        <div className="vector-stage">
          <div className="stage-label">Input Vector</div>
          <div className="vector-bar original" style={{ width: '100px' }}>
            <div className="bar-label">d=512</div>
          </div>
        </div>

        <div className={`arrow-indicator ${stage >= 1 ? 'active' : ''}`}>↓</div>

        <div className="vector-stage">
          <div className={`stage-label ${stage >= 1 ? 'active' : ''}`}>
            Layer 1 (Expand)
          </div>
          <div
            className={`vector-bar expanded ${stage >= 1 ? 'animate' : ''}`}
            style={{ width: stage >= 1 ? '200px' : '100px' }}
          >
            <div className="bar-label">d=2048</div>
          </div>
        </div>

        <div className={`arrow-indicator ${stage >= 2 ? 'active' : ''}`}>↓</div>

        <div className="vector-stage">
          <div className={`stage-label ${stage >= 2 ? 'active' : ''}`}>
            ReLU Activation
          </div>
          <div
            className={`vector-bar relu ${stage >= 2 ? 'animate' : ''}`}
            style={{ width: '200px' }}
          >
            <div className="bar-label">max(0,x)</div>
          </div>
        </div>

        <div className={`arrow-indicator ${stage >= 3 ? 'active' : ''}`}>↓</div>

        <div className="vector-stage">
          <div className={`stage-label ${stage >= 3 ? 'active' : ''}`}>
            Layer 2 (Compress)
          </div>
          <div
            className={`vector-bar compressed ${stage >= 3 ? 'animate' : ''}`}
            style={{ width: stage >= 3 ? '100px' : '200px' }}
          >
            <div className="bar-label">d=512</div>
          </div>
        </div>
      </div>

      <button
        className="transform-btn"
        onClick={handleTransform}
        disabled={isTransforming}
      >
        {isTransforming ? '⚡ Transforming...' : '▶️ Apply MLP'}
      </button>

      <div className="explanation">
        {stage === 0 && !isTransforming && (
          <span>Click "Apply MLP" to see the feedforward transformation!</span>
        )}
        {stage === 1 && <span>🔵 Expanding to higher dimensions (512 → 2048)</span>}
        {stage === 2 && <span>⚡ Applying ReLU: negative values → 0</span>}
        {stage === 3 && <span>🟢 Compressing back to original size (2048 → 512)</span>}
      </div>
    </div>
  );
}

export default ToyExampleFeedforward;
