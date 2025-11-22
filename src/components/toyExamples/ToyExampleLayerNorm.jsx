import React, { useState } from 'react';
import './ToyExampleLayerNorm.css';

function ToyExampleLayerNorm() {
  const [withResidual, setWithResidual] = useState(true);
  const [isFlowing, setIsFlowing] = useState(false);

  const handleToggle = () => {
    setWithResidual(!withResidual);
    triggerFlow();
  };

  const triggerFlow = () => {
    setIsFlowing(true);
    setTimeout(() => setIsFlowing(false), 1500);
  };

  return (
    <div className="layernorm-container">
      <div className="flow-diagram">
        <div className="flow-node input-node">
          <div className="node-label">Input</div>
          <div className="node-box">x</div>
        </div>

        <div className={`flow-path main-path ${isFlowing ? 'flowing' : ''}`}>
          <div className="path-line"></div>
        </div>

        <div className="flow-node layer-node">
          <div className="node-label">Transform Layer</div>
          <div className="node-box">f(x)</div>
        </div>

        {withResidual && (
          <>
            <div className={`residual-path ${isFlowing ? 'flowing' : ''}`}>
              <div className="residual-line"></div>
              <div className="residual-label">Residual Connection</div>
            </div>
            <div className="add-node">+</div>
          </>
        )}

        <div className={`flow-path output-path ${isFlowing ? 'flowing' : ''}`}>
          <div className="path-line"></div>
        </div>

        <div className="flow-node norm-node">
          <div className="node-label">LayerNorm</div>
          <div className="node-box">
            {withResidual ? 'LN(x + f(x))' : 'LN(f(x))'}
          </div>
        </div>

        <div className={`flow-path final-path ${isFlowing ? 'flowing' : ''}`}>
          <div className="path-line"></div>
        </div>

        <div className="flow-node output-node">
          <div className="node-label">Output</div>
          <div className="node-box">y</div>
        </div>
      </div>

      <div className="controls">
        <button className="toggle-btn" onClick={handleToggle}>
          {withResidual ? '✅ With Residual' : '❌ Without Residual'}
        </button>
        <button className="animate-btn" onClick={triggerFlow}>
          🌊 Animate Flow
        </button>
      </div>

      <div className="info-box">
        {withResidual ? (
          <span>
            <strong>With residual connection:</strong> The original input is added back before normalization, 
            creating a "skip connection" that helps gradient flow and preserves information.
          </span>
        ) : (
          <span>
            <strong>Without residual:</strong> Only the transformed output goes through normalization. 
            This can cause information loss and gradient problems in deep networks.
          </span>
        )}
      </div>
    </div>
  );
}

export default ToyExampleLayerNorm;
