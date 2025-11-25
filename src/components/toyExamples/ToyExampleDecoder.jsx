import React, { useState, useEffect } from 'react';
import './ToyExample.css';


const ToyExampleDecoder = () => {
  const [isAnimating, setIsAnimating] = useState(false);
  const [step, setStep] = useState(0);

  const steps = [
    { id: 0, label: "Input (Shifted)", desc: "Target Vector Y (shifted right)" },
    { id: 1, label: "Masked Self-Attention", desc: "Looking at past tokens only (Future masked 🙈)" },
    { id: 2, label: "Add & Norm 1", desc: "Residual + Norm" },
    { id: 3, label: "Cross-Attention", desc: "Looking at Encoder Output (Memory 🧠)" },
    { id: 4, label: "Add & Norm 2", desc: "Residual + Norm" },
    { id: 5, label: "Feed-Forward", desc: "Processing features" },
    { id: 6, label: "Add & Norm 3", desc: "Residual + Norm" },
    { id: 7, label: "Output", desc: "Next Token Probability" }
  ];

  useEffect(() => {
    let timer;
    if (isAnimating) {
      timer = setInterval(() => {
        setStep(prev => {
          if (prev >= steps.length - 1) {
            setIsAnimating(false);
            return prev;
          }
          return prev + 1;
        });
      }, 1200); // Slightly slower to read descriptions
    }
    return () => clearInterval(timer);
  }, [isAnimating, steps.length]);

  const startAnimation = () => {
    setStep(0);
    setIsAnimating(true);
  };

  return (
    <div className="toy-example-container">
      <div className="controls">
        <button 
          className="animate-btn" 
          onClick={startAnimation} 
          disabled={isAnimating}
        >
          {isAnimating ? 'Decoding...' : '▶ Generate Next Token'}
        </button>
      </div>

      <div className="decoder-diagram">
        {steps.map((s, idx) => (
          <div 
            key={s.id} 
            className={`diagram-node ${step === idx ? 'active-node' : ''} ${step > idx ? 'completed-node' : ''}`}
          >
            <div className="node-content">
              <span className="node-icon">
                {idx === 1 ? '🙈' : idx === 3 ? '🧠' : idx === 0 ? '📥' : idx === 7 ? '📤' : idx % 2 === 0 ? '➕' : '⚙️'}
              </span>
              <span className="node-label">{s.label}</span>
            </div>
            {idx < steps.length - 1 && <div className="node-connector">⬇</div>}
            
            {/* Visualizing Cross Attention Input */}
            {idx === 3 && (
              <div className={`side-input ${step === idx ? 'fade-in' : ''}`}>
                ⬅ Encoder Memory
              </div>
            )}
          </div>
        ))}
      </div>

      <div className="explanation-box">
        <h4>Current Step: {steps[step].label}</h4>
        <p>{steps[step].desc}</p>
      </div>
    </div>
  );
};

export default ToyExampleDecoder;
