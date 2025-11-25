import React, { useState, useEffect } from 'react';
import './ToyExample.css';


const ToyExampleEncoder = () => {
  const [isAnimating, setIsAnimating] = useState(false);
  const [step, setStep] = useState(0);

  const steps = [
    { id: 0, label: "Input", desc: "Input Vector X" },
    { id: 1, label: "Self-Attention", desc: "Mixing context from other tokens" },
    { id: 2, label: "Add & Norm 1", desc: "Adding residual connection + Normalization" },
    { id: 3, label: "Feed-Forward", desc: "Processing individual token features" },
    { id: 4, label: "Add & Norm 2", desc: "Adding residual connection + Normalization" },
    { id: 5, label: "Output", desc: "Contextualized Vector" }
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
      }, 1000);
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
          {isAnimating ? 'Processing...' : '▶ Run Encoder Block'}
        </button>
      </div>

      <div className="encoder-diagram">
        {steps.map((s, idx) => (
          <div 
            key={s.id} 
            className={`diagram-node ${step === idx ? 'active-node' : ''} ${step > idx ? 'completed-node' : ''}`}
          >
            <div className="node-content">
              <span className="node-icon">
                {idx === 0 ? '📥' : idx === 5 ? '📤' : idx % 2 !== 0 ? '⚙️' : '➕'}
              </span>
              <span className="node-label">{s.label}</span>
            </div>
            {idx < steps.length - 1 && <div className="node-connector">⬇</div>}
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

export default ToyExampleEncoder;
