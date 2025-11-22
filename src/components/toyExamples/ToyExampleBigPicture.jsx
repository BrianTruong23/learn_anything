import React, { useState } from 'react';
import './ToyExampleBigPicture.css';

const stages = [
  {
    id: 'input',
    name: 'Input Sentence',
    description: 'Raw text input: "The cat sat on the mat"',
    icon: '📝'
  },
  {
    id: 'embeddings',
    name: 'Embeddings',
    description: 'Words converted to dense vector representations',
    icon: '🔢'
  },
  {
    id: 'transformer',
    name: 'Transformer',
    description: 'Self-attention processes all tokens in parallel',
    icon: '⚡'
  },
  {
    id: 'output',
    name: 'Output',
    description: 'Refined representations ready for prediction',
    icon: '✨'
  }
];

function ToyExampleBigPicture() {
  const [activeStage, setActiveStage] = useState(null);

  return (
    <div className="big-picture-container">
      <div className="pipeline">
        {stages.map((stage, index) => (
          <React.Fragment key={stage.id}>
            <div
              className={`pipeline-stage ${activeStage === stage.id ? 'active' : ''}`}
              onClick={() => setActiveStage(stage.id)}
              onMouseEnter={() => setActiveStage(stage.id)}
              onMouseLeave={() => setActiveStage(null)}
            >
              <div className="stage-icon">{stage.icon}</div>
              <div className="stage-name">{stage.name}</div>
            </div>
            {index < stages.length - 1 && (
              <div className="pipeline-arrow">→</div>
            )}
          </React.Fragment>
        ))}
      </div>
      
      {activeStage && (
        <div className="stage-description fade-in">
          <strong>{stages.find(s => s.id === activeStage)?.name}:</strong>{' '}
          {stages.find(s => s.id === activeStage)?.description}
        </div>
      )}
      
      {!activeStage && (
        <div className="instruction-text">
          👆 Hover or click on each stage to see how data flows through the Transformer
        </div>
      )}
    </div>
  );
}

export default ToyExampleBigPicture;
