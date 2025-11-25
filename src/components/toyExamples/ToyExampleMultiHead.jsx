import React, { useState } from 'react';
import './ToyExampleMultiHead.css';

const heads = [
  {
    id: 1,
    name: 'Head 1: Syntax',
    color: '#3b82f6',
    description: 'Focuses on grammatical relationships (adjective→noun, subject→verb)',
    patterns: [
      { from: 'quick', to: 'fox', label: 'adjective' },
      { from: 'brown', to: 'fox', label: 'adjective' }
    ]
  },
  {
    id: 2,
    name: 'Head 2: Action',
    color: '#10b981',
    description: 'Captures subject-verb relationships',
    patterns: [
      { from: 'fox', to: 'jumps', label: 'subject' },
      { from: 'jumps', to: 'over', label: 'verb-prep' }
    ]
  },
  {
    id: 3,
    name: 'Head 3: Objects',
    color: '#f59e0b',
    description: 'Identifies object and location relationships',
    patterns: [
      { from: 'over', to: 'dog', label: 'preposition' },
      { from: 'the', to: 'dog', label: 'determiner' }
    ]
  }
];

const sentence = ['The', 'quick', 'brown', 'fox', 'jumps', 'over', 'the', 'lazy', 'dog'];

function ToyExampleMultiHead() {
  const [activeHead, setActiveHead] = useState(1);

  const currentHead = heads.find(h => h.id === activeHead);

  const isHighlighted = (word) => {
    return currentHead.patterns.some(p => p.from === word || p.to === word);
  };

  return (
    <div className="multihead-container">
      <div className="head-selector">
        {heads.map(head => (
          <button
            key={head.id}
            className={`head-btn ${activeHead === head.id ? 'active' : ''}`}
            style={{
              '--head-color': head.color,
              backgroundColor: activeHead === head.id ? head.color : 'var(--card-bg)',
              color: activeHead === head.id ? 'white' : head.color,
              borderColor: head.color
            }}
            onClick={() => setActiveHead(head.id)}
          >
            {head.name}
          </button>
        ))}
      </div>

      <div className="sentence-visualization">
        {sentence.map((word, index) => (
          <span
            key={`${word}-${index}`}
            className={`word-item ${isHighlighted(word) ? 'highlighted' : ''}`}
            style={{
              '--highlight-color': isHighlighted(word) ? currentHead.color : 'transparent'
            }}
          >
            {word}
          </span>
        ))}
      </div>

      <div className="head-description slide-in">
        <div className="desc-title" style={{ color: currentHead.color }}>
          {currentHead.name}
        </div>
        <div className="desc-text">{currentHead.description}</div>
        <div className="pattern-list">
          {currentHead.patterns.map((pattern, index) => (
            <div key={index} className="pattern-item">
              <span className="pattern-from">{pattern.from}</span>
              <span className="pattern-arrow">→</span>
              <span className="pattern-to">{pattern.to}</span>
              <span className="pattern-label">({pattern.label})</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

export default ToyExampleMultiHead;
