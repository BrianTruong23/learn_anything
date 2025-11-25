import React, { useState } from 'react';
import './ToyExample.css';


const ToyExampleTokenization = () => {
  const [inputText, setInputText] = useState("The cat sat");
  
  // Simple mock tokenizer and embedding lookup
  const mockVocab = {
    "the": { id: 101, vector: [0.1, 0.9, -0.2, 0.5] },
    "cat": { id: 204, vector: [0.8, 0.1, 0.3, -0.1] },
    "sat": { id: 305, vector: [-0.5, 0.2, 0.8, 0.1] },
    "dog": { id: 205, vector: [0.7, 0.2, 0.4, -0.2] },
    "run": { id: 401, vector: [-0.3, 0.6, 0.1, 0.9] },
    "[unk]": { id: 999, vector: [0.0, 0.0, 0.0, 0.0] }
  };

  const tokens = inputText.trim().toLowerCase().split(/\s+/).filter(t => t);

  const getColor = (val) => {
    // Map -1..1 to a color scale (blue to red)
    const intensity = Math.min(Math.abs(val), 1);
    return val > 0 
      ? `rgba(255, 99, 71, ${intensity})` // Red for positive
      : `rgba(70, 130, 180, ${intensity})`; // Blue for negative
  };

  return (
    <div className="toy-example-container">
      <div className="input-section">
        <label>Type a sentence:</label>
        <input 
          type="text" 
          value={inputText}
          onChange={(e) => setInputText(e.target.value)}
          placeholder="e.g., The cat sat"
          className="text-input"
        />
      </div>

      <div className="visualization-flow">
        {/* Step 1: Raw Text */}
        <div className="flow-step">
          <h4>1. Raw Text</h4>
          <div className="data-box text-box">"{inputText}"</div>
        </div>

        <div className="arrow">⬇️ Tokenize</div>

        {/* Step 2: Tokens */}
        <div className="flow-step">
          <h4>2. Tokens</h4>
          <div className="token-list">
            {tokens.map((token, idx) => (
              <span key={idx} className="token-chip">{token}</span>
            ))}
          </div>
        </div>

        <div className="arrow">⬇️ Lookup IDs</div>

        {/* Step 3: Token IDs */}
        <div className="flow-step">
          <h4>3. Token IDs</h4>
          <div className="id-list">
            {tokens.map((token, idx) => {
              const id = mockVocab[token]?.id || mockVocab["[unk]"].id;
              return <span key={idx} className="id-chip">{id}</span>;
            })}
          </div>
        </div>

        <div className="arrow">⬇️ Embedding Lookup</div>

        {/* Step 4: Vectors */}
        <div className="flow-step">
          <h4>4. Vectors (Embeddings)</h4>
          <div className="vector-list">
            {tokens.map((token, idx) => {
              const vector = mockVocab[token]?.vector || mockVocab["[unk]"].vector;
              return (
                <div key={idx} className="vector-item">
                  <span className="token-label">{token}</span>
                  <div className="vector-visual">
                    {vector.map((val, vIdx) => (
                      <div 
                        key={vIdx} 
                        className="vector-cell"
                        style={{ backgroundColor: getColor(val) }}
                        title={`Value: ${val}`}
                      ></div>
                    ))}
                  </div>
                  <span className="vector-values">[{vector.join(', ')}]</span>
                </div>
              );
            })}
          </div>
        </div>
      </div>

      <div className="explanation-box">
        <p>
          <strong>What's happening?</strong> The raw text is split into pieces called <em>tokens</em>. 
          Each token is assigned a unique ID number from the vocabulary. 
          Finally, that ID is used to look up a dense vector (list of numbers) that represents the word's meaning.
        </p>
      </div>
    </div>
  );
};

export default ToyExampleTokenization;
