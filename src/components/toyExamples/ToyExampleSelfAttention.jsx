import React, { useState, useMemo } from 'react';
import './ToyExampleSelfAttention.css';

const sentence = "The animal didn't cross the street because it was too tired".split(' ');

// Mock 4D embeddings for demonstration
// Designed to have some interesting relationships
const embeddings = {
  "The": [0.1, 0.8, 0.1, 0.1],
  "animal": [0.8, 0.1, 0.1, 0.2],
  "didn't": [0.1, 0.1, 0.8, 0.1],
  "cross": [0.2, 0.2, 0.1, 0.8],
  "the": [0.1, 0.8, 0.1, 0.1],
  "street": [0.7, 0.1, 0.2, 0.1], // Related to animal (noun-like)
  "because": [0.1, 0.1, 0.1, 0.5],
  "it": [0.8, 0.1, 0.1, 0.2],    // Very similar to animal
  "was": [0.1, 0.1, 0.1, 0.1],
  "too": [0.1, 0.2, 0.1, 0.1],
  "tired": [0.2, 0.1, 0.1, 0.8]  // Related to cross (verb/adj)
};

const d_k = 4; // Dimension of keys

function ToyExampleSelfAttention() {
  const [selectedWord, setSelectedWord] = useState(null);

  // Calculate attention scores when a word is selected
  const attentionData = useMemo(() => {
    if (!selectedWord) return null;

    const queryVector = embeddings[selectedWord];
    
    // 1. Calculate Dot Products (Scores)
    const scores = sentence.map(word => {
      const keyVector = embeddings[word];
      // Dot product: sum(q[i] * k[i])
      const dotProduct = queryVector.reduce((sum, val, i) => sum + val * keyVector[i], 0);
      return { word, keyVector, dotProduct };
    });

    // 2. Scale Scores
    const scaledScores = scores.map(item => ({
      ...item,
      scaledScore: item.dotProduct / Math.sqrt(d_k)
    }));

    // 3. Apply Softmax
    // First find max for numerical stability (optional here but good practice)
    // Then compute exponentials
    const exponentials = scaledScores.map(item => Math.exp(item.scaledScore));
    const sumExponentials = exponentials.reduce((a, b) => a + b, 0);

    const finalAttention = scaledScores.map((item, index) => ({
      ...item,
      attentionWeight: exponentials[index] / sumExponentials
    }));

    return finalAttention;
  }, [selectedWord]);

  return (
    <div className="attention-container">
      <div className="sentence-display">
        {sentence.map((word, index) => {
          // Find attention weight for this word if something is selected
          const weight = attentionData ? attentionData[index].attentionWeight : 0;
          
          return (
            <span
              key={`${word}-${index}`}
              className={`word-token ${selectedWord === word ? 'selected' : ''}`}
              style={{
                // Dynamic styling based on attention weight
                backgroundColor: selectedWord && selectedWord !== word 
                  ? `rgba(100, 108, 255, ${weight * 2})` // Amplify for visibility
                  : undefined,
                transform: selectedWord && selectedWord !== word 
                  ? `scale(${1 + weight * 0.5})` 
                  : 'scale(1)',
                opacity: selectedWord && selectedWord !== word && weight < 0.05 ? 0.5 : 1
              }}
              onClick={() => setSelectedWord(word === selectedWord ? null : word)}
            >
              {word}
              {/* Tooltip or label for weight could go here */}
              {selectedWord && (
                <span className="attention-value-label">
                  {(weight * 100).toFixed(0)}%
                </span>
              )}
            </span>
          );
        })}
      </div>

      <div className="attention-info">
        {selectedWord && attentionData ? (
          <div className="calculation-breakdown fade-in">
            <h3>Attention Calculation for "{selectedWord}"</h3>
            
            <div className="vector-display">
              <span className="label">Query Vector (<strong>Q</strong>):</span>
              <span className="vector-values">[{embeddings[selectedWord].join(', ')}]</span>
            </div>

            <div className="steps-table-container">
              <table className="steps-table">
                <thead>
                  <tr>
                    <th>Word (Key)</th>
                    <th>Key Vector (<strong>K</strong>)</th>
                    <th>Score<br/><small>(Q · K)</small></th>
                    <th>Scaled<br/><small>(Score / √{d_k})</small></th>
                    <th>Softmax<br/><small>(Attention %)</small></th>
                  </tr>
                </thead>
                <tbody>
                  {attentionData.map((data, idx) => (
                    <tr key={idx} className={data.word === selectedWord ? 'current-word-row' : ''}>
                      <td className="word-cell">{data.word}</td>
                      <td className="vector-cell">[{data.keyVector.join(', ')}]</td>
                      <td>{data.dotProduct.toFixed(2)}</td>
                      <td>{data.scaledScore.toFixed(2)}</td>
                      <td className="weight-cell">
                        <div className="weight-bar-container">
                          <div 
                            className="weight-bar" 
                            style={{ width: `${data.attentionWeight * 100}%` }}
                          ></div>
                          <span>{(data.attentionWeight * 100).toFixed(1)}%</span>
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            
            <div className="formula-explanation">
              <p>
                <strong>Formula:</strong> Attention(Q, K, V) = softmax( (Q · K) / √d_k )
              </p>
              <p>
                We compute the dot product of the <strong>Query</strong> (selected word) with every <strong>Key</strong> (all words), 
                scale it by √{d_k} = {Math.sqrt(d_k)}, and apply softmax to get probabilities.
              </p>
            </div>
          </div>
        ) : (
          <div className="instruction-box">
            👆 Click any word to see the <strong>Scaled Dot-Product Attention</strong> calculation in action!
          </div>
        )}
      </div>
    </div>
  );
}

export default ToyExampleSelfAttention;
