import React, { useState } from 'react';
import './ToyExamplePositionalEmbeddings.css';

const initialWords = ['The', 'cat', 'chased', 'the', 'dog'];

function ToyExamplePositionalEmbeddings() {
  const [words, setWords] = useState(initialWords);
  const [showPositions, setShowPositions] = useState(false);

  const shuffleWords = () => {
    const shuffled = [...words].sort(() => Math.random() - 0.5);
    setWords(shuffled);
  };

  const resetWords = () => {
    setWords(initialWords);
  };

  return (
    <div className="positional-container">
      <div className="word-row">
        {words.map((word, index) => (
          <div key={`${word}-${index}`} className="word-box">
            {showPositions && (
              <div className="position-indicator slide-down">
                pos={index}
              </div>
            )}
            <div className="word-text">{word}</div>
          </div>
        ))}
      </div>

      <div className="controls-row">
        <button
          className="control-button primary"
          onClick={() => setShowPositions(!showPositions)}
        >
          {showPositions ? '🔽 Hide Positions' : '🔼 Show Positions'}
        </button>
        <button
          className="control-button secondary"
          onClick={shuffleWords}
        >
          🔀 Shuffle Order
        </button>
        <button
          className="control-button secondary"
          onClick={resetWords}
        >
          🔄 Reset
        </button>
      </div>

      <div className="explanation-text">
        {showPositions ? (
          <span>
            ✅ Position embeddings are now added! Each word knows its location in the sequence.
          </span>
        ) : (
          <span>
            Click "Show Positions" to see how each word gets a position index 📍
          </span>
        )}
      </div>
    </div>
  );
}

export default ToyExamplePositionalEmbeddings;
