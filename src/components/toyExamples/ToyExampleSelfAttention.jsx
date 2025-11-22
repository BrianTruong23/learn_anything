import React, { useState } from 'react';
import './ToyExampleSelfAttention.css';

const sentence = "The animal didn't cross the street because it was too tired".split(' ');
const attentionMap = {
  'it': ['animal', 'tired'],
  'animal': ['The', 'it'],
  'cross': ['street', "didn't"],
  'street': ['cross', 'the'],
  'tired': ['it', 'was']
};

function ToyExampleSelfAttention() {
  const [selectedWord, setSelectedWord] = useState(null);

  const isRelated = (word) => {
    if (!selectedWord) return false;
    const related = attentionMap[selectedWord] || [];
    return related.includes(word);
  };

  return (
    <div className="attention-container">
      <div className="sentence-display">
        {sentence.map((word, index) => (
          <span
            key={`${word}-${index}`}
            className={`word-token ${selectedWord === word ? 'selected' : ''} ${
              isRelated(word) ? 'related' : ''
            }`}
            onClick={() => setSelectedWord(word === selectedWord ? null : word)}
          >
            {word}
            {isRelated(word) && selectedWord && (
              <div className="attention-line"></div>
            )}
          </span>
        ))}
      </div>

      <div className="attention-info">
        {selectedWord ? (
          <div className="info-box fade-in">
            <div className="info-header">
              🎯 Attention from "{selectedWord}"
            </div>
            <div className="info-body">
              {attentionMap[selectedWord] ? (
                <>
                  This word attends to: <strong>{attentionMap[selectedWord].join(', ')}</strong>
                  <br />
                  <span className="hint">Highlighted words show attention connections!</span>
                </>
              ) : (
                'Click another word to see its attention pattern'
              )}
            </div>
          </div>
        ) : (
          <div className="instruction-box">
            👆 Click any word to see which other words it "pays attention" to
          </div>
        )}
      </div>
    </div>
  );
}

export default ToyExampleSelfAttention;
