import React from 'react';
import './DifficultySelector.css';

function DifficultySelector({ currentDifficulty, onDifficultyChange }) {
  const levels = ['beginner', 'advanced'];
  const labels = {
    beginner: '🌱 Beginner',
    advanced: '🎓 Advanced'
  };

  return (
    <div className="difficulty-selector">
      <div className="difficulty-label">Explanation Level:</div>
      <div className="difficulty-buttons">
        {levels.map(level => (
          <button
            key={level}
            className={`difficulty-btn ${currentDifficulty === level ? 'active' : ''}`}
            onClick={() => onDifficultyChange(level)}
          >
            {labels[level]}
          </button>
        ))}
      </div>
    </div>
  );
}

export default DifficultySelector;
