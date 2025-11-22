import React from 'react';
import './DifficultySelector.css';

function DifficultySelector({ currentDifficulty, onDifficultyChange }) {
  const levels = ['beginner', 'intermediate', 'advanced'];
  const labels = {
    beginner: '🌱 Beginner',
    intermediate: '📚 Intermediate',
    advanced: '🎓 Advanced'
  };

  return (
    <div className="difficulty-selector">
      <div className="difficulty-label">Difficulty Level:</div>
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
