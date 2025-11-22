import React from 'react';
import './CompletionIndicator.css';

function CompletionIndicator({ total, understood }) {
  const isComplete = understood === total;
  const percentage = Math.round((understood / total) * 100);

  return (
    <div className={`completion-indicator ${isComplete ? 'complete' : ''}`}>
      {isComplete ? (
        <div className="completion-content">
          <span className="completion-icon">🎉</span>
          <span className="completion-text">Section Complete!</span>
          <span className="completion-check">✅</span>
        </div>
      ) : (
        <div className="completion-content">
          <span className="completion-text">
            Progress: {understood} / {total} concepts understood
          </span>
          <div className="progress-bar">
            <div 
              className="progress-fill" 
              style={{ width: `${percentage}%` }}
            />
          </div>
        </div>
      )}
    </div>
  );
}

export default CompletionIndicator;
