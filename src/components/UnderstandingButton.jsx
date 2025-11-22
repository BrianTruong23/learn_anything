import React from 'react';
import './UnderstandingButton.css';

function UnderstandingButton({ conceptId, isUnderstood, onToggle }) {
  return (
    <button
      className={`understanding-btn ${isUnderstood ? 'understood' : ''}`}
      onClick={() => onToggle(conceptId)}
      aria-label={isUnderstood ? 'Mark as not understood' : 'Mark as understood'}
    >
      {isUnderstood ? (
        <>
          <span className="checkmark">✅</span> Understood
        </>
      ) : (
        <>
          <span className="circle">○</span> Mark as Understood
        </>
      )}
    </button>
  );
}

export default UnderstandingButton;
