import React from 'react';
import './SubConceptProgress.css';

function SubConceptProgress({ subConcepts, completedSubConcepts, onToggleSubConcept }) {
  return (
    <div className="sub-concept-progress">
      <div className="sub-concept-list">
        {subConcepts.map((subConcept) => {
          const isCompleted = completedSubConcepts.has(subConcept.id);
          
          return (
            <div 
              key={subConcept.id} 
              className={`sub-concept-item ${isCompleted ? 'completed' : ''}`}
            >
              <span className="sub-concept-label">{subConcept.label}</span>
              <button
                className={`sub-concept-btn ${isCompleted ? 'completed' : ''}`}
                onClick={() => onToggleSubConcept(subConcept.id)}
                aria-label={isCompleted ? `Mark ${subConcept.label} as incomplete` : `Mark ${subConcept.label} as complete`}
              >
                {isCompleted ? '✓' : '○'}
              </button>
            </div>
          );
        })}
      </div>
    </div>
  );
}

export default SubConceptProgress;
