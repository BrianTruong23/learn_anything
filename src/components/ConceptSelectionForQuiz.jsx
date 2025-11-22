import React from 'react';
import './ConceptSelectionForQuiz.css';

function ConceptSelectionForQuiz({ concepts, selectedConcepts, onSelectionChange, questionCounts }) {
  const handleToggle = (conceptId) => {
    const newSelection = selectedConcepts.includes(conceptId)
      ? selectedConcepts.filter(id => id !== conceptId)
      : [...selectedConcepts, conceptId];
    
    // Ensure at least one concept is selected
    if (newSelection.length > 0) {
      onSelectionChange(newSelection);
    }
  };

  const handleSelectAll = () => {
    onSelectionChange(concepts.map(c => c.id));
  };

  const handleDeselectAll = () => {
    // Keep at least one selected
    onSelectionChange([concepts[0].id]);
  };

  const getTotalQuestions = () => {
    return selectedConcepts.reduce((sum, conceptId) => {
      return sum + (questionCounts[conceptId] || 0);
    }, 0);
  };

  return (
    <div className="concept-selection-panel">
      <div className="selection-header">
        <h3>🎯 Select Concepts for Quiz</h3>
        <p className="selection-description">
          Choose which concepts you want to be tested on. Questions will only come from the selected concepts.
        </p>
      </div>

      <div className="selection-controls">
        <button className="select-all-btn" onClick={handleSelectAll}>
          ✅ Select All
        </button>
        <button className="deselect-all-btn" onClick={handleDeselectAll}>
          ❌ Deselect All
        </button>
      </div>

      <div className="concepts-grid">
        {concepts.map((concept) => {
          const isSelected = selectedConcepts.includes(concept.id);
          const questionCount = questionCounts[concept.id] || 0;
          
          return (
            <div
              key={concept.id}
              className={`concept-checkbox-card ${isSelected ? 'selected' : ''}`}
              onClick={() => handleToggle(concept.id)}
            >
              <div className="checkbox-wrapper">
                <input
                  type="checkbox"
                  checked={isSelected}
                  onChange={() => handleToggle(concept.id)}
                  onClick={(e) => e.stopPropagation()}
                />
              </div>
              <div className="concept-info">
                <div className="concept-title-small">{concept.title}</div>
                <div className="question-badge">
                  {questionCount} question{questionCount !== 1 ? 's' : ''}
                </div>
              </div>
            </div>
          );
        })}
      </div>

      <div className="selection-summary">
        <strong>{getTotalQuestions()} total questions</strong> from {selectedConcepts.length} selected concept{selectedConcepts.length !== 1 ? 's' : ''}
      </div>
    </div>
  );
}

export default ConceptSelectionForQuiz;
