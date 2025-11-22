import React, { useState } from 'react';
import './ConceptSequence.css';
import { concepts } from './data/conceptsData.js';
import DifficultySelector from './components/DifficultySelector';
import ConceptSection from './components/ConceptSection';

function ConceptSequence({ difficulty, onDifficultyChange }) {
  const [currentConceptIndex, setCurrentConceptIndex] = useState(0);

  const currentConcept = concepts[currentConceptIndex];
  const isFirstConcept = currentConceptIndex === 0;
  const isLastConcept = currentConceptIndex === concepts.length - 1;

  const handlePrevious = () => {
    if (!isFirstConcept) {
      setCurrentConceptIndex(prev => prev - 1);
    }
  };

  const handleNext = () => {
    if (!isLastConcept) {
      setCurrentConceptIndex(prev => prev + 1);
    }
  };

  return (
    <div className="concept-sequence">
      <div className="progress-indicator">
        Step {currentConceptIndex + 1} of {concepts.length}: {currentConcept.title}
      </div>

      <DifficultySelector 
        currentDifficulty={difficulty}
        onDifficultyChange={onDifficultyChange}
      />

      <div className="concept-content">
        <h2 className="concept-title">{currentConcept.title}</h2>

        <ConceptSection 
          concept={currentConcept}
          difficulty={difficulty}
        />
      </div>

      <div className="nav-buttons">
        <button 
          onClick={handlePrevious} 
          disabled={isFirstConcept}
          className="nav-btn prev-btn"
        >
          ← Previous
        </button>
        <div className="concept-dots">
          {concepts.map((_, index) => (
            <span 
              key={index} 
              className={`dot ${index === currentConceptIndex ? 'active' : ''}`}
              onClick={() => setCurrentConceptIndex(index)}
              title={`Go to ${concepts[index].title}`}
            />
          ))}
        </div>
        <button 
          onClick={handleNext} 
          disabled={isLastConcept}
          className="nav-btn next-btn"
        >
          Next →
        </button>
      </div>
    </div>
  );
}

export default ConceptSequence;

