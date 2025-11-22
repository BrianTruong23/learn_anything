import React, { useState } from 'react';
import './ConceptSequence.css';
import { concepts } from './data/conceptsData.js';
import DifficultySelector from './components/DifficultySelector';
import ConceptSection from './components/ConceptSection';
import CompletionIndicator from './components/CompletionIndicator';

function ConceptSequence({ difficulty, onDifficultyChange, setScore }) {
  const [currentConceptIndex, setCurrentConceptIndex] = useState(0);
  const [understoodConcepts, setUnderstoodConcepts] = useState(new Set());

  const currentConcept = concepts[currentConceptIndex];
  const isFirstConcept = currentConceptIndex === 0;
  const isLastConcept = currentConceptIndex === concepts.length - 1;

  const handlePrevious = () => {
    if (!isFirstConcept) {
      setCurrentConceptIndex(prev => prev - 1);
      window.scrollTo({ top: 0, behavior: 'smooth' });
    }
  };

  const handleNext = () => {
    if (!isLastConcept) {
      setCurrentConceptIndex(prev => prev + 1);
      window.scrollTo({ top: 0, behavior: 'smooth' });
    }
  };

  const handleToggleUnderstanding = (conceptId) => {
    setUnderstoodConcepts(prev => {
      const newSet = new Set(prev);
      if (newSet.has(conceptId)) {
        newSet.delete(conceptId);
        // Optional: Decrement score if unchecking? User only asked for increase.
        // Let's stick to increase only on first check to avoid gaming, 
        // but simplistic approach: if unchecking, maybe remove points?
        // User request: "for each concepts marked understood, the score gets increased by 10"
        // I'll implement simple toggle logic: add 10 when marked, remove 10 when unmarked.
        setScore(s => Math.max(0, s - 10));
      } else {
        newSet.add(conceptId);
        setScore(s => s + 10);
      }
      return newSet;
    });
  };



  const allConceptsUnderstood = concepts.length > 0 && understoodConcepts.size === concepts.length;

  return (
    <div className={`concept-sequence ${allConceptsUnderstood ? 'section-complete' : ''}`} style={{ position: 'relative' }}>
      {/* Completion Indicator */}
      <CompletionIndicator 
        total={concepts.length}
        understood={understoodConcepts.size}
      />
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
          isUnderstood={understoodConcepts.has(currentConcept.id)}
          onToggleUnderstanding={handleToggleUnderstanding}
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
              className={`dot ${index === currentConceptIndex ? 'active' : ''} ${understoodConcepts.has(concepts[index].id) ? 'understood' : ''}`}
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

