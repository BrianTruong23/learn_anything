import React, { useState } from 'react';
import './ConceptSection.css';
import './ArchitectureVisual.css';
import UnderstandingButton from './UnderstandingButton';
import MathEquation from './MathEquation';
import LatexText from './LatexText';
import ToyExampleBigPicture from './toyExamples/ToyExampleBigPicture';
import ToyExamplePositionalEmbeddings from './toyExamples/ToyExamplePositionalEmbeddings';
import ToyExampleSelfAttention from './toyExamples/ToyExampleSelfAttention';
import ToyExampleMultiHead from './toyExamples/ToyExampleMultiHead';
import ToyExampleFeedforward from './toyExamples/ToyExampleFeedforward';
import ToyExampleLayerNorm from './toyExamples/ToyExampleLayerNorm';

// Map concept IDs to their interactive toy example components
const toyExampleComponents = {
  'concept_0': ToyExampleBigPicture,
  'concept_1': ToyExamplePositionalEmbeddings,
  'concept_2': ToyExampleSelfAttention,
  'concept_3': ToyExampleMultiHead,
  'concept_4': ToyExampleFeedforward,
  'concept_5': ToyExampleLayerNorm
};

function ConceptSection({ concept, difficulty, isUnderstood, onToggleUnderstanding }) {
  // Get the explanation for the current difficulty level
  const explanation = concept.explanations[difficulty];
  
  // Get the appropriate toy example component for this concept
  const ToyExampleComponent = toyExampleComponents[concept.id];
  
  // State for architecture visualization
  const [isVisualVisible, setIsVisualVisible] = useState(false);

  return (
    <div className="concept-section-container">
      {/* Architecture Visualization (Collapsible) */}
      {concept.architectureImage && (
        <div className="architecture-visual-section">
          <button 
            className={`visual-toggle-btn ${isVisualVisible ? 'active' : ''}`}
            onClick={() => setIsVisualVisible(!isVisualVisible)}
          >
            {isVisualVisible ? '👁️ Hide Architecture Diagram' : '🖼️ Show Architecture Diagram'}
          </button>
          
          {isVisualVisible && (
            <div className="architecture-visual-content fade-in">
              <img 
                src={concept.architectureImage} 
                alt={`${concept.title} Architecture`} 
                className="architecture-image"
              />
              <p className="visual-caption">
                High-level view of the Transformer architecture. Don't worry about the details yet—we'll break it down step by step!
              </p>
            </div>
          )}
        </div>
      )}

      {/* Motivation Card */}
      <div className="concept-card motivation-card">
        <h3 className="card-heading">
          <span className="card-icon">💡</span>
          Motivation
        </h3>
        <p className="card-content">
          <LatexText text={explanation.motivation} />
        </p>
      </div>

      {/* Definition Card */}
      <div className="concept-card definition-card">
        <h3 className="card-heading">
          <span className="card-icon">📚</span>
          Definition
        </h3>
        <p className="card-content">
          <LatexText text={explanation.definition} />
        </p>
        
        {/* Math Equations */}
        {explanation.equations && explanation.equations.length > 0 && (
          <div className="equations-section">
            {explanation.equations.map((eq, index) => (
              <MathEquation key={index} equation={eq} />
            ))}
          </div>
        )}
      </div>

      {/* Toy Example Card - Interactive Component */}
      <div className="concept-card toyexample-card">
        <h3 className="card-heading">
          <span className="card-icon">🔍</span>
          Interactive Example
        </h3>
        
        {ToyExampleComponent ? (
          <ToyExampleComponent />
        ) : (
          <div className="no-example-message">
            Interactive example coming soon for this concept!
          </div>
        )}
      </div>

      {/* Understanding Button */}
      <div className="understanding-section">
        <UnderstandingButton 
          conceptId={concept.id}
          isUnderstood={isUnderstood}
          onToggle={onToggleUnderstanding}
        />
      </div>
    </div>
  );
}

export default ConceptSection;
