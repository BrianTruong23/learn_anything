import React, { useState } from 'react';
import './ConceptSection.css';
import './ArchitectureVisual.css';
import UnderstandingButton from './UnderstandingButton';
import MathEquation from './MathEquation';
import LatexText from './LatexText';
import SimpleSyntaxHighlighter from './SimpleSyntaxHighlighter';
import ToyExampleBigPicture from './toyExamples/ToyExampleBigPicture';
import ToyExampleTokenization from './toyExamples/ToyExampleTokenization';
import ToyExamplePositionalEmbeddings from './toyExamples/ToyExamplePositionalEmbeddings';
import ToyExampleSelfAttention from './toyExamples/ToyExampleSelfAttention';
import ToyExampleMultiHead from './toyExamples/ToyExampleMultiHead';
import ToyExampleFeedforward from './toyExamples/ToyExampleFeedforward';
import ToyExampleLayerNorm from './toyExamples/ToyExampleLayerNorm';
import ToyExampleEncoder from './toyExamples/ToyExampleEncoder';
import ToyExampleCrossAttention from './toyExamples/ToyExampleCrossAttention';
import ToyExampleDecoder from './toyExamples/ToyExampleDecoder';

// Map concept IDs to their interactive toy example components
const toyExampleComponents = {
  'concept_0': ToyExampleBigPicture,
  'concept_1': ToyExampleTokenization, // Tokenization & Embeddings
  'concept_2': ToyExamplePositionalEmbeddings,
  'concept_3': ToyExampleSelfAttention,
  'concept_4': ToyExampleMultiHead,
  'concept_5': ToyExampleFeedforward, // Feed-Forward Networks
  'concept_6': ToyExampleLayerNorm,   // Add & Norm
  'concept_7': ToyExampleEncoder,     // Encoder Block
  'concept_8': ToyExampleCrossAttention, // Cross-Attention
  'concept_9': ToyExampleDecoder      // Decoder Block
};

function ConceptSection({ concept, difficulty, isUnderstood, onToggleUnderstanding }) {
  // Get the explanation for the current difficulty level
  const explanation = concept.explanations[difficulty];
  
  // Get the appropriate toy example component for this concept
  const ToyExampleComponent = toyExampleComponents[concept.id];
  
  // State for architecture visualization
  const [isVisualVisible, setIsVisualVisible] = useState(false);
  
  // State for code expansion
  const [isCodeExpanded, setIsCodeExpanded] = useState(false);

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
              {concept.architectureImage === 'PLACEHOLDER' ? (
                <div className="architecture-placeholder">
                  <span className="placeholder-icon">🖼️</span>
                  <p>Image Coming Soon</p>
                </div>
              ) : (
                <img 
                  src={concept.architectureImage} 
                  alt={`${concept.title} Architecture`} 
                  className="architecture-image"
                />
              )}
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
      
      {/* Code Implementation Card */}
      {concept.codeSnippet && (
        <div className="concept-card code-card">
          <div 
            className="card-heading clickable-heading"
            onClick={() => setIsCodeExpanded(!isCodeExpanded)}
            style={{ cursor: 'pointer', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}
          >
            <span>
              <span className="card-icon">💻</span>
              Code Implementation (PyTorch)
            </span>
            <span className="toggle-icon">{isCodeExpanded ? '▼' : '▶'}</span>
          </div>
          
          {isCodeExpanded && (
            <div className="code-content fade-in">
              <SimpleSyntaxHighlighter code={concept.codeSnippet} language="python" />
            </div>
          )}
        </div>
      )}

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
