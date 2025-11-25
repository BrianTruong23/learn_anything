import React from 'react';
import './ConceptExplorer.css';

function ConceptExplorer({ onSelectConcept }) {
  return (
    <div className="concept-explorer">
      <header className="explorer-header">
        <h1 className="explorer-title"> AI Concept Explorer</h1>
        <p className="explorer-subtitle">Build Understanding Concept by Concept</p>
      </header>

      <div className="concepts-grid">
        {/* Primary Card: Transformer */}
        <div className="concept-card-item primary-card">
          <div className="card-badge intermediate">Intermediate</div>
          <h2 className="card-title">Transformer Architecture</h2>
          <p className="card-description">
            Understand attention, positional embeddings, and how modern LLMs work.
          </p>
          <button 
            className="card-cta-btn"
            onClick={() => onSelectConcept('transformer')}
          >
            Start Transformer Journey
          </button>
  
        </div>



        {/* BERT Card */}
        <div className="concept-card-item placeholder-card">
          <div className="card-badge coming-soon">Coming Soon</div>
          <h2 className="card-title">BERT Architecture</h2>
          <p className="card-description">
            Explore bidirectional training, masked language modeling, and transfer learning.
          </p>
          <button className="card-cta-btn disabled" disabled>
            Explore
          </button>
        </div>

        
        {/* CNN Card */}
        <div className="concept-card-item placeholder-card">
          <div className="card-badge coming-soon">Coming Soon</div>
          <h2 className="card-title">CNN Architecture</h2>
          <p className="card-description">
            Explore the fundamental concepts of CNN.
          </p>
          <button className="card-cta-btn disabled" disabled>
            Explore
          </button>
        </div>

        {/* Placeholder Card: RNN */}
        <div className="concept-card-item placeholder-card">
          <div className="card-badge coming-soon">Coming Soon</div>
          <h2 className="card-title">Recurrent Neural Networks (RNN)</h2>
          <p className="card-description">
            Understand sequence processing and the vanishing gradient problem.
          </p>
          <button className="card-cta-btn disabled" disabled>
            Explore
          </button>
        </div>



        {/* Placeholder Card: Reinforcement Learning */}
        <div className="concept-card-item placeholder-card">
          <div className="card-badge coming-soon">Coming Soon</div>
          <h2 className="card-title">Reinforcement Learning</h2>
          <p className="card-description">
            Agents learning from rewards, policies, and value functions.
          </p>
          <button className="card-cta-btn disabled" disabled>
            Explore
          </button>
        </div>

        {/* Placeholder Card: Classic Search Strategies */}
        <div className="concept-card-item placeholder-card">
          <div className="card-badge coming-soon">Coming Soon</div>
          <h2 className="card-title">Classic Search Strategies</h2>
          <p className="card-description">
            A*, BFS, DFS, and other fundamental algorithms for problem solving.
          </p>
          <button className="card-cta-btn disabled" disabled>
            Explore
          </button>
          


        


        


        {/* INSERT_NEW_CARD_HERE */}
      </div>
      </div>
    </div>
  );
}

export default ConceptExplorer;
