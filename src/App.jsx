import React, { useState, useEffect } from 'react';
import './App.css';
import TimerScoreWidget from './TimerScoreWidget';
import QuizSection from './QuizSection';
import ConceptSequence from './ConceptSequence';
import ConceptSummary from './components/ConceptSummary';
import ConceptExplorer from './components/ConceptExplorer';
import { transformerConcepts, bertConcepts, cnnConcepts } from './data/conceptsData.js';
import { transformerSources, bertSources } from './data/sourcesData.js';



function App() {

  
  // Navigation state
  const [currentView, setCurrentView] = useState('explorer'); // 'explorer' or 'transformer'
  
  // Timer & Score state
  const [seconds, setSeconds] = useState(0);
  const [score, setScore] = useState(0);
  const [lastMinuteCounted, setLastMinuteCounted] = useState(0);
  const [isTimerVisible, setIsTimerVisible] = useState(true);
  
  // Quiz state
  const [correctQuestionIds, setCorrectQuestionIds] = useState(new Set());
  
  // Difficulty level state
  const [difficulty, setDifficulty] = useState('beginner');
  
  // Selected concepts state (for quiz filtering)
  // Selected concepts state (for quiz filtering)
  const [selectedConcepts, setSelectedConcepts] = useState([]);

  // Update selected concepts when view changes
  useEffect(() => {
    if (currentView === 'latent') {
      setSelectedConcepts(latentConcepts.map(c => c.id));
    } else if (currentView === 'cnn') {
      setSelectedConcepts(cnnConcepts.map(c => c.id));
    } else if (currentView === 'bert') {
      setSelectedConcepts(bertConcepts.map(c => c.id));
    } else if (currentView === 'transformer') {
      setSelectedConcepts(transformerConcepts.map(c => c.id));
    } else {
      // Default to all if needed, or empty
      setSelectedConcepts([]);
    }
  }, [currentView, transformerConcepts, bertConcepts, cnnConcepts]);
  
  // Quiz visibility state
  const [isQuizVisible, setIsQuizVisible] = useState(false);

  // Auto-start timer on mount
  useEffect(() => {
    const interval = setInterval(() => {
      if (!document.hidden) {
        setSeconds(prevSeconds => prevSeconds + 1);
      }
    }, 1000);
    
    return () => clearInterval(interval);
  }, []);

  // Automatic scoring: 1 point per minute
  useEffect(() => {
    const minutesElapsed = Math.floor(seconds / 60);
    if (minutesElapsed > lastMinuteCounted) {
      setScore(prev => prev + (minutesElapsed - lastMinuteCounted));
      setLastMinuteCounted(minutesElapsed);
    }
  }, [seconds, lastMinuteCounted]);



  const toggleTimerVisibility = () => {
    setIsTimerVisible(prev => !prev);
  };

  const handleCorrectQuestion = (questionId) => {
    setCorrectQuestionIds(prev => {
      if (prev.has(questionId)) return prev; // already scored
      const updated = new Set(prev);
      updated.add(questionId);
      setScore(oldScore => oldScore + 5); // award 5 points per new correct question
      return updated;
    });
  };

  return (

    <div className="app-container">
      {/* Toggle Timer Button */}
      <button onClick={toggleTimerVisibility} className="toggle-timer-btn">
        {isTimerVisible ? '👁️ Hide timer & score' : '👁️‍🗨️ Show timer & score'}
      </button>

      {/* Timer & Score Widget */}
      {isTimerVisible && (
        <TimerScoreWidget 
          seconds={seconds} 
          score={score} 
        />
      )}

      {/* Content Area */}
      <main className="content-area fade-in">
        {currentView === 'explorer' ? (
          <ConceptExplorer 
            onSelectConcept={(conceptId) => {
              if (conceptId === 'transformer' || conceptId === 'bert' || conceptId === 'cnn' || conceptId === 'latent') {
                setCurrentView(conceptId);
                window.scrollTo({ top: 0, behavior: 'smooth' });
              }
            }} 
          />
        ) : (
          <>
            <header className="topic-header">
              <div style={{ marginBottom: '1rem' }}>
                <button 
                  onClick={() => setCurrentView('explorer')}
                  style={{
                    background: 'none',
                    border: 'none',
                    color: '#64748B',
                    cursor: 'pointer',
                    fontSize: '0.9rem',
                    display: 'flex',
                    alignItems: 'center',
                    gap: '0.5rem',
                    padding: 0
                  }}
                >
                  ← Back to Explorer
                </button>
              </div>
              <header className="app-header">
                <h1>{currentView === 'latent' ? 'Latent Architecture' : currentView === 'cnn' ? 'CNN Architecture' : currentView === 'bert' ? 'BERT Architecture' : 'Transformer Architecture'}</h1>
                {currentView === 'transformer' && (
                  <p className="intro-text" style={{ maxWidth: '800px', margin: '0 auto 1.5rem', lineHeight: '1.6', color: '#4a5568' }}>
                    We’ll build the transformer story step by step, like a ladder. We start from the big picture of what a transformer is trying to do, then move into embeddings and positions, then self-attention, then multi-head attention, then Add & LayerNorm + feed-forward, and finally how these pieces assemble into the encoder and decoder stacks that make up the full architecture.
                  </p>
                )}
              </header>
            </header>

            {/* Concept Sequence */}
            <ConceptSequence 
              key={currentView}
              difficulty={difficulty}
              onDifficultyChange={setDifficulty}
              setScore={setScore}
              concepts={
                currentView === 'cnn' ? cnnConcepts : 
                currentView === 'bert' ? bertConcepts : 
                transformerConcepts
              } 
            />
            <div className="app-summary-section">
              <ConceptSummary sectionTitle="transformers" setScore={setScore} />
            </div>

            {/* Sources Section */}
            <section className="sources-section">
              <h2>Learn from papers & credible sources</h2>
              <p className="sources-intro">
                Here are some foundational resources to go deeper into Transformers:
              </p>
              <div className="sources-grid">
                {(currentView === 'latent' ? [] : currentView === 'cnn' ? [] : currentView === 'bert' ? bertSources : transformerSources).map((source, index) => (
                  <div key={index} className="source-card">
                    <div className="source-header">
                      <span className="source-type">{source.type}</span>
                    </div>
                    <h3 className="source-title">
                      <a href={source.url} target="_blank" rel="noreferrer">
                        {source.title}
                      </a>
                    </h3>
                    <p className="source-description">{source.description}</p>
                  </div>
                ))}
              </div>
            </section>

            {/* Quiz Toggle Button */}
            <div className="quiz-toggle-container">
              <button 
                onClick={() => setIsQuizVisible(!isQuizVisible)}
                className="quiz-toggle-btn"
              >
                {isQuizVisible ? '📚 Hide Quiz' : '📝 Show Quiz'}
              </button>
            </div>

            {/* Quiz Section - Collapsible */}
            <div className={`quiz-wrapper ${isQuizVisible ? 'quiz-visible' : 'quiz-hidden'}`}>
              <QuizSection 
                onCorrectQuestion={handleCorrectQuestion}
                selectedConcepts={selectedConcepts}
                onConceptSelectionChange={setSelectedConcepts}
              />
            </div>
          </>
        )}
      </main>

      {/* Footer */}
      <footer className="app-footer">
        <p>
          &copy; {new Date().getFullYear()} Learn Anything. Created by{' '}
          <a href="https://truongthoithang.com" target="_blank" rel="noopener noreferrer">
            Thang Truong
          </a>
        </p>
      </footer>
    </div>
  );
}


export default App;
