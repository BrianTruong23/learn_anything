import React, { useState, useEffect } from 'react';
import './App.css';
import TimerScoreWidget from './TimerScoreWidget';
import QuizSection from './QuizSection';
import ConceptSequence from './ConceptSequence';
import ConceptSummary from './components/ConceptSummary';
import ConceptExplorer from './components/ConceptExplorer';
import { concepts } from './data/conceptsData.js';

const transformerSources = [
  {
    title: "Attention Is All You Need (Vaswani et al., 2017)",
    type: "Paper",
    url: "https://arxiv.org/abs/1706.03762",
    description: "The original paper that introduced the Transformer architecture."
  },
  {
    title: "The Illustrated Transformer",
    type: "Blog",
    url: "https://jalammar.github.io/illustrated-transformer/",
    description: "A visual and intuitive explanation of the Transformer."
  },
  {
    title: "Transformers from Scratch",
    type: "Article",
    url: "https://e2eml.school/transformers.html",
    description: "Building intuition for how Transformers work from the ground up."
  },
  {
    title: "Stanford CS224N: Transformers and Self-Attention",
    type: "Course",
    url: "https://web.stanford.edu/class/cs224n/",
    description: "Academic course materials covering NLP and Transformers in depth."
  }
];

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
  const allConceptIds = concepts.map(c => c.id);
  const [selectedConcepts, setSelectedConcepts] = useState(allConceptIds);
  
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
              if (conceptId === 'transformer') {
                setCurrentView('transformer');
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
              <h1>Transformer Architecture</h1>
            </header>

            {/* Concept Sequence */}
            <ConceptSequence 
              difficulty={difficulty}
              onDifficultyChange={setDifficulty}
              setScore={setScore}
            />

            {/* Summary Section */}
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
                {transformerSources.map((source, index) => (
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
    </div>
  );
}


export default App;
