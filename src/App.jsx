import React, { useState, useEffect } from 'react';
import './App.css';
import TimerScoreWidget from './TimerScoreWidget';
import QuizSection from './QuizSection';
import ConceptSequence from './ConceptSequence';

const transformerExplanations = {
  topic: "Transformer Architecture",
  levels: {
    beginner: {
      label: "Beginner",
      content: (
        <>
          <p>
            Imagine you are reading a sentence. To understand the word "bank" in "river bank", you need to look at the word "river". 
            Older AI models read one word at a time, from left to right, like a person with a very short memory. 
            If the sentence was very long, they might forget the beginning by the time they reached the end.
          </p>
          <p>
            <strong>Transformers</strong> are different. They can look at the <em>entire</em> sentence at once. 
            They use a mechanism called <strong>Self-Attention</strong> to "pay attention" to all relevant words simultaneously, 
            understanding context much better and faster than previous methods.
          </p>
        </>
      )
    },
    intermediate: {
      label: "Intermediate",
      content: (
        <>
          <p>
            The Transformer architecture, introduced in the paper "Attention Is All You Need" (2017), abandoned recurrence (RNNs) entirely in favor of an attention-based mechanism.
            It consists of an <strong>Encoder</strong> (which processes the input) and a <strong>Decoder</strong> (which generates output), though models like BERT use only the Encoder, and GPT uses only the Decoder.
          </p>
          <h3>Key Components:</h3>
          <ul>
            <li><strong>Embeddings:</strong> Words are converted into dense vectors.</li>
            <li><strong>Positional Encoding:</strong> Since processing is parallel, these vectors are added to give the model information about word order.</li>
            <li><strong>Self-Attention:</strong> The core mechanism that weighs the importance of other words in the sequence relative to the current word.</li>
            <li><strong>Feed-Forward Networks:</strong> Apply non-linear transformations to the attention outputs.</li>
          </ul>
        </>
      )
    },
    advanced: {
      label: "Advanced",
      content: (
        <>
          <p>
            At a technical level, the Transformer relies on <strong>Scaled Dot-Product Attention</strong>. 
            For each token, we generate three vectors: <strong>Query (Q)</strong>, <strong>Key (K)</strong>, and <strong>Value (V)</strong>.
          </p>
          <p>
            The attention score is calculated as:
            <br />
            <code>Attention(Q, K, V) = softmax(QK<sup>T</sup> / √d<sub>k</sub>)V</code>
          </p>
          <p>
            This score determines how much focus to place on other parts of the input. 
            The division by <code>√d<sub>k</sub></code> stabilizes gradients. 
            Multi-Head Attention allows the model to jointly attend to information from different representation subspaces at different positions.
            Furthermore, each sub-layer (attention, feed-forward) is surrounded by a <strong>residual connection</strong> followed by <strong>Layer Normalization</strong>, 
            facilitating the training of very deep networks.
          </p>
        </>
      )
    }
  }
};

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
  const [searchQuery, setSearchQuery] = useState('');
  const [inputValue, setInputValue] = useState('');
  const [hasSearched, setHasSearched] = useState(false);
  
  // Timer & Score state
  const [seconds, setSeconds] = useState(0);
  const [score, setScore] = useState(0);
  const [lastMinuteCounted, setLastMinuteCounted] = useState(0);
  const [isTimerVisible, setIsTimerVisible] = useState(true);
  
  // Quiz state
  const [correctQuestionIds, setCorrectQuestionIds] = useState(new Set());

  // Auto-start timer on mount
  useEffect(() => {
    const interval = setInterval(() => {
      setSeconds(prevSeconds => prevSeconds + 1);
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

  const handleSearch = (e) => {
    e.preventDefault();
    if (!inputValue.trim()) return;
    setSearchQuery(inputValue);
    setHasSearched(true);
  };

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
      {hasSearched && (
        <button onClick={toggleTimerVisibility} className="toggle-timer-btn">
          {isTimerVisible ? '👁️ Hide timer & score' : '👁️‍🗨️ Show timer & score'}
        </button>
      )}

      {/* Search Section */}
      <div className={`search-section ${hasSearched ? 'compact' : 'centered'}`}>
        {!hasSearched && <h1 className="main-title">Learn Anything</h1>}
        <form onSubmit={handleSearch} className="search-form">
          <input
            type="text"
            placeholder="What do you want to learn? (e.g., Transformers)"
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            className="search-input"
          />
          <button type="submit" className="search-button">Search</button>
        </form>
      </div>

      {/* Content Area */}
      {hasSearched && (
        <main className="content-area fade-in">
          <header className="topic-header">
            <p className="search-meta">
              Showing results for: <strong>{searchQuery}</strong> (mapped to {transformerExplanations.topic})
            </p>
            <h1>{transformerExplanations.topic}</h1>
          </header>

          {/* Concept Sequence */}
          <ConceptSequence />

          {/* Toy Example */}
          <section className="toy-example-section">
            <h3>Toy Example</h3>
            <div className="toy-example-box">
              <p className="example-sentence">"The <strong>animal</strong> didn't cross the <strong>street</strong> because <strong>it</strong> was too tired."</p>
              <p className="example-explanation">
                Self-Attention helps the model link "it" to "animal" by computing attention scores between all words.
              </p>
            </div>
          </section>

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

          {/* Quiz Section */}
          <QuizSection onCorrectQuestion={handleCorrectQuestion} />
        </main>
      )}

      {/* Timer & Score Widget (bottom-right) */}
      {isTimerVisible && <TimerScoreWidget seconds={seconds} score={score} />}
    </div>
  );
}

export default App;
