import React, { useState, useMemo, useEffect } from 'react';
import './QuizSection.css';
import { quizQuestions } from './data/quizData.js';
import { concepts } from './data/conceptsData.js';
import ConceptSelectionForQuiz from './components/ConceptSelectionForQuiz';

function QuizSection({ onCorrectQuestion, selectedConcepts, onConceptSelectionChange }) {
  const [answers, setAnswers] = useState({});
  const [feedback, setFeedback] = useState({});
  const [correctQuestions, setCorrectQuestions] = useState(new Set());
  const [selectedLevel, setSelectedLevel] = useState('All');
  const [shuffledQuestions, setShuffledQuestions] = useState([]);
  const [showConceptSelection, setShowConceptSelection] = useState(false);

  // Filter questions based on selected concepts and difficulty level
  // Then shuffle and limit to 10
  useEffect(() => {
    const filtered = quizQuestions.filter(q => {
      const matchesConcept = selectedConcepts.includes(q.conceptTag);
      const matchesLevel = selectedLevel === 'All' || q.level === selectedLevel;
      return matchesConcept && matchesLevel;
    });

    // Fisher-Yates shuffle
    const shuffled = [...filtered];
    for (let i = shuffled.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
    }

    setShuffledQuestions(shuffled.slice(0, 10));
    
    // Reset state when filters change
    setAnswers({});
    setFeedback({});
    
  }, [selectedConcepts, selectedLevel]);

  // Calculate question counts per concept (total available)
  const questionCounts = useMemo(() => {
    const counts = {};
    quizQuestions.forEach(q => {
      counts[q.conceptTag] = (counts[q.conceptTag] || 0) + 1;
    });
    return counts;
  }, []);

  const handleReset = () => {
    setAnswers({});
    setFeedback({});
  };

  const handleMCQAnswer = (questionId, optionIndex) => {
    const question = quizQuestions.find(q => q.id === questionId);
    const isCorrect = optionIndex === question.correctOptionIndex;
    
    setAnswers(prev => ({ ...prev, [questionId]: optionIndex }));
    setFeedback(prev => ({ ...prev, [questionId]: { isCorrect, explanation: question.explanation } }));
    
    if (isCorrect && !correctQuestions.has(questionId)) {
      setCorrectQuestions(prev => new Set(prev).add(questionId));
      onCorrectQuestion(questionId);
    }
  };

  const handleTrueFalseAnswer = (questionId, userAnswer) => {
    const question = quizQuestions.find(q => q.id === questionId);
    const isCorrect = userAnswer === question.answer;
    
    setAnswers(prev => ({ ...prev, [questionId]: userAnswer }));
    setFeedback(prev => ({ ...prev, [questionId]: { isCorrect, explanation: question.explanation } }));
    
    if (isCorrect && !correctQuestions.has(questionId)) {
      setCorrectQuestions(prev => new Set(prev).add(questionId));
      onCorrectQuestion(questionId);
    }
  };

  const handleFreeFormSubmit = (questionId, userText) => {
    const question = quizQuestions.find(q => q.id === questionId);
    const lowerText = userText.toLowerCase();
    
    // Check if at least 1 keyword is present (relaxed for better UX)
    const matchedKeywords = question.keywords.filter(keyword => 
      lowerText.includes(keyword.toLowerCase())
    );
    const isCorrect = matchedKeywords.length >= 1;
    
    setAnswers(prev => ({ ...prev, [questionId]: userText }));
    setFeedback(prev => ({ 
      ...prev, 
      [questionId]: { 
        isCorrect, 
        explanation: question.explanation,
        matchedKeywords: matchedKeywords.length
      } 
    }));
    
    if (isCorrect && !correctQuestions.has(questionId)) {
      setCorrectQuestions(prev => new Set(prev).add(questionId));
      onCorrectQuestion(questionId);
    }
  };

  return (
    <section className="quiz-section">
      <h2>Quiz: Test Your Understanding</h2>
      <p className="quiz-intro">
        Answer these questions to test your knowledge about Transformers. Each correct answer awards <strong>+5 points</strong>!
      </p>

      <div className="quiz-filters-container">
        <button 
          className="toggle-concepts-btn"
          onClick={() => setShowConceptSelection(!showConceptSelection)}
        >
          {showConceptSelection ? 'Hide Concept Filters' : 'Filter by Concepts'}
        </button>

        {showConceptSelection && (
          <ConceptSelectionForQuiz
            concepts={concepts}
            selectedConcepts={selectedConcepts}
            onSelectionChange={onConceptSelectionChange}
            questionCounts={questionCounts}
          />
        )}

        <div className="difficulty-filter">
          <span className="filter-label">Difficulty Level:</span>
          <div className="filter-buttons">
            {['All', 'Beginner', 'Intermediate', 'Advanced'].map(level => (
              <button
                key={level}
                className={`filter-btn ${selectedLevel === level ? 'active' : ''}`}
                onClick={() => setSelectedLevel(level)}
              >
                {level}
              </button>
            ))}
          </div>
        </div>
      </div>

      <div className="quiz-controls">
        <div className="quiz-progress">
          Showing <strong>{shuffledQuestions.length}</strong> questions. 
          You've answered <strong>{correctQuestions.size}</strong> correctly total.
        </div>
        <button className="reset-quiz-btn" onClick={handleReset}>
          Reset Quiz
        </button>
      </div>

      <div className="quiz-questions">
        {shuffledQuestions.length > 0 ? (
          shuffledQuestions.map((question, index) => (
            <div key={question.id} className="question-card">
              <div className="question-header">
                <span className="question-number">Question {index + 1}</span>
                <span className={`difficulty-badge ${question.level.toLowerCase()}`}>{question.level}</span>
              </div>
              
              {question.type === 'mcq' && (
                <MCQQuestion
                  question={question}
                  selectedOption={answers[question.id]}
                  feedback={feedback[question.id]}
                  onAnswer={(optionIndex) => handleMCQAnswer(question.id, optionIndex)}
                />
              )}
              
              {question.type === 'true_false' && (
                <TrueFalseQuestion
                  question={question}
                  selectedAnswer={answers[question.id]}
                  feedback={feedback[question.id]}
                  onAnswer={(answer) => handleTrueFalseAnswer(question.id, answer)}
                />
              )}
              
              {question.type === 'free_form' && (
                <FreeFormQuestion
                  question={question}
                  userAnswer={answers[question.id]}
                  feedback={feedback[question.id]}
                  onSubmit={(text) => handleFreeFormSubmit(question.id, text)}
                />
              )}
            </div>
          ))
        ) : (
          <div className="no-questions">
            No questions found for the selected filters. Try selecting more concepts or a different difficulty level.
          </div>
        )}
      </div>
    </section>
  );
}

function MCQQuestion({ question, selectedOption, feedback, onAnswer }) {
  return (
    <>
      <p className="question-text">{question.question}</p>
      <div className="mcq-options">
        {question.options.map((option, index) => (
          <button
            key={index}
            className={`quiz-option-btn ${selectedOption === index ? 'selected' : ''}`}
            onClick={() => onAnswer(index)}
            disabled={feedback !== undefined}
          >
            {option}
          </button>
        ))}
      </div>
      {feedback && (
        <div className={`feedback ${feedback.isCorrect ? 'feedback-correct' : 'feedback-incorrect'}`}>
          {feedback.isCorrect ? '✅ Correct! +5 points' : '❌ Incorrect'}
          <p className="feedback-explanation">{feedback.explanation}</p>
        </div>
      )}
    </>
  );
}

function TrueFalseQuestion({ question, selectedAnswer, feedback, onAnswer }) {
  return (
    <>
      <p className="question-text">{question.question}</p>
      <div className="true-false-buttons">
        <button
          className={`quiz-option-btn ${selectedAnswer === true ? 'selected' : ''}`}
          onClick={() => onAnswer(true)}
          disabled={feedback !== undefined}
        >
          True
        </button>
        <button
          className={`quiz-option-btn ${selectedAnswer === false ? 'selected' : ''}`}
          onClick={() => onAnswer(false)}
          disabled={feedback !== undefined}
        >
          False
        </button>
      </div>
      {feedback && (
        <div className={`feedback ${feedback.isCorrect ? 'feedback-correct' : 'feedback-incorrect'}`}>
          {feedback.isCorrect ? '✅ Correct! +5 points' : '❌ Incorrect'}
          <p className="feedback-explanation">{feedback.explanation}</p>
        </div>
      )}
    </>
  );
}

function FreeFormQuestion({ question, userAnswer, feedback, onSubmit }) {
  const [inputText, setInputText] = useState('');

  const handleSubmit = () => {
    if (inputText.trim()) {
      onSubmit(inputText);
    }
  };

  return (
    <>
      <p className="question-text">{question.question}</p>
      <textarea
        className="free-form-input"
        placeholder="Type your answer here..."
        value={inputText}
        onChange={(e) => setInputText(e.target.value)}
        rows={3}
        disabled={feedback !== undefined}
      />
      <button 
        className="submit-answer-btn" 
        onClick={handleSubmit}
        disabled={feedback !== undefined}
      >
        Submit Answer
      </button>
      {feedback && (
        <div className={`feedback ${feedback.isCorrect ? 'feedback-correct' : 'feedback-incorrect'}`}>
          {feedback.isCorrect 
            ? `✅ Correct! +5 points` 
            : '❌ Not quite'}
          <p className="feedback-explanation">{feedback.explanation}</p>
        </div>
      )}
    </>
  );
}

export default QuizSection;
