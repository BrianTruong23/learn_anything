import React, { useState, useMemo } from 'react';
import './QuizSection.css';
import { quizQuestions } from './data/quizQuestionsData.js';
import { concepts } from './data/conceptsData.js';
import ConceptSelectionForQuiz from './components/ConceptSelectionForQuiz';

function QuizSection({ onCorrectQuestion, selectedConcepts, onConceptSelectionChange }) {
  const [answers, setAnswers] = useState({});
  const [feedback, setFeedback] = useState({});
  const [correctQuestions, setCorrectQuestions] = useState(new Set());

  // Filter questions based on selected concepts
  const filteredQuestions = useMemo(() => {
    return quizQuestions.filter(q => selectedConcepts.includes(q.conceptTag));
  }, [selectedConcepts]);

  // Calculate question counts per concept
  const questionCounts = useMemo(() => {
    const counts = {};
    quizQuestions.forEach(q => {
      counts[q.conceptTag] = (counts[q.conceptTag] || 0) + 1;
    });
    return counts;
  }, []);

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
    
    // Check if at least 2 keywords are present
    const matchedKeywords = question.keywords.filter(keyword => 
      lowerText.includes(keyword.toLowerCase())
    );
    const isCorrect = matchedKeywords.length >= 2;
    
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

      <ConceptSelectionForQuiz
        concepts={concepts}
        selectedConcepts={selectedConcepts}
        onSelectionChange={onConceptSelectionChange}
        questionCounts={questionCounts}
      />

      <div className="quiz-progress">
        You've answered <strong>{correctQuestions.size}/{filteredQuestions.length}</strong> questions correctly.
      </div>

      <div className="quiz-questions">
        {filteredQuestions.map((question, index) => (
          <div key={question.id} className="question-card">
            <div className="question-number">Question {index + 1}</div>
            
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
        ))}
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
      <p className="question-text">{question.statement}</p>
      <div className="true-false-buttons">
        <button
          className={`quiz-option-btn ${selectedAnswer === true ? 'selected' : ''}`}
          onClick={() => onAnswer(true)}
        >
          True
        </button>
        <button
          className={`quiz-option-btn ${selectedAnswer === false ? 'selected' : ''}`}
          onClick={() => onAnswer(false)}
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
      />
      <button className="submit-answer-btn" onClick={handleSubmit}>
        Submit Answer
      </button>
      {feedback && (
        <div className={`feedback ${feedback.isCorrect ? 'feedback-correct' : 'feedback-incorrect'}`}>
          {feedback.isCorrect 
            ? `✅ Correct! +5 points (matched ${feedback.matchedKeywords} keywords)` 
            : '❌ Not quite'}
          <p className="feedback-explanation">{feedback.explanation}</p>
        </div>
      )}
    </>
  );
}

export default QuizSection;
