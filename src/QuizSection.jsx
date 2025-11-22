import React, { useState } from 'react';
import './QuizSection.css';

const quizQuestions = [
  {
    id: "q1",
    type: "mcq",
    question: "What is the core mechanism that differentiates Transformers from RNNs?",
    options: [
      "Recurrent connections over time",
      "Self-attention over all tokens in parallel",
      "Convolution over fixed-size windows",
      "Hand-crafted linguistic features"
    ],
    correctOptionIndex: 1,
    explanation: "Transformers use self-attention to relate all tokens in a sequence in parallel, unlike RNNs which process sequentially."
  },
  {
    id: "q2",
    type: "true_false",
    statement: "Transformers must process tokens strictly left-to-right, just like standard RNNs.",
    answer: false,
    explanation: "False. Transformers process tokens in parallel using self-attention, not sequentially like RNNs."
  },
  {
    id: "q3",
    type: "free_form",
    question: "In simple terms, why does self-attention help with long-range dependencies?",
    keywords: ["attend", "token", "parallel", "dependencies", "direct"],
    explanation: "Each token can directly attend to any other token in the sequence without passing information step by step, making long-range dependencies easier to capture."
  },
  {
    id: "q4",
    type: "mcq",
    question: "What is the purpose of positional encoding in Transformers?",
    options: [
      "To make the model run faster",
      "To reduce memory usage",
      "To give the model information about token order",
      "To prevent overfitting"
    ],
    correctOptionIndex: 2,
    explanation: "Since Transformers process all tokens in parallel, positional encodings are added to give the model information about the order of tokens in the sequence."
  },
  {
    id: "q5",
    type: "true_false",
    statement: "BERT uses only the Decoder part of the Transformer architecture.",
    answer: false,
    explanation: "False. BERT uses only the Encoder part of the Transformer. GPT uses only the Decoder part."
  },
  {
    id: "q6",
    type: "mcq",
    question: "In the self-attention mechanism, what are the three vectors computed for each token?",
    options: [
      "Input, Output, Hidden",
      "Query, Key, Value",
      "Encoder, Decoder, Embedding",
      "Position, Context, Attention"
    ],
    correctOptionIndex: 1,
    explanation: "The three vectors in self-attention are Query (Q), Key (K), and Value (V), which are used to compute attention scores."
  },
  {
    id: "q7",
    type: "free_form",
    question: "What operation is applied to the attention scores before they're used to weight the Values?",
    keywords: ["softmax", "normalize", "normalized"],
    explanation: "The softmax function is applied to normalize the attention scores, ensuring they sum to 1 and can be interpreted as probabilities."
  },
  {
    id: "q8",
    type: "mcq",
    question: "What is the main benefit of Multi-Head Attention compared to single-head attention?",
    options: [
      "It reduces computational cost",
      "It allows the model to attend to information from different representation subspaces",
      "It eliminates the need for positional encoding",
      "It makes training faster"
    ],
    correctOptionIndex: 1,
    explanation: "Multi-Head Attention allows the model to jointly attend to information from different representation subspaces at different positions."
  },
  {
    id: "q9",
    type: "true_false",
    statement: "Residual connections in Transformers help with gradient flow during backpropagation.",
    answer: true,
    explanation: "True. Residual connections (skip connections) help gradients flow through deep networks, making training more stable."
  },
  {
    id: "q10",
    type: "free_form",
    question: "What mathematical operation combines the Query and Key to compute attention scores?",
    keywords: ["dot product", "multiply", "multiplication"],
    explanation: "The dot product (Q·K^T) is used to compute the attention scores, measuring the similarity between queries and keys."
  }
];

function QuizSection({ onCorrectQuestion }) {
  const [answers, setAnswers] = useState({});
  const [feedback, setFeedback] = useState({});
  const [correctQuestions, setCorrectQuestions] = useState(new Set());

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
      <div className="quiz-progress">
        You've answered <strong>{correctQuestions.size}/10</strong> questions correctly.
      </div>

      <div className="quiz-questions">
        {quizQuestions.map((question, index) => (
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
