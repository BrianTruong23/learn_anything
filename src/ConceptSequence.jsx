import React, { useState } from 'react';
import './ConceptSequence.css';

const transformerConceptSequence = [
  {
    id: "big_picture",
    title: "Transformer: Big Picture",
    motivation: "Traditional RNN-based models process sequences one token at a time, which is slow and makes it hard to learn long-range dependencies. Transformers were introduced to solve these problems by processing all tokens in parallel while still capturing relationships between them.",
    whatItIs: "A Transformer is a neural network architecture built around the self-attention mechanism. Instead of processing tokens sequentially, it looks at all tokens simultaneously and learns which tokens are most relevant to each other. This parallel processing makes training much faster and helps the model understand context better.",
    toyExample: "Consider the sentence: 'The cat sat on the mat because it was comfortable.' A Transformer can look at all words at once and learn that 'it' refers to 'the mat' (not 'the cat') by computing attention scores between all word pairs simultaneously."
  },
  {
    id: "positional_embeddings",
    title: "Positional Embeddings",
    motivation: "Since Transformers process all tokens in parallel (not sequentially like RNNs), the model has no inherent sense of word order. Without position information, 'the cat chased the dog' and 'the dog chased the cat' would look identical to the model.",
    whatItIs: "Positional embeddings are vectors that encode the position of each token in the sequence. These are added to the token embeddings before feeding them into the Transformer, giving the model information about token order. The original paper used sinusoidal functions to generate these position vectors.",
    toyExample: "For the phrase 'the cat chased the dog', we add position vectors: position_0 to 'the', position_1 to 'cat', position_2 to 'chased', etc. This way, swapping 'cat' and 'dog' changes the input vectors, allowing the model to distinguish different word orders."
  },
  {
    id: "self_attention",
    title: "Self-Attention",
    motivation: "Fixed-window or purely local interactions (like in CNNs) limit a model's ability to relate distant tokens. We want each token to be able to 'look at' any other token in the sequence, regardless of distance, to capture long-range dependencies effectively.",
    whatItIs: "Self-attention is a mechanism where each token generates three vectors: a Query (Q), a Key (K), and a Value (V). The model computes attention scores by measuring how similar each Query is to all Keys (using dot products), then uses these scores to create a weighted sum of the Values. This lets each token attend to relevant context from anywhere in the sequence.",
    toyExample: "In 'The animal didn't cross the street because it was too tired', the word 'it' creates a Query that compares against Keys from all other words. The Key for 'animal' scores highest, so 'it' attends most to 'animal', helping the model understand the reference."
  },
  {
    id: "multi_head_attention",
    title: "Multi-Head Attention",
    motivation: "A single attention pattern might capture one type of relationship (e.g., syntactic structure) but miss others (e.g., semantic similarity or coreference). Different relationships are important for understanding language, so we want multiple parallel attention mechanisms.",
    whatItIs: "Multi-Head Attention runs multiple self-attention operations (called 'heads') in parallel, each with its own learned Query, Key, and Value projections. Each head can learn to focus on different types of relationships. The outputs from all heads are concatenated and linearly transformed to produce the final result.",
    toyExample: "In 'The quick brown fox jumps', one attention head might focus on adjective-noun pairs ('quick'→'fox', 'brown'→'fox'), while another head focuses on subject-verb relationships ('fox'→'jumps'). This multi-faceted attention captures richer linguistic patterns."
  },
  {
    id: "feedforward_block",
    title: "Feedforward / MLP Block",
    motivation: "While attention allows tokens to gather information from each other, we still need non-linear transformations to process and refine this information. The attention mechanism alone doesn't provide enough computational depth to transform representations effectively.",
    whatItIs: "After each attention layer, a position-wise feedforward network (a small multi-layer perceptron) is applied independently to each token. This typically consists of two linear transformations with a ReLU activation in between. The same feedforward network is applied to every position, but each token is processed independently.",
    toyExample: "After attention gathers context (e.g., 'cat' + information from 'fluffy', 'pet', 'meowed'), the feedforward block applies the same two-layer MLP to refine each token's representation: transform → ReLU → transform. This adds expressiveness and helps integrate the attended information."
  }
];

function ConceptSequence() {
  const [currentConceptIndex, setCurrentConceptIndex] = useState(0);

  const currentConcept = transformerConceptSequence[currentConceptIndex];
  const isFirstConcept = currentConceptIndex === 0;
  const isLastConcept = currentConceptIndex === transformerConceptSequence.length - 1;

  const handlePrevious = () => {
    if (!isFirstConcept) {
      setCurrentConceptIndex(prev => prev - 1);
    }
  };

  const handleNext = () => {
    if (!isLastConcept) {
      setCurrentConceptIndex(prev => prev + 1);
    }
  };

  return (
    <div className="concept-sequence">
      <div className="progress-indicator">
        Step {currentConceptIndex + 1} of {transformerConceptSequence.length}: {currentConcept.title}
      </div>

      <div className="concept-content">
        <h2 className="concept-title">{currentConcept.title}</h2>

        <div className="concept-section">
          <h3 className="section-heading">💡 Motivation</h3>
          <p className="section-content">{currentConcept.motivation}</p>
        </div>

        <div className="concept-section">
          <h3 className="section-heading">📚 What it is</h3>
          <p className="section-content">{currentConcept.whatItIs}</p>
        </div>

        <div className="concept-section">
          <h3 className="section-heading">🔍 Toy Example</h3>
          <p className="section-content">{currentConcept.toyExample}</p>
        </div>
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
          {transformerConceptSequence.map((_, index) => (
            <span 
              key={index} 
              className={`dot ${index === currentConceptIndex ? 'active' : ''}`}
              onClick={() => setCurrentConceptIndex(index)}
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
