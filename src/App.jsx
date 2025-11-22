import React from 'react';
import './App.css';

function App() {
  return (
    <div className="app-container">
      <header className="app-header">
        <h1>Transformer Architecture – Concept Overview</h1>
      </header>
      
      <main className="app-content">
        <section className="info-section">
          <h2>Simple Explanation</h2>
          <p>
            A Transformer is a type of deep learning model introduced in 2017 that revolutionized natural language processing. 
            Unlike previous models (like RNNs) that process words one by one in order, Transformers can process the entire sequence of words at once (in parallel). 
            Their key innovation is "Self-Attention," which allows the model to look at all words in a sentence simultaneously and decide which ones are most relevant to each other, regardless of how far apart they are.
          </p>
        </section>

        <section className="info-section">
          <h2>Detailed Explanation</h2>
          <p>
            The Transformer architecture consists of an <strong>Encoder</strong> and a <strong>Decoder</strong> stack (though models like BERT use only the Encoder, and GPT uses only the Decoder).
            Key components include:
          </p>
          <ul>
            <li><strong>Embeddings:</strong> Converting words into continuous vector representations.</li>
            <li><strong>Positional Encoding:</strong> Since the model processes words in parallel, this adds information about the order of words.</li>
            <li><strong>Self-Attention Mechanism:</strong> Calculates attention scores to weigh the importance of different words relative to the current word (using Query, Key, and Value vectors).</li>
            <li><strong>Feed-Forward Networks:</strong> Processes the information from the attention layer.</li>
            <li><strong>Layer Normalization & Residual Connections:</strong> Helps in stabilizing training and allowing gradients to flow through deep networks.</li>
          </ul>
        </section>

        <section className="info-section">
          <h2>Toy Example</h2>
          <div className="toy-example-box">
            <p className="example-sentence">"The <strong>animal</strong> didn't cross the <strong>street</strong> because <strong>it</strong> was too tired."</p>
            <p className="example-explanation">
              When the model processes the word "<strong>it</strong>", the Self-Attention mechanism assigns a higher weight (attention score) to "<strong>animal</strong>" than to "street". 
              This helps the model understand that "it" refers to the animal, not the street.
            </p>
          </div>
        </section>
      </main>
    </div>
  );
}

export default App;
