export const quizQuestions = [
  // Concept 0: Big Picture questions
  {
    id: "q1",
    type: "mcq",
    conceptTag: "concept_0",
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
    conceptTag: "concept_0",
    statement: "Transformers must process tokens strictly left-to-right, just like standard RNNs.",
    answer: false,
    explanation: "False. Transformers process tokens in parallel using self-attention, not sequentially like RNNs."
  },
  {
    id: "q3",
    type: "free_form",
    conceptTag: "concept_0",
    question: "In simple terms, why does parallel processing in Transformers help with training speed?",
    keywords: ["parallel", "gpu", "simultaneous", "faster", "at once"],
    explanation: "Parallel processing allows all tokens to be processed simultaneously on GPU cores, unlike sequential RNN processing which must wait for each step to complete."
  },
  
  // Concept 1: Positional Embeddings questions
  {
    id: "q4",
    type: "mcq",
    conceptTag: "concept_1",
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
    conceptTag: "concept_1",
    statement: "Without positional embeddings, the sentences 'dog bites man' and 'man bites dog' would have identical representations.",
    answer: true,
    explanation: "True. Self-attention is permutation-invariant, so without position information, any permutation of tokens produces the same output."
  },
  {
    id: "q6",
    type: "free_form",
    conceptTag: "concept_1",
    question: "Why did the original Transformer use sinusoidal functions for positional encoding instead of learned embeddings?",
    keywords: ["extrapolate", "generalize", "longer", "unseen", "length"],
    explanation: "Sinusoidal encodings allow the model to extrapolate to sequence lengths not seen during training, providing better generalization to longer sequences."
  },

  // Concept 2: Self-Attention questions
  {
    id: "q7",
    type: "mcq",
    conceptTag: "concept_2",
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
    id: "q8",
    type: "free_form",
    conceptTag: "concept_2",
    question: "What operation is applied to the attention scores before they're used to weight the Values?",
    keywords: ["softmax", "normalize", "normalized"],
    explanation: "The softmax function is applied to normalize the attention scores, ensuring they sum to 1 and can be interpreted as probabilities."
  },
  {
    id: "q9",
    type: "true_false",
    conceptTag: "concept_2",
    statement: "The scaling factor (dividing by √d_k) in attention prevents the softmax from saturating in regions with small gradients.",
    answer: true,
    explanation: "True. Without scaling, large dot products push the softmax into regions where gradients are extremely small, hindering learning."
  },

  // Concept 3: Multi-Head Attention questions
  {
    id: "q10",
    type: "mcq",
    conceptTag: "concept_3",
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
    id: "q11",
    type: "true_false",
    conceptTag: "concept_3",
    statement: "In multi-head attention with 8 heads and d_model=512, each head typically uses dimension d_k=64.",
    answer: true,
    explanation: "True. Each head uses d_k = d_model / h = 512 / 8 = 64 dimensions."
  },
  {
    id: "q12",
    type: "free_form",
    conceptTag: "concept_3",
    question: "Why might different attention heads learn to specialize in different linguistic patterns?",
    keywords: ["subspace", "different", "independent", "separate", "learn"],
    explanation: "Each head has independent parameters and operates in a different representation subspace, allowing it to learn distinct attention patterns (e.g., syntactic vs. semantic relationships)."
  },

  // Concept 4: Feedforward Block questions
  {
    id: "q13",
    type: "mcq",
    conceptTag: "concept_4",
    question: "What is the typical ratio between the feedforward hidden dimension (d_ff) and the model dimension (d_model)?",
    options: [
      "1:1 (same size)",
      "2:1 (2× larger)",
      "4:1 (4× larger)",
      "8:1 (8× larger)"
    ],
    correctOptionIndex: 2,
    explanation: "Typically d_ff = 4 × d_model (e.g., 2048 when d_model = 512), providing the intermediate expansion."
  },
  {
    id: "q14",
    type: "true_false",
    conceptTag: "concept_4",
    statement: "The feedforward network applies the same transformation to each token position independently.",
    answer: true,
    explanation: "True. It's 'position-wise', meaning the same parameters are used for all positions, but each position is transformed independently."
  },
  {
    id: "q15",
    type: "free_form",
    conceptTag: "concept_4",
    question: "Why does the feedforward block need a non-linear activation function like ReLU?",
    keywords: ["non-linear", "complexity", "learn", "transform", "express"],
    explanation: "Non-linearity enables the network to learn complex transformations and increases expressive power. Without it, multiple linear layers would collapse to a single linear transformation."
  },

  // Concept 5: Layer Norm & Residual Connections questions
  {
    id: "q16",
    type: "mcq",
    conceptTag: "concept_5",
    question: "What is the primary purpose of residual connections in Transformers?",
    options: [
      "To reduce the number of parameters",
      "To enable gradient flow through deep networks",
      "To normalize the layer outputs",
      "To add positional information"
    ],
    correctOptionIndex: 1,
    explanation: "Residual connections create gradient highways that allow gradients to flow directly through the network, enabling training of very deep models."
  },
  {
    id: "q17",
    type: "true_false",
    conceptTag: "concept_5",
    statement: "Layer Normalization computes statistics across the batch dimension, similar to Batch Normalization.",
    answer: false,
    explanation: "False. Layer Normalization computes mean and variance across the feature dimension for each sample independently, not across the batch."
  },
  {
    id: "q18",
    type: "free_form",
    conceptTag: "concept_5",
    question: "How do residual connections help prevent vanishing gradients?",
    keywords: ["identity", "direct", "path", "gradient", "flow", "highway"],
    explanation: "Residual connections provide an identity mapping that creates a direct path for gradients to flow backward through the network without diminishing, acting as a gradient highway."
  }
];
