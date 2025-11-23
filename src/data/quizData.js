export const quizQuestions = [
  // --- Big Picture (concept_0) ---
  {
    id: 1,
    question: "What is the primary architectural advantage of the Transformer over RNNs?",
    type: "mcq",
    options: [
      "It requires fewer parameters",
      "It processes input sequences sequentially",
      "It allows for parallel computation across the sequence",
      "It uses convolutional filters"
    ],
    correctOptionIndex: 2,
    answer: "It allows for parallel computation across the sequence",
    level: "Beginner",
    conceptTag: "concept_0",
    explanation: "Transformers process all tokens simultaneously, unlike RNNs which must process them one by one."
  },
  {
    id: 2,
    question: "Which component is used for 'understanding' input sequences in models like BERT?",
    type: "mcq",
    options: [
      "Encoder",
      "Decoder",
      "Generator",
      "Discriminator"
    ],
    correctOptionIndex: 0,
    answer: "Encoder",
    level: "Beginner",
    conceptTag: "concept_0",
    explanation: "The Encoder stack is designed to build bidirectional representations of the input, ideal for understanding tasks."
  },
  {
    id: 3,
    question: "T5 treats every NLP problem as a text-to-text generation problem.",
    type: "true_false",
    answer: true,
    level: "Intermediate",
    conceptTag: "concept_0",
    explanation: "T5 (Text-to-Text Transfer Transformer) unifies tasks by converting inputs and outputs to text strings."
  },
  {
    id: 4,
    question: "According to Chinchilla scaling laws, if you double the model size, you should roughly double the training dataset size.",
    type: "true_false",
    answer: true,
    level: "Advanced",
    conceptTag: "concept_0",
    explanation: "Compute-optimal scaling suggests increasing model size and data size in roughly equal proportions."
  },
  {
    id: 5,
    question: "Explain the difference between 'Encoder' and 'Decoder' roles in the original Transformer.",
    type: "free_form",
    keywords: ["encoder", "decoder", "input", "output", "generation", "understanding"],
    level: "Beginner",
    conceptTag: "concept_0",
    explanation: "The Encoder processes the input to create a representation (understanding), while the Decoder generates the output sequence one token at a time."
  },

  // --- Positional Embeddings (concept_1) ---
  {
    id: 6,
    question: "Why does the Transformer need Positional Encodings?",
    type: "mcq",
    options: [
      "To increase the model size",
      "Because it has no inherent sense of token order",
      "To normalize the input vectors",
      "To mask future tokens"
    ],
    correctOptionIndex: 1,
    answer: "Because it has no inherent sense of token order",
    level: "Beginner",
    conceptTag: "concept_1",
    explanation: "Without recurrence or convolution, the model sees the input as a 'bag of words' unless position info is added."
  },
  {
    id: 7,
    question: "The original Transformer uses learned positional embeddings instead of fixed sinusoidal ones.",
    type: "true_false",
    answer: false,
    level: "Intermediate",
    conceptTag: "concept_1",
    explanation: "The original paper used fixed sinusoidal functions, though learned embeddings are also common (e.g., in BERT)."
  },
  {
    id: 8,
    question: "RoPE (Rotary Positional Embeddings) rotates the Query and Key vectors to encode relative positions.",
    type: "true_false",
    answer: true,
    level: "Advanced",
    conceptTag: "concept_1",
    explanation: "RoPE encodes position by rotating the vectors in the complex plane, preserving relative distance in the dot product."
  },
  {
    id: 9,
    question: "ALiBi achieves length extrapolation by biasing attention scores based on distance.",
    type: "true_false",
    answer: true,
    level: "Advanced",
    conceptTag: "concept_1",
    explanation: "ALiBi adds a static, non-learnable bias to attention scores proportional to the distance between tokens."
  },
  {
    id: 10,
    question: "What happens if you swap two words in the input without positional encodings?",
    type: "free_form",
    keywords: ["same", "identical", "representation", "change", "output"],
    level: "Beginner",
    conceptTag: "concept_1",
    explanation: "Without positional encodings, the self-attention output would be identical because the set of tokens is the same."
  },

  // --- Self-Attention (concept_2) ---
  {
    id: 11,
    question: "In Self-Attention, what does the 'Query' vector represent?",
    type: "mcq",
    options: [
      "The information to be retrieved",
      "The current token looking for relevant information",
      "The content to be matched against",
      "The final output vector"
    ],
    correctOptionIndex: 1,
    answer: "The current token looking for relevant information",
    level: "Intermediate",
    conceptTag: "concept_2",
    explanation: "The Query represents the current token's 'search intent' to find relevant context from other tokens."
  },
  {
    id: 12,
    question: "Why is the dot product scaled by 1/sqrt(d_k)?",
    type: "mcq",
    options: [
      "To reduce memory usage",
      "To prevent gradients from vanishing due to softmax saturation",
      "To make the calculation faster",
      "To normalize the batch size"
    ],
    correctOptionIndex: 1,
    answer: "To prevent gradients from vanishing due to softmax saturation",
    level: "Intermediate",
    conceptTag: "concept_2",
    explanation: "Large dot products push softmax values to 0 or 1, where gradients are extremely small."
  },
  {
    id: 13,
    question: "Flash Attention optimizes memory access to speed up training.",
    type: "true_false",
    answer: true,
    level: "Advanced",
    conceptTag: "concept_2",
    explanation: "Flash Attention reduces High Bandwidth Memory (HBM) accesses by tiling the computation."
  },
  {
    id: 14,
    question: "Sparse Attention mechanisms reduce the complexity from O(N^2) to something lower (e.g., O(N)).",
    type: "true_false",
    answer: true,
    level: "Advanced",
    conceptTag: "concept_2",
    explanation: "Sparse attention restricts the attention window to reduce the quadratic cost of long sequences."
  },
  {
    id: 15,
    question: "Describe the role of the 'Value' vector in self-attention.",
    type: "free_form",
    keywords: ["content", "information", "retrieve", "weighted", "sum"],
    level: "Intermediate",
    conceptTag: "concept_2",
    explanation: "The Value vector contains the actual information content that is retrieved and aggregated based on the attention weights."
  },

  // --- Multi-Head Attention (concept_3) ---
  {
    id: 16,
    question: "What is the main benefit of Multi-Head Attention?",
    type: "mcq",
    options: [
      "It reduces the number of parameters",
      "It allows the model to focus on different positions/aspects simultaneously",
      "It eliminates the need for positional encodings",
      "It speeds up inference"
    ],
    correctOptionIndex: 1,
    answer: "It allows the model to focus on different positions/aspects simultaneously",
    level: "Beginner",
    conceptTag: "concept_3",
    explanation: "Different heads can learn different relationships (e.g., syntax vs. semantics) in parallel."
  },
  {
    id: 17,
    question: "Grouped Query Attention (GQA) shares Key and Value heads across multiple Query heads.",
    type: "true_false",
    answer: true,
    level: "Advanced",
    conceptTag: "concept_3",
    explanation: "GQA is an interpolation between Multi-Head and Multi-Query attention to improve inference speed."
  },
  {
    id: 18,
    question: "In Multi-Head Attention, the outputs of all heads are averaged together.",
    type: "true_false",
    answer: false,
    level: "Intermediate",
    conceptTag: "concept_3",
    explanation: "The outputs are concatenated and then projected linearly, not averaged."
  },
  {
    id: 19,
    question: "KV-Cache stores the Query vectors for previous tokens during inference.",
    type: "true_false",
    answer: false,
    level: "Advanced",
    conceptTag: "concept_3",
    explanation: "KV-Cache stores Keys and Values, not Queries, because Queries change for each new token being generated."
  },
  {
    id: 20,
    question: "How does Multi-Head Attention help with ambiguity in a sentence?",
    type: "free_form",
    keywords: ["different", "perspectives", "meanings", "heads", "context"],
    level: "Intermediate",
    conceptTag: "concept_3",
    explanation: "It allows different heads to attend to different context words, resolving ambiguity by combining multiple perspectives."
  },

  // --- Feedforward / MLP Block (concept_4) ---
  {
    id: 21,
    question: "The Feed-Forward Network is applied to the whole sequence at once as a single vector.",
    type: "true_false",
    answer: false,
    level: "Beginner",
    conceptTag: "concept_4",
    explanation: "It is applied position-wise, meaning the same network processes each token independently."
  },
  {
    id: 22,
    question: "What is the typical expansion ratio of the hidden dimension in the FFN?",
    type: "mcq",
    options: [
      "2x",
      "4x",
      "8x",
      "Same size"
    ],
    correctOptionIndex: 1,
    answer: "4x",
    level: "Intermediate",
    conceptTag: "concept_4",
    explanation: "In the original paper, d_ff is 2048 while d_model is 512, a 4x expansion."
  },
  {
    id: 23,
    question: "Mixture of Experts (MoE) replaces the dense FFN with a sparse layer of experts.",
    type: "true_false",
    answer: true,
    level: "Advanced",
    conceptTag: "concept_4",
    explanation: "MoE activates only a subset of 'expert' networks for each token to scale capacity without increasing compute."
  },
  {
    id: 24,
    question: "GELU activation is often preferred over ReLU in modern Transformers.",
    type: "true_false",
    answer: true,
    level: "Advanced",
    conceptTag: "concept_4",
    explanation: "GELU (Gaussian Error Linear Unit) provides smoother gradients and is used in GPT-2, BERT, etc."
  },
  {
    id: 25,
    question: "What is the purpose of the non-linearity (ReLU/GELU) in the FFN?",
    type: "free_form",
    keywords: ["complex", "functions", "learn", "linear", "capacity"],
    level: "Intermediate",
    conceptTag: "concept_4",
    explanation: "Without non-linearity, the entire network would collapse into a single linear transformation, limiting its learning power."
  },

  // --- Layer Norm & Residuals (concept_5) ---
  {
    id: 26,
    question: "What is the main benefit of Residual Connections?",
    type: "mcq",
    options: [
      "They compress the data",
      "They allow gradients to flow through the network easily",
      "They act as a regularizer",
      "They calculate the loss"
    ],
    correctOptionIndex: 1,
    answer: "They allow gradients to flow through the network easily",
    level: "Beginner",
    conceptTag: "concept_5",
    explanation: "Residual connections mitigate the vanishing gradient problem by providing a direct path for gradients."
  },
  {
    id: 27,
    question: "In 'Post-LN', Layer Normalization is applied before the sub-layer.",
    type: "true_false",
    answer: false,
    level: "Intermediate",
    conceptTag: "concept_5",
    explanation: "Post-LN applies it after the residual connection. Pre-LN applies it before the sub-layer."
  },
  {
    id: 28,
    question: "DeepSpeed ZeRO-3 partitions model parameters across GPUs.",
    type: "true_false",
    answer: true,
    level: "Advanced",
    conceptTag: "concept_5",
    explanation: "ZeRO-3 partitions optimizer states, gradients, AND parameters to fit large models in memory."
  },
  {
    id: 29,
    question: "Layer Normalization computes statistics over the batch dimension.",
    type: "true_false",
    answer: false,
    level: "Intermediate",
    conceptTag: "concept_5",
    explanation: "Layer Norm computes statistics over the feature dimension for a single sample, unlike Batch Norm."
  },
  {
    id: 30,
    question: "Why is 'Pre-LN' often preferred for training very deep Transformers?",
    type: "free_form",
    keywords: ["stability", "gradients", "training", "deep", "stable"],
    level: "Advanced",
    conceptTag: "concept_5",
    explanation: "Pre-LN improves training stability by keeping the gradient norm consistent across layers, avoiding the need for warm-up."
  }
];
