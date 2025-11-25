export const quizQuestions = [
  // Concept 0: Big Picture
  {
    id: "q1",
    type: "mcq",
    level: "beginner",
    conceptTag: "concept_0",
    question: "What is the main purpose of the Transformer architecture in NLP tasks?",
    options: [
      "To compress images using convolutions",
      "To process sequences using self-attention instead of recurrence",
      "To cluster documents using k-means",
      "To generate random text without training"
    ],
    correctOptionIndex: 1,
    explanation: "Transformers rely on self-attention mechanisms to process sequences in parallel, overcoming the sequential limitations of RNNs."
  },
  {
    id: "q2",
    type: "mcq",
    level: "beginner",
    conceptTag: "concept_0",
    question: "Which of the following best describes a Transformer model?",
    options: [
      "A purely convolutional neural network",
      "A recurrent neural network with LSTM cells",
      "An encoder–decoder model built from attention and feed-forward blocks",
      "A k-nearest neighbors classifier"
    ],
    correctOptionIndex: 2,
    explanation: "The original Transformer consists of an encoder stack and a decoder stack, both built from self-attention and feed-forward layers."
  },
  {
    id: "q3",
    type: "mcq",
    level: "intermediate",
    conceptTag: "concept_0",
    question: "Why are Transformers more parallelizable than RNNs?",
    options: [
      "They only process one token at a time",
      "Self-attention allows all tokens to be processed simultaneously",
      "They use fewer parameters",
      "They do not use GPUs"
    ],
    correctOptionIndex: 1,
    explanation: "Self-attention computes relationships between all tokens at once, allowing for massive parallelism unlike the sequential processing of RNNs."
  },
  {
    id: "q4",
    type: "mcq",
    level: "intermediate",
    conceptTag: "concept_0",
    question: "In the original 'Attention Is All You Need' paper, what is the typical number of encoder and decoder layers in the base model?",
    options: [
      "1 encoder, 1 decoder",
      "3 encoders, 3 decoders",
      "6 encoders, 6 decoders",
      "12 encoders, 12 decoders"
    ],
    correctOptionIndex: 2,
    explanation: "The base model in the original paper used a stack of 6 encoder layers and 6 decoder layers."
  },
  {
    id: "q5",
    type: "mcq",
    level: "advanced",
    conceptTag: "concept_0",
    question: "Which of the following is a key reason Transformers scale well to very large models?",
    options: [
      "Their operations are mostly matrix multiplications that map efficiently to GPUs",
      "They avoid using any normalization layers",
      "They skip non-linear activation functions",
      "They rely on recurrent connections for long-term memory"
    ],
    correctOptionIndex: 0,
    explanation: "The architecture is composed largely of matrix multiplications, which are highly optimized on modern hardware like GPUs and TPUs."
  },

  // Concept 1: Tokenization & Embeddings
  {
    id: "q6",
    type: "mcq",
    level: "beginner",
    conceptTag: "concept_1",
    question: "Why do Transformers use an embedding layer?",
    options: [
      "To convert token IDs into dense continuous vectors",
      "To sort tokens in alphabetical order",
      "To remove stop words",
      "To apply convolution filters"
    ],
    correctOptionIndex: 0,
    explanation: "Embeddings map discrete token IDs to continuous vector representations that capture semantic meaning."
  },
  {
    id: "q7",
    type: "mcq",
    level: "beginner",
    conceptTag: "concept_1",
    question: "What is a common type of tokenizer used before feeding text to a Transformer?",
    options: [
      "Character-level hashing only",
      "Byte Pair Encoding (BPE) or similar subword tokenizers",
      "Bag-of-words without order",
      "One-hot encoding with no compression"
    ],
    correctOptionIndex: 1,
    explanation: "Subword tokenizers like BPE balance vocabulary size and the ability to represent rare words by breaking them down into common sub-units."
  },
  {
    id: "q8",
    type: "mcq",
    level: "intermediate",
    conceptTag: "concept_1",
    question: "What is the shape of the embedding matrix for a vocabulary of size V and model dimension d_model?",
    options: [
      "(d_model, d_model)",
      "(V, d_model)",
      "(V, V)",
      "(d_model, 1)"
    ],
    correctOptionIndex: 1,
    explanation: "The embedding matrix has one row for each token in the vocabulary (V) and d_model columns for the vector dimension."
  },
  {
    id: "q9",
    type: "mcq",
    level: "intermediate",
    conceptTag: "concept_1",
    question: "Why are subword tokenizers like BPE preferred over pure word-level tokenization?",
    options: [
      "They increase the number of out-of-vocabulary tokens",
      "They reduce flexibility in representing rare words",
      "They balance vocabulary size and ability to represent rare or new words",
      "They remove the need for embeddings"
    ],
    correctOptionIndex: 2,
    explanation: "They allow the model to handle rare or unknown words by representing them as sequences of known subword units."
  },
  {
    id: "q10",
    type: "mcq",
    level: "advanced",
    conceptTag: "concept_1",
    question: "What is the main benefit of tying input embeddings with the output projection matrix?",
    options: [
      "It forces the model to use fewer attention heads",
      "It increases the number of parameters dramatically",
      "It reduces parameters and aligns input and output token spaces",
      "It removes the need for positional encoding"
    ],
    correctOptionIndex: 2,
    explanation: "Weight tying reduces the total number of parameters and ensures that the embedding space and the output logit space are consistent."
  },

  // Concept 2: Positional Encoding
  {
    id: "q11",
    type: "mcq",
    level: "beginner",
    conceptTag: "concept_2",
    question: "Why do we add positional encodings to token embeddings?",
    options: [
      "To reduce model size",
      "Because self-attention alone does not encode token order",
      "To convert text to lowercase",
      "To remove punctuation"
    ],
    correctOptionIndex: 1,
    explanation: "Self-attention is permutation invariant, so positional encodings are necessary to provide information about the sequence order."
  },
  {
    id: "q12",
    type: "mcq",
    level: "beginner",
    conceptTag: "concept_2",
    question: "Which of the following is a property of sinusoidal positional encodings?",
    options: [
      "They are random at every step",
      "They are fixed, deterministic functions of position",
      "They are learned independently for each token",
      "They depend on the token identity"
    ],
    correctOptionIndex: 1,
    explanation: "Sinusoidal encodings are generated using fixed sine and cosine functions of different frequencies."
  },
  {
    id: "q13",
    type: "mcq",
    level: "intermediate",
    conceptTag: "concept_2",
    question: "In sinusoidal positional encoding, which of the following is true?",
    options: [
      "Even indices use sine and odd indices use cosine",
      "All indices use only sine",
      "All indices use only cosine",
      "Encodings are constant across positions"
    ],
    correctOptionIndex: 0,
    explanation: "The formula typically uses sine for even dimension indices (2i) and cosine for odd dimension indices (2i+1)."
  },
  {
    id: "q14",
    type: "mcq",
    level: "intermediate",
    conceptTag: "concept_2",
    question: "What is one advantage of sinusoidal positional encodings over learned positional embeddings?",
    options: [
      "They completely eliminate the need for training",
      "They allow extrapolation to sequence lengths not seen during training",
      "They require fewer arithmetic operations",
      "They make the model recurrent"
    ],
    correctOptionIndex: 1,
    explanation: "Because they are mathematical functions, they can theoretically be generated for any position, even beyond the training context length."
  },
  {
    id: "q15",
    type: "mcq",
    level: "advanced",
    conceptTag: "concept_2",
    question: "Why can relative positions be recovered linearly from sinusoidal positional encodings?",
    options: [
      "Because sine and cosine are orthogonal periodic basis functions",
      "Because the encoding is a random projection",
      "Because the encodings are one-hot vectors",
      "Because each position is encoded as a unique integer"
    ],
    correctOptionIndex: 0,
    explanation: "The trigonometric properties allow PE(pos+k) to be expressed as a linear function of PE(pos), facilitating relative position learning."
  },

  // Concept 3: Scaled Dot-Product Self-Attention
  {
    id: "q16",
    type: "mcq",
    level: "beginner",
    conceptTag: "concept_3",
    question: "What does self-attention allow a token to do?",
    options: [
      "Only attend to itself",
      "Attend to all tokens in the sequence, including itself",
      "Attend only to the first token",
      "Attend only to the last token"
    ],
    correctOptionIndex: 1,
    explanation: "Self-attention allows each token to aggregate information from every other token in the sequence to build a context-aware representation."
  },
  {
    id: "q17",
    type: "mcq",
    level: "beginner",
    conceptTag: "concept_3",
    question: "Which vectors are used to compute self-attention scores?",
    options: [
      "Query and value",
      "Key and value",
      "Query and key",
      "Embedding and positional vectors"
    ],
    correctOptionIndex: 2,
    explanation: "The attention score is computed by the dot product of the Query (Q) and Key (K) vectors."
  },
  {
    id: "q18",
    type: "mcq",
    level: "intermediate",
    conceptTag: "concept_3",
    question: "What is the purpose of dividing the dot product of query and key by sqrt(d_k)?",
    options: [
      "To increase the magnitude of scores",
      "To reduce variance and stabilize gradients",
      "To change the sequence length",
      "To normalize the values to sum to 1"
    ],
    correctOptionIndex: 1,
    explanation: "Scaling prevents the dot products from becoming too large, which would push the softmax function into regions with extremely small gradients."
  },
  {
    id: "q19",
    type: "mcq",
    level: "intermediate",
    conceptTag: "concept_3",
    question: "After computing attention scores for a token, what function is applied to convert them into attention weights?",
    options: [
      "Sigmoid",
      "Softmax",
      "ReLU",
      "Tanh"
    ],
    correctOptionIndex: 1,
    explanation: "Softmax normalizes the scores so they are all positive and sum to 1, creating a probability distribution."
  },
  {
    id: "q20",
    type: "mcq",
    level: "advanced",
    conceptTag: "concept_3",
    question: "Which operation produces the final output representation for a token in self-attention?",
    options: [
      "Weighted sum of value vectors",
      "Weighted sum of key vectors",
      "Concatenation of query and key",
      "Dot product between key and value"
    ],
    correctOptionIndex: 0,
    explanation: "The final output is the weighted sum of the Value (V) vectors, where the weights are the attention probabilities."
  },

  // Concept 4: Multi-Head Attention
  {
    id: "q21",
    type: "mcq",
    level: "beginner",
    conceptTag: "concept_4",
    question: "What does 'multi-head' refer to in multi-head attention?",
    options: [
      "Several attention mechanisms run in parallel with different learned projections",
      "Multiple tokenizers process the text",
      "Multiple GPUs compute attention",
      "Multiple models vote on each attention score"
    ],
    correctOptionIndex: 0,
    explanation: "Multi-head attention runs multiple self-attention operations in parallel, each with different learned linear projections."
  },
  {
    id: "q22",
    type: "mcq",
    level: "beginner",
    conceptTag: "concept_4",
    question: "After computing all head outputs, what happens next?",
    options: [
      "Averages are taken",
      "They are concatenated and projected back to d_model",
      "Only the best head is retained",
      "They are ignored"
    ],
    correctOptionIndex: 1,
    explanation: "The outputs from all heads are concatenated and then linearly projected back to the model dimension."
  },
  {
    id: "q23",
    type: "mcq",
    level: "intermediate",
    conceptTag: "concept_4",
    question: "Why do different heads tend to learn different attention patterns?",
    options: [
      "They share all weights",
      "Each head uses different learned projection matrices",
      "They operate on different batches",
      "They are initialized randomly every step"
    ],
    correctOptionIndex: 1,
    explanation: "Because each head has its own unique weight matrices for Q, K, and V, they can specialize in capturing different types of relationships (e.g., syntactic vs. semantic)."
  },
  {
    id: "q24",
    type: "mcq",
    level: "intermediate",
    conceptTag: "concept_4",
    question: "If d_model = 512 and there are 8 heads, what is the dimension per head?",
    options: [
      "8",
      "64",
      "128",
      "512"
    ],
    correctOptionIndex: 1,
    explanation: "Typically d_k = d_model / num_heads, so 512 / 8 = 64."
  },
  {
    id: "q25",
    type: "mcq",
    level: "advanced",
    conceptTag: "concept_4",
    question: "What is the role of the output projection matrix W_O in multi-head attention?",
    options: [
      "It generates positional encodings",
      "It recombines head outputs into a single d_model vector",
      "It normalizes attention weights",
      "It computes Q, K, and V"
    ],
    correctOptionIndex: 1,
    explanation: "W_O mixes the information from the different heads back into a single unified representation."
  },

  // Concept 5: Feed-Forward Networks (FFN)
  {
    id: "q26",
    type: "mcq",
    level: "beginner",
    conceptTag: "concept_5",
    question: "Where is the feed-forward network applied?",
    options: [
      "To each token independently after attention",
      "To the entire sequence jointly",
      "Only before embeddings",
      "Only at the decoder output"
    ],
    correctOptionIndex: 0,
    explanation: "The FFN is applied position-wise, meaning it processes each token's vector independently and identically."
  },
  {
    id: "q27",
    type: "mcq",
    level: "beginner",
    conceptTag: "concept_5",
    question: "What is the structure of the FFN?",
    options: [
      "One linear layer",
      "Linear → Activation → Linear",
      "Convolution → Pooling",
      "Only ReLU activations without linear layers"
    ],
    correctOptionIndex: 1,
    explanation: "It consists of two linear transformations with a non-linear activation function (like ReLU) in between."
  },
  {
    id: "q28",
    type: "mcq",
    level: "intermediate",
    conceptTag: "concept_5",
    question: "If d_model = 512 and d_ff = 2048, what does the larger d_ff dimension provide?",
    options: [
      "Lower capacity",
      "Higher nonlinear expressiveness",
      "Fewer parameters",
      "No practical benefit"
    ],
    correctOptionIndex: 1,
    explanation: "Expanding the dimension allows the model to learn more complex, non-linear functions and interactions within the token representation."
  },
  {
    id: "q29",
    type: "mcq",
    level: "intermediate",
    conceptTag: "concept_5",
    question: "Which activation function was used in the original Transformer?",
    options: [
      "ReLU",
      "GELU",
      "Sigmoid",
      "Tanh"
    ],
    correctOptionIndex: 0,
    explanation: "The original paper used the Rectified Linear Unit (ReLU) activation function."
  },
  {
    id: "q30",
    type: "mcq",
    level: "advanced",
    conceptTag: "concept_5",
    question: "Why is the FFN called 'position-wise'?",
    options: [
      "Different FFN weights for each position",
      "Same weights applied independently to each token position",
      "It updates positional encodings",
      "It computes the causal mask"
    ],
    correctOptionIndex: 1,
    explanation: "Because the exact same parameters (weights and biases) are used for every position in the sequence."
  },

  // Concept 6: Add & Norm
  {
    id: "q31",
    type: "mcq",
    level: "beginner",
    conceptTag: "concept_6",
    question: "What is the purpose of the residual connection in a Transformer layer?",
    options: [
      "To normalize gradients",
      "To combine layer input with its output",
      "To reduce sequence length",
      "To remove positional encodings"
    ],
    correctOptionIndex: 1,
    explanation: "Residual connections add the input of a sub-layer to its output (x + SubLayer(x)), helping to preserve information."
  },
  {
    id: "q32",
    type: "mcq",
    level: "beginner",
    conceptTag: "concept_6",
    question: "What does Layer Normalization normalize?",
    options: [
      "Across the batch dimension",
      "Across feature dimensions for each token",
      "Across all tokens simultaneously",
      "Across labels"
    ],
    correctOptionIndex: 1,
    explanation: "LayerNorm normalizes the features of a single token, making it independent of the batch size."
  },
  {
    id: "q33",
    type: "mcq",
    level: "intermediate",
    conceptTag: "concept_6",
    question: "Why is Add & Norm applied after both attention and FFN blocks?",
    options: [
      "To add regularization",
      "To stabilize training and preserve information flow",
      "To compute QKV",
      "To create embeddings"
    ],
    correctOptionIndex: 1,
    explanation: "It stabilizes the training of deep networks and ensures gradients can flow through the network effectively."
  },
  {
    id: "q34",
    type: "mcq",
    level: "intermediate",
    conceptTag: "concept_6",
    question: "Which parameters are learned in LayerNorm?",
    options: [
      "Weight and bias (gamma and beta)",
      "Only bias",
      "Only mean",
      "None"
    ],
    correctOptionIndex: 0,
    explanation: "LayerNorm learns a scale parameter (gamma) and a shift parameter (beta) to restore the representation power."
  },
  {
    id: "q35",
    type: "mcq",
    level: "advanced",
    conceptTag: "concept_6",
    question: "What problem do residual connections help mitigate?",
    options: [
      "Exploding sequence lengths",
      "Vanishing gradients in deep models",
      "Token misalignment",
      "Vocabulary explosion"
    ],
    correctOptionIndex: 1,
    explanation: "By providing a direct path for gradients to flow backwards, they prevent the gradients from vanishing in very deep networks."
  },

  // Concept 7: Encoder Block
  {
    id: "q36",
    type: "mcq",
    level: "beginner",
    conceptTag: "concept_7",
    question: "What sublayers are inside an encoder block?",
    options: [
      "Self-attention and FFN",
      "Masked attention only",
      "Convolution and pooling",
      "GRU only"
    ],
    correctOptionIndex: 0,
    explanation: "An encoder block consists of a Multi-Head Self-Attention layer followed by a Feed-Forward Network."
  },
  {
    id: "q37",
    type: "mcq",
    level: "intermediate",
    conceptTag: "concept_7",
    question: "What does the encoder output?",
    options: [
      "A single vector",
      "Context-enhanced token representations",
      "Only positional encodings",
      "Only token IDs"
    ],
    correctOptionIndex: 1,
    explanation: "The encoder transforms the input embeddings into a sequence of vectors where each vector contains context from the entire sequence."
  },
  {
    id: "q38",
    type: "mcq",
    level: "intermediate",
    conceptTag: "concept_7",
    question: "How does stacking encoder layers improve representation quality?",
    options: [
      "By removing token dependencies",
      "By enabling deeper contextual transformations",
      "By increasing vocabulary size",
      "By reducing d_model"
    ],
    correctOptionIndex: 1,
    explanation: "Stacking layers allows the model to learn increasingly complex and abstract relationships between tokens."
  },
  {
    id: "q39",
    type: "mcq",
    level: "advanced",
    conceptTag: "concept_7",
    question: "In encoder self-attention, what can each token attend to?",
    options: [
      "Only previous tokens",
      "Only future tokens",
      "Any token including itself",
      "No tokens"
    ],
    correctOptionIndex: 2,
    explanation: "In the encoder, attention is unmasked (bidirectional), so every token can attend to every other token in the sequence."
  },
  {
    id: "q40",
    type: "mcq",
    level: "advanced",
    conceptTag: "concept_7",
    question: "How is encoder output used by the decoder?",
    options: [
      "As input embeddings",
      "As keys and values for cross-attention",
      "As positional encodings",
      "As the final softmax output"
    ],
    correctOptionIndex: 1,
    explanation: "The encoder output serves as the memory (Keys and Values) that the decoder queries in the cross-attention layers."
  },

  // Concept 8: Cross-Attention
  {
    id: "q41",
    type: "mcq",
    level: "beginner",
    conceptTag: "concept_8",
    question: "What is the main purpose of cross-attention?",
    options: [
      "To attend within the target sequence",
      "To attend from the decoder to encoder outputs",
      "To generate positional encodings",
      "To compute token embeddings"
    ],
    correctOptionIndex: 1,
    explanation: "Cross-attention bridges the encoder and decoder, allowing the decoder to focus on relevant parts of the input sequence."
  },
  {
    id: "q42",
    type: "mcq",
    level: "beginner",
    conceptTag: "concept_8",
    question: "In cross-attention, which components come from the encoder?",
    options: [
      "Query only",
      "Key and value",
      "Query, key, and value",
      "Positional encoding only"
    ],
    correctOptionIndex: 1,
    explanation: "The Keys (K) and Values (V) come from the encoder's output, representing the source information."
  },
  {
    id: "q43",
    type: "mcq",
    level: "intermediate",
    conceptTag: "concept_8",
    question: "What does the decoder contribute to cross-attention?",
    options: [
      "Queries",
      "Keys and values",
      "Only positional encodings",
      "Random noise vectors"
    ],
    correctOptionIndex: 0,
    explanation: "The decoder provides the Queries (Q), representing what it is currently looking for in the source."
  },
  {
    id: "q44",
    type: "mcq",
    level: "intermediate",
    conceptTag: "concept_8",
    question: "What does cross-attention allow the decoder to do?",
    options: [
      "Attend to encoder representations and incorporate source information",
      "Modify the encoder outputs directly",
      "Replace tokenization with attention weights",
      "Compute model parameters"
    ],
    correctOptionIndex: 0,
    explanation: "It allows the generation process to be conditioned on the input sequence, which is crucial for tasks like translation."
  },
  {
    id: "q45",
    type: "mcq",
    level: "advanced",
    conceptTag: "concept_8",
    question: "Which is the correct flow of Q, K, V in cross-attention?",
    options: [
      "Q from encoder, K/V from decoder",
      "Q from decoder, K/V from encoder",
      "Q/K/V all from decoder",
      "Q/K/V all from encoder"
    ],
    correctOptionIndex: 1,
    explanation: "Queries come from the target sequence (decoder), while Keys and Values come from the source sequence (encoder)."
  },

  // Concept 9: Decoder Block
  {
    id: "q46",
    type: "mcq",
    level: "beginner",
    conceptTag: "concept_9",
    question: "What sublayers exist in a decoder block?",
    options: [
      "Masked self-attention, cross-attention, FFN",
      "Self-attention and RNN",
      "Only FFN",
      "Only cross-attention"
    ],
    correctOptionIndex: 0,
    explanation: "A decoder block has three sub-layers: Masked Self-Attention, Cross-Attention, and FFN."
  },
  {
    id: "q47",
    type: "mcq",
    level: "beginner",
    conceptTag: "concept_9",
    question: "Why does a decoder use masked self-attention?",
    options: [
      "To prevent attending to future tokens",
      "To remove embeddings",
      "To ignore encoder outputs",
      "To reduce model size"
    ],
    correctOptionIndex: 0,
    explanation: "Masking prevents the model from 'cheating' by looking at tokens it hasn't generated yet during training."
  },
  {
    id: "q48",
    type: "mcq",
    level: "intermediate",
    conceptTag: "concept_9",
    question: "What is the role of cross-attention?",
    options: [
      "Ignore encoder outputs",
      "Attend to encoder outputs using decoder queries",
      "Normalize logits",
      "Compute positional encoding"
    ],
    correctOptionIndex: 1,
    explanation: "Cross-attention integrates the source information (from encoder) into the target generation process."
  },
  {
    id: "q49",
    type: "mcq",
    level: "intermediate",
    conceptTag: "concept_9",
    question: "During training with teacher forcing, what is fed to the decoder?",
    options: [
      "The model's predicted token",
      "The ground-truth previous token",
      "A random token",
      "The encoder’s positional encoding"
    ],
    correctOptionIndex: 1,
    explanation: "Teacher forcing feeds the actual correct previous token (ground truth) to the decoder, regardless of what the model predicted."
  },
  {
    id: "q50",
    type: "mcq",
    level: "advanced",
    conceptTag: "concept_9",
    question: "Which attention sublayer allows the decoder to align with the source sequence?",
    options: [
      "Masked self-attention",
      "Cross-attention",
      "Output softmax",
      "FFN"
    ],
    correctOptionIndex: 1,
    explanation: "Cross-attention is specifically designed for alignment between the source and target sequences."
  },

  // BERT Questions
  // BERT Big Picture
  {
    id: "bert_q1",
    type: "mcq",
    level: "intermediate",
    conceptTag: "bert_0",
    question: "What is the main difference between BERT and the original Transformer?",
    options: [
      "BERT uses only the Decoder",
      "BERT uses only the Encoder",
      "BERT uses both Encoder and Decoder",
      "BERT uses RNNs"
    ],
    correctOptionIndex: 1,
    explanation: "BERT is a stack of Transformer Encoders, designed to understand language bidirectionally, unlike the original Transformer which has both Encoder and Decoder for translation."
  },
  {
    id: "bert_q2",
    type: "true_false",
    level: "intermediate",
    conceptTag: "bert_0",
    statement: "BERT reads text sequentially from left to right, just like GPT.",
    answer: false,
    explanation: "False. BERT is bidirectional, meaning it attends to the entire sequence (left and right context) simultaneously."
  },

  // Masked Language Modeling
  {
    id: "bert_q3",
    type: "mcq",
    level: "advanced",
    conceptTag: "bert_1",
    conceptTag: "bert_1",
    question: "In Masked Language Modeling (MLM), what percentage of tokens are typically masked?",
    options: [
      "10%",
      "15%",
      "50%",
      "100%"
    ],
    correctOptionIndex: 1,
    explanation: "BERT typically masks 15% of the input tokens for prediction during pre-training."
  },
  {
    id: "bert_q4",
    type: "free_form",
    level: "advanced",
    conceptTag: "bert_1",
    question: "Why does BERT replace some masked tokens with random words instead of just [MASK]?",
    keywords: ["mismatch", "finetuning", "fine-tuning", "adapt", "real"],
    explanation: "To mitigate the mismatch between pre-training (where [MASK] appears) and fine-tuning (where it doesn't), so the model learns to handle real words too."
  },

  // Next Sentence Prediction
  {
    id: "bert_q5",
    type: "true_false",
    level: "intermediate",
    conceptTag: "bert_2",
    statement: "Next Sentence Prediction (NSP) is used to teach BERT relationships between sentences.",
    answer: true,
    explanation: "True. NSP helps BERT understand long-range dependencies and relationships between pairs of sentences."
  },

  // Fine-tuning
  {
    id: "bert_q6",
    type: "mcq",
    level: "intermediate",
    conceptTag: "bert_3",
    question: "What is the main advantage of fine-tuning BERT?",
    options: [
      "It requires training from scratch",
      "It allows using a pre-trained model for specific tasks with little data",
      "It makes the model smaller",
      "It removes the need for a GPU"
    ],
    correctOptionIndex: 1,
    explanation: "Fine-tuning leverages the massive knowledge BERT learned during pre-training, allowing it to achieve state-of-the-art results on specific tasks with relatively small datasets."
  },

  // CNN Questions
  {
    id: "cnn_q1",
    type: "mcq",
    level: "beginner",
    conceptTag: "cnn_0",
    question: "What is the primary goal of CNN?",
    options: [
      "Option A",
      "Option B",
      "Option C",
      "Option D"
    ],
    correctOptionIndex: 0,
    explanation: "Placeholder explanation for CNN."
  },


  // Latent Diffusion Questions
  {
    id: "latent_q1",
    type: "mcq",
    level: "advanced",
    conceptTag: "latent_0",
    question: "What is the primary goal of Latent Diffusion?",
    options: [
      "Option A",
      "Option B",
      "Option C",
      "Option D"
    ],
    correctOptionIndex: 0,
    explanation: "Placeholder explanation for Latent Diffusion."
  },
];
