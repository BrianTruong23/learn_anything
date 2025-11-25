export const concepts = [
  {
    id: "concept_0",
    title: "Big Picture",
    architectureImage: "/transformers.png",
    subConcepts: [
      { id: "sub_0_1", label: "Parallel Processing" },
      { id: "sub_0_2", label: "Self-Attention Mechanism" },
      { id: "sub_0_3", label: "Encoder-Decoder Structure" },
      { id: "sub_0_4", label: "Why Transformers Work" }
    ],
    explanations: {
      beginner: {
        motivation: "Traditional AI models read text one word at a time, like reading with a narrow flashlight in the dark. If a sentence is very long, they might forget the beginning by the time they reach the end. We need a better way to understand language where the model can see the whole picture at once.",
        definition: "Transformers are like having the entire sentence lit up at once. Instead of reading word by word, they look at ALL words simultaneously and figure out which words are most important to each other. This is done through a mechanism called 'Self-Attention' that lets each word 'pay attention' to relevant words anywhere in the sentence.",
        toyExample: {
          description: "Let's see how a Transformer processes a sentence:",
          steps: [
            "📝 Input: 'The cat sat on the mat because it was comfortable'",
            "🔍 The Transformer looks at ALL words at the same time (not one by one)",
            "🎯 For the word 'it', it calculates how related 'it' is to every other word",
            "✨ It discovers that 'it' is most related to 'mat' (not 'cat')",
            "💡 This happens in parallel for ALL words simultaneously!"
          ]
        }
      },
      intermediate: {
        motivation: "Recurrent Neural Networks (RNNs) process sequences sequentially, which creates two major problems: (1) training is slow because each step depends on the previous one, and (2) information from distant tokens gets degraded as it passes through many recurrent steps. The Transformer architecture was designed to solve both issues.",
        definition: "A Transformer is an encoder-decoder architecture that replaces recurrence with self-attention. The encoder processes the entire input sequence in parallel, computing relationships between all token pairs simultaneously. Each layer consists of multi-head self-attention followed by position-wise feed-forward networks, with residual connections and layer normalization around each sub-layer.",
        equations: [
          {
            latex: "\\text{Attention}(Q, K, V) = \\text{softmax}\\left(\\frac{QK^\\top}{\\sqrt{d_k}}\\right)V",
            explanation: "This is the core attention mechanism that computes weighted combinations of values based on query-key similarities.",
            terms: {
              "Q": "Query matrix (what we're looking for)",
              "K": "Key matrix (what we're matching against)",
              "V": "Value matrix (what we want to retrieve)",
              "d_k": "Dimension of key vectors (typically 64)"
            },
            intuition: "Think of it like a search engine: Q are your search queries, K are document titles, and V are the actual documents. The softmax ensures we get a weighted average where weights sum to 1."
          }
        ],
        toyExample: {
          description: "Understanding the parallel processing advantage:",
          steps: [
            "🔄 RNN approach: Process 'The' → 'cat' → 'sat' → 'on' → 'the' → 'mat' (sequential, slow)",
            "⚡ Transformer approach: Process ALL tokens ['The', 'cat', 'sat', 'on', 'the', 'mat'] at once",
            "📊 Compute attention: Each token attends to all others, creating a 6×6 attention matrix",
            "🎯 Result: 'sat' can directly see both 'cat' (subject) and 'mat' (location) without sequential steps",
            "⏱️ Training time: Much faster due to parallelization across GPU cores"
          ]
        }
      },
      advanced: {
        motivation: "The theoretical limitation of RNNs is their \\( O(n) \\) sequential complexity, making them fundamentally unable to leverage modern parallel hardware. Additionally, the path length between distant positions grows linearly, limiting gradient flow. Transformers achieve \\( O(1) \\) sequential operations and constant-length paths between any positions through self-attention, at the cost of \\( O(n^2) \\) memory complexity.",
        definition: "The Transformer implements Scaled Dot-Product Attention: \\[ \\text{Attention}(Q,K,V) = \\text{softmax}\\left(\\frac{QK^\\top}{\\sqrt{d_k}}\\right)V \\] where queries, keys, and values are learned linear projections of the input. Multi-head attention applies this mechanism \\( h \\) times in parallel with different learned projections, allowing the model to jointly attend to information from different representation subspaces. The architecture uses sinusoidal positional encodings to inject sequence order information.",
        toyExample: {
          description: "Mathematical flow of self-attention computation:",
          steps: [
            "📐 Input embedding: x ∈ R^(n×d_model), where n=sequence length, d_model=512",
            "🔄 Linear projections: Q = xW^Q, K = xW^K, V = xW^V (W ∈ R^(d_model×d_k))",
            "🧮 Attention scores: S = QK^T ∈ R^(n×n), each cell S[i,j] measures similarity",
            "⚖️ Scaling: S = S/√d_k to prevent softmax saturation in high dimensions",
            "🎲 Softmax: α = softmax(S) row-wise, ensures attention weights sum to 1",
            "✅ Output: Z = αV ∈ R^(n×d_k), weighted combination of values"
          ]
        }
      }
    }
  },
  {
    id: "concept_1",
    title: "Positional Embeddings",

    subConcepts: [
      { id: "sub_1_1", label: "Sine/Cosine Formula" },
      { id: "sub_1_2", label: "Index to Angle Mapping" },
      { id: "sub_1_3", label: "Why It Helps Attention" },
      { id: "sub_1_4", label: "Relative Position Learning" }
    ],
    explanations: {
      beginner: {
        motivation: "Imagine shuffling the words in a sentence: 'dog the chased cat the' vs 'the cat chased the dog'. The meaning completely changes! But if a Transformer looks at all words at once, how does it know which word came first? Without position information, both sentences would look identical to the model.",
        definition: "Positional embeddings are like adding a number tag to each word that says 'I am word #1', 'I am word #2', etc. These position tags are special vectors that get added to the word's meaning, so the model knows both WHAT each word is AND WHERE it appears in the sentence.",
        toyExample: {
          description: "Seeing positions added to words:",
          steps: [
            "📝 Sentence: 'The cat chased the dog'",
            "🔢 Assign positions: The[0], cat[1], chased[2], the[3], dog[4]",
            "➕ Add position info: 'The' = word_embedding('The') + position_embedding(0)",
            "🔄 Now swap: 'The dog chased the cat' → positions change!",
            "✅ Model can now tell the difference: dog[1] vs dog[4] have different representations"
          ]
        }
      },
      intermediate: {
        motivation: "Self-attention is permutation-invariant: swapping the order of input tokens doesn't change the attention computation (it's based on content similarity, not position). Without explicit position information, 'John loves Mary' and 'Mary loves John' would produce identical representations, which is unacceptable for language understanding.",
        definition: "Positional encoding adds position-dependent signals to token embeddings. The original Transformer uses sinusoidal functions: \\[ PE_{(pos,2i)} = \\sin\\left(\\frac{pos}{10000^{2i/d_{\\text{model}}}}\\right) \\] and \\[ PE_{(pos,2i+1)} = \\cos\\left(\\frac{pos}{10000^{2i/d_{\\text{model}}}}\\right) \\] These are added to the input embeddings before the first layer, providing the model with explicit information about token positions.",
        equations: [
          {
            latex: "PE_{(pos,2i)} = \\sin\\left(\\frac{pos}{10000^{2i/d_{\\text{model}}}}\\right)",
            explanation: "Positional encoding for even dimensions uses sine waves with different frequencies.",
            terms: {
              "pos": "Position in the sequence (0, 1, 2, ...)",
              "i": "Dimension index (0 to d_model/2)",
              "d_{\\text{model}}": "Model dimension (typically 512)"
            },
            intuition: "Each position gets a unique 'fingerprint' made of sine waves. Low dimensions change quickly (high frequency), high dimensions change slowly (low frequency), creating a unique pattern for each position."
          },
          {
            latex: "PE_{(pos,2i+1)} = \\cos\\left(\\frac{pos}{10000^{2i/d_{\\text{model}}}}\\right)",
            explanation: "Positional encoding for odd dimensions uses cosine waves, paired with the sine waves.",
            terms: {
              "pos": "Position in the sequence",
              "i": "Dimension index",
              "d_{\\text{model}}": "Model dimension"
            },
            intuition: "Using both sin and cos allows the model to easily learn to attend to relative positions, since PE(pos+k) can be expressed as a linear function of PE(pos)."
          }
        ],
        toyExample: {
          description: "How sinusoidal encoding works:",
          steps: [
            "📊 For position pos=5, dimension d_model=512, compute 256 sin/cos pairs",
            "🌊 Low dimensions (i=0): sin(5/10000^0) = sin(5) → high frequency, changes quickly",
            "🌊 High dimensions (i=255): sin(5/10000^1) ≈ sin(0.0005) → low frequency, changes slowly",
            "🎯 This creates a unique 'fingerprint' for each position",
            "✨ Benefit: Model can learn to attend to relative positions (e.g., 'word 3 positions back')"
          ]
        }
      },
      advanced: {
        motivation: "The choice of positional encoding affects the model's ability to generalize to sequence lengths beyond those seen during training. Fixed sinusoidal encodings provide this extrapolation capability, while learned positional embeddings (like in BERT) may not generalize well to longer sequences. The sinusoidal pattern also enables the model to learn relative position attention through linear transformations.",
        definition: "Sinusoidal positional encoding maps each position to a \\( d_{\\text{model}} \\)-dimensional vector using wavelengths forming a geometric progression from \\( 2\\pi \\) to \\( 10000 \\cdot 2\\pi \\). Formally, \\( PE \\in \\mathbb{R}^{\\text{max\\_len} \\times d_{\\text{model}}} \\) where: \\[ PE_{(pos,2i)} = \\sin\\left(\\frac{pos}{10000^{2i/d_{\\text{model}}}}\\right) \\] and \\[ PE_{(pos,2i+1)} = \\cos\\left(\\frac{pos}{10000^{2i/d_{\\text{model}}}}\\right) \\] This encoding is added element-wise to token embeddings: \\( x' = x + PE[pos] \\).",
        toyExample: {
          description: "Mathematical properties of sinusoidal encoding:",
          steps: [
            "📐 For any fixed offset k, PE[pos+k] can be represented as a linear function of PE[pos]",
            "🔄 This is because sin(α+β) = sin(α)cos(β) + cos(α)sin(β) [angle addition formula]",
            "🧮 PE[pos+k,2i] = sin((pos+k)/w) = sin(pos/w)cos(k/w) + cos(pos/w)sin(k/w)",
            "📊 This equals a linear combination of PE[pos,2i] and PE[pos,2i+1]",
            "✅ The model can learn to attend to relative positions via learned linear transformations",
            "🎯 Example: To attend to 'the previous word', model learns weights to compute PEpos-1 from PEpos"
          ]
        }
      }
    }
  },
  {
    id: "concept_2",
    title: "Self-Attention",

    subConcepts: [
      { id: "sub_2_1", label: "Query, Key, Value" },
      { id: "sub_2_2", label: "Attention Scores" },
      { id: "sub_2_3", label: "Softmax Normalization" },
      { id: "sub_2_4", label: "Scaling Factor√d_k" }
    ],
    explanations: {
      beginner: {
        motivation: "When you read 'The animal didn't cross the street because it was too tired', you automatically know 'it' refers to 'animal'. But how? Your brain looks back at the sentence and connects 'it' to 'animal'. The model needs a similar ability to link words together based on their meaning and context.",
        definition: "Self-attention lets each word 'look at' all other words in the sentence and decide which ones are most important for understanding it. For the word 'it', the model computes a score showing how related 'it' is to 'animal', 'street', 'cross', etc., then uses these scores to create a better representation of 'it' that incorporates information from 'animal'.",
        toyExample: {
          description: "Watch self-attention connect words:",
          steps: [
            "📝 Sentence: 'The animal didn't cross the street because it was too tired'",
            "🎯 Focus on the word: 'it'",
            "🔍 Compute relationship scores: animal=0.8, street=0.1, cross=0.05, tired=0.05",
            "📊 'it' pays 80% attention to 'animal', 10% to 'street', etc.",
            "✨ New representation of 'it' = 0.8×(meaning of 'animal') + 0.1×(meaning of 'street') + ...",
            "💡 Result: 'it' now contains information that helps identify the referent as 'animal'"
          ]
        }
      },
      intermediate: {
        motivation: "Fixed-window approaches (like CNNs) can only relate tokens within a local neighborhood, requiring many layers to connect distant tokens. Self-attention allows any token to directly interact with any other token in a single layer, providing direct access to the full context. This is crucial for capturing long-range dependencies like coreference resolution.",
        definition: "Self-attention computes three vectors for each token: Query (Q), Key (K), and Value (V) through learned linear transformations. The attention weight between token \\( i \\) and \\( j \\) is computed as \\( \\text{softmax}(Q_i \\cdot K_j / \\sqrt{d_k}) \\), measuring the similarity between \\( i \\)'s query and \\( j \\)'s key. The output for token \\( i \\) is a weighted sum of all value vectors: \\[ \\text{output}_i = \\sum_j \\text{attention}(i,j) \\times V_j \\]",
        equations: [
          {
            latex: "\\text{Attention}(Q, K, V) = \\text{softmax}\\left(\\frac{QK^\\top}{\\sqrt{d_k}}\\right)V",
            explanation: "The complete self-attention mechanism that computes context-aware representations.",
            terms: {
              "Q": "Query matrix (n × d_k), where n is sequence length",
              "K": "Key matrix (n × d_k)",
              "V": "Value matrix (n × d_k)",
              "\\sqrt{d_k}": "Scaling factor to prevent large dot products"
            },
            intuition: "For each word (query), we compute how much it should attend to every other word (keys), creating attention weights via softmax. Then we take a weighted combination of the values."
          },
          {
            latex: "Q = XW^Q, \\quad K = XW^K, \\quad V = XW^V",
            explanation: "Linear transformations that project input X into query, key, and value spaces.",
            terms: {
              "X": "Input embeddings (n × d_model)",
              "W^Q, W^K, W^V": "Learned projection matrices (d_model × d_k)"
            },
            intuition: "We don't use the raw word embeddings directly. Instead, we learn to project them into specialized representations optimized for matching (Q,K) and retrieval (V)."
          }
        ],
        toyExample: {
          description: "Step-by-step attention computation:",
          steps: [
            "📥 Input: token embeddings [x₁, x₂, ..., xₙ] where each xᵢ ∈ R^d_model",
            "🔄 Compute Q, K, V: Qᵢ = xᵢW^Q, Kᵢ = xᵢW^K, Vᵢ = xᵢW^V (learned matrices)",
            "🧮 For token i=5 ('it'), compute scores: s₅,ⱼ = Q₅ · Kⱼ for all j",
            "⚖️ Scale: s₅,ⱼ = s₅,ⱼ / √d_k (prevents large magnitudes)",
            "🎲 Normalize: α₅,ⱼ = softmax(s₅,₁, s₅,₂, ..., s₅,ₙ) → attention weights sum to 1",
            "✅ Output: z₅ = Σⱼ α₅,ⱼ × Vⱼ (weighted sum of all value vectors)"
          ]
        }
      },
      advanced: {
        motivation: "The attention mechanism must balance expressiveness with computational efficiency. The scaled dot-product formulation enables efficient batch matrix multiplication on GPUs, while the scaling factor \\( \\sqrt{d_k} \\) prevents the dot products from growing large in magnitude (which would push the softmax into regions with extremely small gradients). The Query-Key-Value decomposition provides more flexibility than a simple similarity measure.",
        definition: "Scaled Dot-Product Attention: \\[ \\text{Attention}(Q,K,V) = \\text{softmax}\\left(\\frac{QK^\\top}{\\sqrt{d_k}}\\right)V \\] where \\( Q,K,V \\in \\mathbb{R}^{n \\times d_k} \\). The scaling factor \\( \\sqrt{d_k} \\) compensates for the variance of the dot product (\\( \\text{Var}[q \\cdot k] = d_k \\) if \\( q,k \\) are i.i.d. standard normal). The softmax operates row-wise, ensuring each query's attention distribution sums to 1. Complexity is \\( O(n^2 d_k) \\) in time and \\( O(n^2) \\) in space for the attention matrix.",
        toyExample: {
          description: "Deriving the scaling factor necessity:",
          steps: [
            "🎲 Assume Q, K entries are i.i.d. ~ N(0,1)",
            "🧮 For q,k ∈ R^d_k, the dot product q·k = Σᵢ qᵢkᵢ",
            "📊 E[q·k] = 0 (sum of zero-mean random variables)",
            "📈 Var[q·k] = Σᵢ Var[qᵢkᵢ] = Σᵢ E[qᵢ²]E[kᵢ²] = d_k",
            "⚠️ For large d_k (e.g., 64), q·k has std dev = √64 = 8, creating very large values",
            "🎯 Softmax of [80, -70, 85] ≈ [0.47, 0, 0.53] → gradients near 0 for middle element",
            "✅ Scaling by 1/√d_k reduces variance to 1, keeping softmax in a sensitive region"
          ]
        }
      }
    }
  },
  {
    id: "concept_3",
    title: "Multi-Head Attention",

    subConcepts: [
      { id: "sub_3_1", label: "Multiple Attention Heads" },
      { id: "sub_3_2", label: "Head Specialization" },
      { id: "sub_3_3", label: "Concatenation" },
      { id: "sub_3_4", label: "Output Projection" }
    ],
    explanations: {
      beginner: {
        motivation: "When reading a sentence, you pay attention to different things: the grammar structure, the meaning of words, and who is talking to whom. Using just one attention mechanism is like looking through one lens. We want the model to look at the sentence from multiple perspectives simultaneously.",
        definition: "Multi-head attention runs several attention mechanisms in parallel, like having multiple pairs of eyes looking at the same sentence but focusing on different aspects. One 'head' might focus on grammar (like connecting verbs to subjects), another on meaning (like synonyms), and another on references (like pronouns to nouns). Then all these perspectives are combined together.",
        toyExample: {
          description: "Multiple attention heads working together:",
          steps: [
            "📝 Sentence: 'The quick brown fox jumps over the lazy dog'",
            "👁️ Head 1 (syntax): Connects 'fox' ← 'quick', 'fox' ← 'brown' (adjectives to noun)",
            "👁️ Head 2 (action): Connects 'jumps' ← 'fox' (subject to verb)",
            "👁️ Head 3 (location): Connects 'jumps' → 'over' → 'dog' (action path)",
            "🔀 Combine: Each word now has a richer representation from all 3 perspectives",
            "✨ 'fox' knows: it's quick and brown (head 1), it performs jumping (head 2), it relates to 'over dog' (head 3)"
          ]
        }
      },
      intermediate: {
        motivation: "A single attention mechanism with \\( d_{\\text{model}} \\)-dimensional Q,K,V may learn to represent one dominant pattern but struggle to capture diverse linguistic phenomena simultaneously (e.g., syntactic vs. semantic dependencies). Multi-head attention allows the model to jointly attend to information from different representation subspaces, capturing multiple types of relationships in parallel.",
        definition: "Multi-Head Attention runs \\( h \\) parallel attention operations (heads), each with its own learned Q,K,V projection matrices of dimension \\( d_k = d_{\\text{model}}/h \\). After computing \\( h \\) attention outputs, they are concatenated and linearly projected: \\[ \\text{MultiHead}(Q,K,V) = \\text{Concat}(\\text{head}_1,...,\\text{head}_h)W^O \\] where \\( \\text{head}_i = \\text{Attention}(QW_i^Q, KW_i^K, VW_i^V) \\). Typical values: \\( h=8, d_{\\text{model}}=512, d_k=d_v=64 \\).",
        toyExample: {
          description: "Multi-head computation flow:",
          steps: [
            "📥 Input: X ∈ R^(n×512), using h=8 heads",
            "🔄 For each head i ∈ {1..8}: compute Qᵢ = XWᵢ^Q, Kᵢ = XWᵢ^K, Vᵢ = XWᵢ^V (each ∈ R^(n×64))",
            "🧮 For each head: compute headᵢ = Attention(Qᵢ,Kᵢ,Vᵢ) ∈ R^(n×64)",
            "🔗 Concatenate: [head₁ | head₂ | ... | head₈] ∈ R^(n×512)",
            "📊 Final projection: output = Concat(heads)W^O where W^O ∈ R^(512×512)",
            "✅ Each head learns different attention patterns over the same input"
          ]
        }
      },
      advanced: {
        motivation: "The theoretical motivation is representation learning in multiple subspaces. By splitting \\( d_{\\text{model}} \\) dimensions into \\( h \\) groups, we enable the model to learn \\( h \\) distinct attention patterns simultaneously without increasing the parameter count significantly (\\( h \\) sets of smaller matrices vs. one large matrix). Empirically, different heads learn to specialize: some capture positional patterns, others semantic similarities, and others syntactic dependencies.",
        definition: "Formally, \\[ \\text{MultiHead}(Q,K,V) = \\text{Concat}(\\text{head}_1,...,\\text{head}_h)W^O \\] where \\( \\text{head}_i = \\text{Attention}(QW_i^Q,KW_i^K,VW_i^V) \\) and \\( W_i^Q,W_i^K \\in \\mathbb{R}^{d_{\\text{model}} \\times d_k} \\), \\( W_i^V \\in \\mathbb{R}^{d_{\\text{model}} \\times d_v} \\), \\( W^O \\in \\mathbb{R}^{hd_v \\times d_{\\text{model}}} \\). Common choice: \\( d_k = d_v = d_{\\text{model}}/h \\), maintaining total computational cost similar to single-head with full dimensions. Total parameters: \\( h(2d_{\\text{model}} d_k + d_{\\text{model}} d_v) + hd_v d_{\\text{model}} \\).",
        toyExample: {
          description: "Analyzing learned head specialization (from research):",
          steps: [
            "🔬 Analysis of trained Transformer heads shows distinct patterns:",
            "📍 Head 1: Attends to previous token (position offset -1), learning sequential patterns",
            "📍 Head 3: Attends to matching delimiter pairs (parentheses, quotes)",
            "📍 Head 5: Broad attention for semantic similarity (high entropy distribution)",
            "📍 Head 7: Sharp attention to specific tokens (low entropy), captures coreference",
            "🧮 Computational efficiency: 8 heads with d_k=64 vs. 1 head with d_k=512:",
            "   8×(n²·64) = n²·512 vs. n²·512 → same cost but 8× representation capacity"
          ]
        }
      }
    }
  },
  {
    id: "concept_4",
    title: "Feedforward / MLP Block",

    subConcepts: [
      { id: "sub_4_1", label: "Two-Layer Network" },
      { id: "sub_4_2", label: "ReLU Activation" },
      { id: "sub_4_3", label: "Dimension Expansion" },
      { id: "sub_4_4", label: "Position-wise Application" }
    ],
    explanations: {
      beginner: {
        motivation: "After attention gathers information from different words, the model needs to actually 'think' about this information. Attention is like collecting ingredients for a recipe, but we still need to cook! The feedforward network is where the model processes and transforms the gathered information into something useful.",
        definition: "The feedforward block is a simple two-layer neural network that looks at each word independently (the same network applied to every word). It's like a mini-brain that transforms the representation: first it expands the information (making it bigger), applies a non-linear function (ReLU) to add complexity, then compresses it back down. Think of it as expand → process → compress.",
        toyExample: {
          description: "Watching the feedforward transformation:",
          steps: [
            "📥 Input: word representation with 512 numbers",
            "📈 Expand: multiply by a matrix → now we have 2048 numbers (4× bigger!)",
            "⚡ ReLU activation: replace all negative numbers with zero (adds non-linearity)",
            "📉 Compress: multiply by another matrix → back to 512 numbers",
            "✅ Output: transformed representation that's the same size as input",
            "💡 This expand-process-compress happens independently for each word"
          ]
        }
      },
      intermediate: {
        motivation: "Self-attention is effective at aggregating information across positions but provides limited non-linear transformation capability (it's primarily a weighted average operation). The position-wise feedforward network adds the necessary depth and non-linearity to process the attended information, enabling the model to learn complex feature transformations independently for each position.",
        definition: "The position-wise feedforward network consists of two linear transformations with a ReLU activation: \\[ \\text{FFN}(x) = \\max(0, xW_1 + b_1)W_2 + b_2 \\] The hidden dimension is typically \\( 4 \\times \\) the model dimension (e.g., 2048 when \\( d_{\\text{model}}=512 \\)). 'Position-wise' means the same parameters are shared across all positions, but each position is processed independently—it's equivalent to two \\( 1 \\times 1 \\) convolutions.",
        toyExample: {
          description: "FFN computation for one position:",
          steps: [
            "📥 After attention: x ∈ R^512 (representation for one token)",
            "🔄 First linear layer: h = xW₁ + b₁, where W₁ ∈ R^(512×2048), b₁ ∈ R^2048",
            "📊 h ∈ R^2048 (expanded to 4× the original size)",
            "⚡ ReLU: h = max(0, h) → sparsity (typically ~50% of values become zero)",
            "🔄 Second linear layer: y = hW₂ + b₂, where W₂ ∈ R^(2048×512), b₂ ∈ R^512",
            "📤 Output: y ∈ R^512 (back to original dimension)",
            "🔁 This same transformation applied independently to all n tokens in parallel"
          ]
        }
      },
      advanced: {
        motivation: "The theoretical justification relates to universal approximation and feature learning. The two-layer MLP with ReLU can approximate arbitrary continuous functions over compact domains (universal approximation theorem). The intermediate expansion to \\( d_{ff} = 4d_{\\text{model}} \\) provides additional representational capacity and creates a higher-dimensional space for learning complex, non-linear feature interactions. The position-wise application ensures permutation equivariance is maintained.",
        definition: "The feedforward network is defined as: \\[ \\text{FFN}(x) = \\text{ReLU}(xW_1 + b_1)W_2 + b_2 \\] where \\( W_1 \\in \\mathbb{R}^{d_{\\text{model}} \\times d_{ff}} \\), \\( W_2 \\in \\mathbb{R}^{d_{ff} \\times d_{\\text{model}}} \\), typically \\( d_{ff} = 4d_{\\text{model}} \\). Parameters: \\( 2d_{\\text{model}} d_{ff} + d_{ff} + d_{\\text{model}} \\approx 8d_{\\text{model}}^2 \\) (dominates parameter count). The ReLU introduces non-linearity and sparsity; modern variants use GELU (Gaussian Error Linear Unit) for smoother gradients: \\( \\text{GELU}(x) = x\\Phi(x) \\) where \\( \\Phi \\) is the standard Gaussian CDF.",
        toyExample: {
          description: "Analysis of FFN's role and variants:",
          steps: [
            "📊 Parameter distribution in standard Transformer layer:",
            "   Multi-head attention: ~4d_model² params (Q,K,V,O projections)",
            "   FFN: ~8d_model² params (W₁, W₂) → FFN has 2× more parameters!",
            "🧮 For d_model=512, d_ff=2048: W₁ has 1M params, W₂ has 1M params",
            "🎯 Why 4× expansion? Empirically optimal; 2× gives worse performance, 8× no improvement",
            "⚡ ReLU vs GELU: GELU(x) = x·Φ(x) is smoother (differentiable everywhere)",
            "   GELU approximation: 0.5x(1 + tanh[√(2/π)(x + 0.044715x³)])",
            "🔬 Recent finding: FFN neurons learn interpretable features (some detect entities, others syntax)"
          ]
        }
      }
    }
  },
  {
    id: "concept_5",
    title: "Layer Normalization & Residual Connections",

    subConcepts: [
      { id: "sub_5_1", label: "Residual Connections" },
      { id: "sub_5_2", label: "Layer Normalization" },
      { id: "sub_5_3", label: "Gradient Flow" },
      { id: "sub_5_4", label: "Training Stability" }
    ],
    explanations: {
      beginner: {
        motivation: "When training deep neural networks, two problems emerge: (1) the numbers can grow too large or too small as they pass through many layers, making learning unstable, and (2) information can get 'lost' as it goes through many transformations. We need mechanisms to keep the network stable and preserve important information.",
        definition: "Layer Normalization keeps the numbers in a reasonable range by normalizing them (making their average 0 and spread consistent). Residual Connections are like bypass highways—they let information skip a layer and add directly to the output. So instead of output = layer(input), we do output = input + layer(input). This way, even if the layer transformation isn't perfect, the original information is preserved.",
        toyExample: {
          description: "Seeing residual connections in action:",
          steps: [
            "📥 Input: x = [word representation]",
            "🔄 Apply attention: attention_output = Attention(x)",
            "➕ Add residual: x₁ = x + attention_output (input shortcuts around attention)",
            "📏 Normalize: x₁_norm = LayerNorm(x₁) (standardize to mean=0, variance=1)",
            "🔄 Apply FFN: ffn_output = FFN(x₁_norm)",
            "➕ Add residual again: x₂ = x₁_norm + ffn_output",
            "📏 Normalize again: output = LayerNorm(x₂)",
            "💡 The original input x flows through via residual connections!"
          ]
        }
      },
      intermediate: {
        motivation: "Deep networks suffer from vanishing/exploding gradients and internal covariate shift (distributions changing between layers). Residual connections provide gradient highways, allowing gradients to flow directly through the network during backpropagation. Layer Normalization stabilizes the distribution of layer inputs, reducing sensitivity to parameter initialization and enabling higher learning rates.",
        definition: "Residual connections: \\( \\text{output} = x + \\text{Sublayer}(x) \\), creating an identity mapping that gradients can flow through unimpeded. Layer Normalization: \\[ \\text{LN}(x) = \\gamma \\odot ((x-\\mu)/\\sigma) + \\beta \\] where \\( \\mu,\\sigma \\) are mean and std computed across features (not batch). In Transformers, each sub-layer (attention, FFN) is wrapped with: \\( \\text{LayerNorm}(x + \\text{Sublayer}(x)) \\). This is 'Post-LN'; variants like 'Pre-LN' apply LayerNorm before the sub-layer: \\( x + \\text{Sublayer}(\\text{LayerNorm}(x)) \\).",
        toyExample: {
          description: "Computing Layer Normalization:",
          steps: [
            "📥 Input after residual: x ∈ R^d_model, e.g., x = [2.5, -1.0, 3.5, 0.5] (d=4)",
            "📊 Compute mean: μ = (2.5 - 1.0 + 3.5 + 0.5)/4 = 1.375",
            "📊 Compute variance: σ² = mean[(xᵢ - μ)²] = 3.234, so σ = 1.798",
            "🔄 Normalize: x̂ᵢ = (xᵢ - μ)/σ → x̂ = [0.625, -1.320, 1.182, -0.487]",
            "📏 Now x̂ has mean=0, std=1",
            "⚖️ Scale and shift: y = γ⊙x̂ + β (γ,β are learned parameters)",
            "✅ This stabilizes the distribution going into the next layer"
          ]
        }
      },
      advanced: {
        motivation: "Residual connections enable training of networks with hundreds of layers by creating direct paths for gradient flow. Without residuals, gradient magnitude decreases exponentially with depth: \\( ||\\partial L/\\partial x_1|| \\approx ||\\partial L/\\partial x_l|| \\cdot \\prod_i ||J_i|| \\), where \\( J_i \\) are Jacobians. With residuals, the gradient contains an additive identity term. LayerNorm has advantages over BatchNorm for sequence models: it doesn't depend on batch statistics (better for small batches) and treats each sequence independently (crucial for variable-length sequences).",
        definition: "Residual connection: \\( y = x + F(x) \\). During backpropagation: \\[ \\frac{\\partial L}{\\partial x} = \\frac{\\partial L}{\\partial y} \\cdot (1 + \\frac{\\partial F}{\\partial x}) \\] ensuring gradient always has the identity component. LayerNorm: \\[ \\text{LN}(x; \\gamma, \\beta) = \\gamma \\odot \\frac{x - \\mu}{\\sqrt{\sigma^2 + \\epsilon}} + \\beta \\] where \\( \\mu = \\frac{1}{d}\\sum x_i \\), \\( \\sigma^2 = \\frac{1}{d}\\sum (x_i - \\mu)^2 \\), computed per sample across features. \\( \\epsilon \\approx 10^{-5} \\) prevents division by zero. Learnable parameters: \\( \\gamma, \\beta \\in \\mathbb{R}^{d_{\\text{model}}} \\) (2 params per feature dimension).",
        toyExample: {
          description: "Gradient flow analysis with residuals:",
          steps: [
            "🔬 Without residual: y = F₁(F₂(...Fₗ(x)))",
            "   Gradient: ∂L/∂x = ∂L/∂y · ∏ᵢ₌₁ˡ ∂Fᵢ/∂Fᵢ₋₁",
            "   If ||∂Fᵢ/∂Fᵢ₋₁|| < 1, gradient vanishes as l↗",
            "🔬 With residual: y = x + F₁(x), and iteratively: yₗ = yₗ₋₁ + Fₗ(yₗ₋₁)",
            "   Gradient: ∂L/∂x = ∂L/∂yₗ · ∏ᵢ₌₁ˡ (1 + ∂Fᵢ/∂yᵢ₋₁)",
            "   Expanding: ∂L/∂x = ∂L/∂yₗ · (1 + Σ terms with ∂Fᵢ/∂yᵢ₋₁)",
            "   The '1' term provides a gradient highway!",
            "✅ Even if all ∂Fᵢ/∂yᵢ₋₁ ≈ 0, gradient is still ∂L/∂yₗ"
          ]
        }
      }
    }
  }
];

export const bertConcepts = [
  {
    id: "bert_0",
    title: "Big Picture: BERT",
    architectureImage: "PLACEHOLDER", // Placeholder, ideally should be BERT specific
    subConcepts: [
      { id: "bert_sub_0_1", label: "Bidirectional Context" },
      { id: "bert_sub_0_2", label: "Masked Language Model" },
      { id: "bert_sub_0_3", label: "Next Sentence Prediction" },
      { id: "bert_sub_0_4", label: "Transfer Learning" }
    ],
    explanations: {
      beginner: {
        motivation: "Transformers read the whole sentence at once, but the original Transformer was designed for translation (reading one language and writing another). What if we just want to understand a language really well? We need a model that can look at a word and understand it based on BOTH what comes before it AND what comes after it.",
        definition: "BERT (Bidirectional Encoder Representations from Transformers) is like a super-reader that reads text in both directions simultaneously. It's designed to create deep understandings of language that can be used for many different tasks, like answering questions or classifying emotions.",
        toyExample: {
          description: "Understanding Bidirectionality:",
          steps: [
            "📝 Sentence: 'The bank of the river' vs 'The bank of money'",
            "👀 Left-to-right model sees 'The bank' -> doesn't know which bank yet",
            "👀 Right-to-left model sees 'of the river' -> knows it's a river bank",
            "✨ BERT sees BOTH at once: 'The bank ... river' -> Ah! It's a river bank!",
            "💡 Context from both sides helps clarify meaning immediately"
          ]
        }
      },
      intermediate: {
        motivation: "Standard language models (like GPT) are unidirectional (left-to-right), limiting their ability to use future context for current token representation. ELMo used separate forward and backward LSTMs, but the combination was shallow. BERT introduces deep bidirectional training using the Transformer Encoder, allowing it to learn representations that are deeply fused from both directions.",
        definition: "BERT is a multi-layer bidirectional Transformer encoder. Unlike GPT which uses masked self-attention (looking only back), BERT uses full self-attention (looking everywhere). It is pre-trained on two tasks: Masked Language Modeling (MLM) to learn word relationships, and Next Sentence Prediction (NSP) to learn sentence relationships.",
        toyExample: {
          description: "BERT's Pre-training Tasks:",
          steps: [
            "🎭 MLM: Input 'The [MASK] sat on the mat'",
            "🤖 Model predicts: 'cat' (based on 'The', 'sat', 'on', 'mat')",
            "🔮 NSP: Input 'Sentence A: The man went to the store. Sentence B: He bought milk.'",
            "🤖 Model predicts: Is B the next sentence? YES",
            "📚 Result: A model that understands both word-level and sentence-level context"
          ]
        }
      },
      advanced: {
        motivation: "The limitation of unidirectional models is that the attention mechanism is causal. For tasks like SQuAD (Question Answering) or NER, knowing the future context is as important as the past. BERT solves this by using a 'Cloze' task (MLM) objective, which allows the model to see the full context while still having a prediction goal. This enables the learning of deep bidirectional representations.",
        definition: "BERT Architecture: A stack of Transformer Encoders (e.g., BERT-Base has 12 layers, 768 hidden size, 12 heads). Input representation = Token Embeddings + Segment Embeddings + Position Embeddings. Pre-training objective: Loss = Loss_MLM + Loss_NSP. Fine-tuning: Add a simple classification layer on top of the pre-trained BERT and train on the downstream task for a few epochs.",
        toyExample: {
          description: "Fine-tuning BERT:",
          steps: [
            "🏗️ Pre-trained BERT: Knows language structure (syntax, semantics)",
            "🎯 Task: Sentiment Analysis (Positive/Negative)",
            "➕ Add Classifier: A simple linear layer on top of the [CLS] token output",
            "🎓 Train: Update all weights (BERT + Classifier) on the sentiment dataset",
            "🚀 Result: State-of-the-art performance with minimal task-specific architecture"
          ]
        }
      }
    }
  },
  {
    id: "bert_1",
    title: "Masked Language Modeling",
    subConcepts: [
      { id: "bert_sub_1_1", label: "The Cloze Task" },
      { id: "bert_sub_1_2", label: "15% Masking Rule" },
      { id: "bert_sub_1_3", label: "Token Replacement" },
      { id: "bert_sub_1_4", label: "Prediction Softmax" }
    ],
    explanations: {
      beginner: {
        motivation: "How do you teach a computer to understand context? If you just let it read left-to-right, it cheats by just looking at the next word. We need a way to force it to use context clues to fill in the blanks.",
        definition: "Masked Language Modeling is like a 'fill-in-the-blank' game. We hide some words in a sentence (replace them with [MASK]) and ask BERT to guess what they are. To guess correctly, BERT has to understand the whole sentence.",
        toyExample: {
          description: "Playing Fill-in-the-Blank:",
          steps: [
            "📝 Original: 'The chef cooked a delicious meal'",
            "🙈 Masking: 'The chef [MASK] a delicious meal'",
            "🤔 BERT thinks: What fits here? 'cooked', 'made', 'prepared'...",
            "💡 Context Clue: 'chef' and 'delicious meal' suggest cooking verbs",
            "✅ Prediction: 'cooked'"
          ]
        }
      },
      intermediate: {
        motivation: "Standard conditional language models maximize P(w_t | w_1...w_{t-1}). This is strictly left-to-right. To enable bidirectional conditioning, we could try P(w_t | w_1...w_{t-1}, w_{t+1}...w_n), but in a multi-layer model, words would 'see themselves' in future layers, making the task trivial. MLM solves this by physically removing the token from the input.",
        definition: "In MLM, 15% of input tokens are selected for prediction. Of these: 80% are replaced with [MASK], 10% with a random token, and 10% kept unchanged. The model must predict the original token based on the context provided by the non-masked tokens. The loss is calculated only on the masked tokens.",
        toyExample: {
          description: "The 80-10-10 Rule:",
          steps: [
            "🎲 Select 'dog' to be masked in 'My dog is cute'",
            "8️⃣0️⃣% of time: Input 'My [MASK] is cute' -> Predict 'dog'",
            "1️⃣0️⃣% of time: Input 'My apple is cute' -> Predict 'dog' (learns to correct errors)",
            "1️⃣0️⃣% of time: Input 'My dog is cute' -> Predict 'dog' (learns to trust input)",
            "🎯 Why? Keeps the model adaptable and prevents it from only relying on [MASK] token"
          ]
        }
      },
      advanced: {
        motivation: "The mismatch between pre-training (where [MASK] appears) and fine-tuning (where it doesn't) is a concern. The 80-10-10 strategy mitigates this. Furthermore, since only 15% of tokens are predicted per batch, BERT requires more pre-training steps than autoregressive models (which predict every token) to converge.",
        definition: "Objective: Maximize log P(x_masked | x_unmasked). The final hidden vectors corresponding to the mask tokens are fed into an output softmax over the vocabulary. Unlike Denoising Autoencoders, BERT predicts the masked tokens directly rather than reconstructing the entire input.",
        toyExample: {
          description: "MLM Training Dynamics:",
          steps: [
            "📉 Loss calculation: CrossEntropy(Predicted_Prob, Actual_Token)",
            "🔄 Backprop: Gradients flow from the mask position back to ALL input tokens",
            "🧠 Attention's role: The [MASK] token attends to all other tokens to gather clues",
            "🤝 Other tokens attend to [MASK] to understand sentence structure (even if content is missing)",
            "📈 Result: Robust contextual embeddings"
          ]
        }
      }
    }
  },
  {
    id: "bert_2",
    title: "Next Sentence Prediction",
    subConcepts: [
      { id: "bert_sub_2_1", label: "Sentence Pairs" },
      { id: "bert_sub_2_2", label: "[CLS] and [SEP]" },
      { id: "bert_sub_2_3", label: "Segment Embeddings" },
      { id: "bert_sub_2_4", label: "Binary Classification" }
    ],
    explanations: {
      beginner: {
        motivation: "Many tasks involve two sentences, like Question & Answer or Cause & Effect. Just understanding words isn't enough; the model needs to understand the relationship between two separate sentences.",
        definition: "Next Sentence Prediction (NSP) teaches BERT to understand if two sentences belong together. We give it two sentences and ask: 'Did Sentence B come right after Sentence A?'",
        toyExample: {
          description: "Is this the next sentence?",
          steps: [
            "Example 1 (True Pair):",
            "A: 'I went to the store.'",
            "B: 'I bought some milk.'",
            "✅ BERT says: YES (IsNext)",
            "Example 2 (False Pair):",
            "A: 'I went to the store.'",
            "B: 'Penguins live in Antarctica.'",
            "❌ BERT says: NO (NotNext)"
          ]
        }
      },
      intermediate: {
        motivation: "To handle tasks like Question Answering (SQuAD) and Natural Language Inference (NLI), the model must understand the relationship between two text segments. NSP forces the model to capture long-term dependencies across sentences.",
        definition: "Input format: [CLS] Sentence A [SEP] Sentence B [SEP]. 50% of the time B is the actual next sentence (IsNext), 50% it's a random sentence from the corpus (NotNext). The [CLS] token at the start encodes the entire sequence representation, and a classifier on top of [CLS] predicts IsNext vs NotNext.",
        toyExample: {
          description: "Input Construction for NSP:",
          steps: [
            "📥 Tokens: [CLS] I like cats [SEP] They are cute [SEP]",
            "🔢 Segment IDs: [0, 0, 0, 0, 0, 1, 1, 1, 1]",
            "📍 Position IDs: [0, 1, 2, 3, 4, 5, 6, 7, 8]",
            "🧠 Model sees: Token + Segment + Position embeddings summed up",
            "🎯 Prediction target: IsNext (True)"
          ]
        }
      },
      advanced: {
        motivation: "While NSP was included in the original BERT, later research (like RoBERTa) showed that it might not be as crucial as thought, or that the way it was implemented (with short sequences) was the issue. However, understanding the mechanism is key to understanding BERT's input structure.",
        definition: "The [CLS] token is a special token added to the start of every sequence. The final hidden state of this token C ∈ R^H is used as the aggregate sequence representation. For NSP, we compute P(IsNext | C) = softmax(CW^T). This vector C is also what is typically used for classification tasks during fine-tuning.",
        toyExample: {
          description: "The role of [CLS]:",
          steps: [
            "🏗️ [CLS] is just another token during self-attention",
            "👀 It attends to all tokens in A and B",
            "🧠 It aggregates information from the whole pair",
            "🎓 During NSP training, it learns to encode 'relationship' features",
            "🚀 During Fine-tuning, this same [CLS] token is used for Sentiment, Entailment, etc."
          ]
        }
      }
    }
  },
  {
    id: "bert_3",
    title: "Fine-tuning BERT",
    subConcepts: [
      { id: "bert_sub_3_1", label: "Task-Specific Heads" },
      { id: "bert_sub_3_2", label: "Transfer Learning" },
      { id: "bert_sub_3_3", label: "Few-Shot Learning" },
      { id: "bert_sub_3_4", label: "Catastrophic Forgetting" }
    ],
    explanations: {
      beginner: {
        motivation: "Training a model like BERT from scratch takes huge computers and weeks of time. We don't want to do that for every single problem. Instead, we want to take the 'smart' BERT and teach it a specific job quickly.",
        definition: "Fine-tuning is like taking a college graduate (Pre-trained BERT) and giving them job-specific training. They already know how to read and think (Language), so they can learn the new job (Sentiment Analysis, Q&A) very fast with just a little bit of practice.",
        toyExample: {
          description: "Training vs. Fine-tuning:",
          steps: [
            "🏋️ Pre-training: Read the entire internet (Takes weeks, huge cost)",
            "🎓 Result: General purpose language understanding",
            "⚡ Fine-tuning: Learn to classify movie reviews (Takes minutes/hours)",
            "🛠️ How? Add a small 'decision maker' on top of BERT and train just a little bit",
            "✅ Result: Expert movie reviewer!"
          ]
        }
      },
      intermediate: {
        motivation: "The power of BERT comes from Transfer Learning. The features learned during MLM and NSP (syntax, semantics, context) are universally useful. We can adapt these features to downstream tasks without substantial architectural changes.",
        definition: "To fine-tune, we feed the input into BERT and take the output of the [CLS] token (for classification) or all tokens (for tagging/QA). We add a simple linear layer on top (W_task) and train the whole model (BERT + W_task) on the task dataset. Learning rate is typically small (e.g., 2e-5) to avoid destroying pre-trained weights.",
        toyExample: {
          description: "Fine-tuning for Question Answering (SQuAD):",
          steps: [
            "📥 Input: [CLS] Question [SEP] Passage [SEP]",
            "🎯 Goal: Find the start and end of the answer in the Passage",
            "🧠 BERT Output: Vector for every token in Passage",
            "➕ Add two vectors: Start_Vector and End_Vector",
            "🧮 Compute dot product of Token_Vectors with Start/End Vectors",
            "✅ Highest scores mark the answer span"
          ]
        }
      },
      advanced: {
        motivation: "Fine-tuning updates all parameters of the model, which can be computationally expensive for storing many task-specific models. Techniques like Adapters or LoRA (Low-Rank Adaptation) allow fine-tuning only a small subset of parameters, but full fine-tuning remains the gold standard for performance.",
        definition: "Catastrophic Forgetting is a risk where the model forgets its general language knowledge while learning the specific task. This is mitigated by using a very small learning rate and few epochs (2-4). The objective function changes from MLM/NSP to the task-specific loss (e.g., CrossEntropy for classification).",
        toyExample: {
          description: "Hyperparameters for Fine-tuning:",
          steps: [
            "📉 Learning Rate: 2e-5 to 5e-5 (much smaller than pre-training)",
            "⏳ Epochs: 2 to 4 (converges very quickly)",
            "📦 Batch Size: 16 or 32 (constrained by GPU memory)",
            "🌡️ Warmup: Linear warmup for first 10% of steps",
            "📉 Weight Decay: 0.01 to prevent overfitting on small datasets"
          ]
        }
      }
    }
  }
];

export const cnnConcepts = [
  {
    id: "cnn_0",
    title: "Big Picture: CNN",
    architectureImage: "PLACEHOLDER", // Placeholder
    subConcepts: [
      { id: "cnn_sub_0_1", label: "Concept 1" },
      { id: "cnn_sub_0_2", label: "Concept 2" },
      { id: "cnn_sub_0_3", label: "Concept 3" }
    ],
    explanations: {
      beginner: {
        motivation: "Motivation for CNN...",
        definition: "Definition of CNN...",
        toyExample: {
          description: "Toy example for CNN:",
          steps: ["Step 1", "Step 2", "Step 3"]
        }
      },
      intermediate: {
        motivation: "Intermediate motivation...",
        definition: "Intermediate definition...",
        toyExample: {
          description: "Intermediate example:",
          steps: ["Step 1", "Step 2", "Step 3"]
        }
      },
      advanced: {
        motivation: "Advanced motivation...",
        definition: "Advanced definition...",
        toyExample: {
          description: "Advanced example:",
          steps: ["Step 1", "Step 2", "Step 3"]
        }
      }
    }
  },
  {
    id: "cnn_1",
    title: "Core Concept 1",
    subConcepts: [],
    explanations: {
      beginner: { motivation: "...", definition: "...", toyExample: { description: "...", steps: [] } },
      intermediate: { motivation: "...", definition: "...", toyExample: { description: "...", steps: [] } },
      advanced: { motivation: "...", definition: "...", toyExample: { description: "...", steps: [] } }
    }
  },
  {
    id: "cnn_2",
    title: "Core Concept 2",
    subConcepts: [],
    explanations: {
      beginner: { motivation: "...", definition: "...", toyExample: { description: "...", steps: [] } },
      intermediate: { motivation: "...", definition: "...", toyExample: { description: "...", steps: [] } },
      advanced: { motivation: "...", definition: "...", toyExample: { description: "...", steps: [] } }
    }
  }
];





export const latentConcepts = [
  {
    id: "latent_0",
    title: "Big Picture: Latent",
    architectureImage: "PLACEHOLDER", // Placeholder
    subConcepts: [
      { id: "latent_sub_0_1", label: "Concept 1" },
      { id: "latent_sub_0_2", label: "Concept 2" },
      { id: "latent_sub_0_3", label: "Concept 3" }
    ],
    explanations: {
      beginner: {
        motivation: "Motivation for Latent...",
        definition: "Definition of Latent...",
        toyExample: {
          description: "Toy example for Latent:",
          steps: ["Step 1", "Step 2", "Step 3"]
        }
      },
      intermediate: {
        motivation: "Intermediate motivation...",
        definition: "Intermediate definition...",
        toyExample: {
          description: "Intermediate example:",
          steps: ["Step 1", "Step 2", "Step 3"]
        }
      },
      advanced: {
        motivation: "Advanced motivation...",
        definition: "Advanced definition...",
        toyExample: {
          description: "Advanced example:",
          steps: ["Step 1", "Step 2", "Step 3"]
        }
      }
    }
  },
  {
    id: "latent_1",
    title: "Core Concept 1",
    subConcepts: [],
    explanations: {
      beginner: { motivation: "...", definition: "...", toyExample: { description: "...", steps: [] } },
      intermediate: { motivation: "...", definition: "...", toyExample: { description: "...", steps: [] } },
      advanced: { motivation: "...", definition: "...", toyExample: { description: "...", steps: [] } }
    }
  },
  {
    id: "latent_2",
    title: "Core Concept 2",
    subConcepts: [],
    explanations: {
      beginner: { motivation: "...", definition: "...", toyExample: { description: "...", steps: [] } },
      intermediate: { motivation: "...", definition: "...", toyExample: { description: "...", steps: [] } },
      advanced: { motivation: "...", definition: "...", toyExample: { description: "...", steps: [] } }
    }
  }
];




