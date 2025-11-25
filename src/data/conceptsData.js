export const transformerConcepts = [
  {
    id: "concept_0",
    title: "1. Big Picture",
    architectureImage: "/transformers.png",
    codeSnippet: `import torch
import torch.nn as nn

class TinyTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = nn.Parameter(torch.zeros(1, 512, d_model))  # toy
        self.encoder_layers = nn.ModuleList(
            [nn.TransformerEncoderLayer(d_model=d_model, nhead=8, dim_feedforward=2048)
             for _ in range(n_layers)]
        )
        self.decoder_layers = nn.ModuleList(
            [nn.TransformerDecoderLayer(d_model=d_model, nhead=8, dim_feedforward=2048)
             for _ in range(n_layers)]
        )
        self.out_proj = nn.Linear(d_model, vocab_size)

    def forward(self, src_ids, tgt_ids):
        src = self.embedding(src_ids) + self.pos_encoding[:, :src_ids.size(1)]
        tgt = self.embedding(tgt_ids) + self.pos_encoding[:, :tgt_ids.size(1)]
        for layer in self.encoder_layers:
            src = layer(src)
        memory = src
        for layer in self.decoder_layers:
            tgt = layer(tgt, memory)
        logits = self.out_proj(tgt)
        return logits`,
    subConcepts: [],
    explanations: {
      beginner: {
        motivation: "We want a model that can read a sentence (e.g., in English) and produce another sentence (e.g., in French) while understanding context and word meaning.",
        definition: "A transformer is a model that first turns input words into vectors, processes them with attention blocks (encoder), and then uses another set of blocks (decoder) to generate the output one token at a time."
      },
      advanced: {
        motivation: "We want a model that uses content-based addressing (attention) over sequences, is fully parallelizable via matrix multiplications, and scales well in depth and width without recurrence or convolution.",
        definition: "A transformer is a sequence transducer composed of stacked self-attention + position-wise feed-forward modules, using multi-head attention to capture multiple representation subspaces, and masking for autoregressive decoding."
      }
    }
  },
  {
    id: "concept_1",
    title: "2. Tokenization & Embeddings",

    codeSnippet: `import torch
import torch.nn as nn

vocab_size = 10000
d_model = 512

embedding = nn.Embedding(vocab_size, d_model)

# Example token IDs: batch_size=2, seq_len=5
token_ids = torch.tensor([[5, 8, 123, 9, 2],
                          [7, 42, 99, 1, 0]])

embedded = embedding(token_ids)  # (2, 5, 512)`,
    subConcepts: [],
    explanations: {
      beginner: {
        motivation: "Computers work with numbers, not raw text. We need a way to turn words into numbers.",
        definition: "Tokenization splits text into tokens (like “think”, “##ing”). Embeddings are lookup tables mapping each token ID to a vector."
      },
      advanced: {
        motivation: "The discrete tokenization (BPE, WordPiece, Unigram LM) approximates an optimal trade-off between vocabulary size, sequence length, and statistical efficiency; embeddings map these discrete indices to dense vectors that live in a continuous space where similar tokens are close.",
        definition: "Embeddings are learned parameters that serve as the initial point in a high-dimensional latent space; they are often shared between input and output layers to tie representational and predictive distributions."
      }
    }
  },
  {
    id: "concept_2",
    title: "3. Positional Encoding",

    codeSnippet: `import math
import torch

def sinusoidal_positional_encoding(max_len, d_model):
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (max_len, 1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                         (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe  # (max_len, d_model)

max_len = 100
d_model = 512
pos_enc = sinusoidal_positional_encoding(max_len, d_model)  # (100, 512)`,
    subConcepts: [],
    explanations: {
      beginner: {
        motivation: "Attention alone doesn’t know the order of words, so we must tell the model where each token is in the sentence.",
        definition: "Positional encoding is a vector that represents the position (1st, 2nd, 3rd…) of each token, added to the embedding."
      },
      advanced: {
        motivation: "Sinusoidal positional encodings give a smooth, absolute position representation where relative positions can be expressed as linear functions; alternative schemes (learned, rotary, ALiBi, etc.) encode relative or implicit distance.",
        definition: "The sinusoidal basis forms a set of fixed, orthogonal frequency components so that any relative shift can be represented via phase shifts; the model can extrapolate to longer sequences because the pattern is deterministic.",
        equations: [
          {
            latex: "\\mathrm{PE}_{\\text{pos}, 2i} = \\sin\\left(\\frac{\\text{pos}}{10000^{2i / d_{\\text{model}}}}\\right), \\quad \\mathrm{PE}_{\\text{pos}, 2i+1} = \\cos\\left(\\frac{\\text{pos}}{10000^{2i / d_{\\text{model}}}}\\right)",
            explanation: "Sinusoidal positional encoding formulas."
          }
        ]
      }
    }
  },
  {
    id: "concept_3",
    title: "4. Scaled Dot-Product Self-Attention",

    codeSnippet: `import torch
import torch.nn as nn
import math

class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k):
        super().__init__()
        self.d_k = d_k

    def forward(self, Q, K, V, mask=None):
        # Q, K, V: (batch, seq_len, d_k)
        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.d_k)  # (batch, seq, seq)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = torch.softmax(scores, dim=-1)  # attention weights
        output = attn @ V  # (batch, seq, d_k)
        return output, attn`,
    subConcepts: [],
    explanations: {
      beginner: {
        motivation: "Each word should be able to “look at” other words to decide what’s important for understanding its meaning.",
        definition: "Self-attention computes scores between each pair of tokens, turns them into probabilities (with softmax), and then combines the value vectors using these probabilities."
      },
      advanced: {
        motivation: "Self-attention implements a soft, learned similarity kernel between query–key pairs; scaling by $1/\\sqrt{d_k}$ stabilizes gradients by keeping dot-product magnitudes controlled.",
        definition: "This is a differentiable, low-rank approximation of pairwise interactions among sequence elements, facilitating modeling of long-range dependencies with $O(n^2)$ complexity and excellent parallelism.",
        equations: [
          {
            latex: "\\mathrm{Attention}(Q, K, V) = \\mathrm{softmax}\\left(\\frac{QK^\\top}{\\sqrt{d_k}}\\right) V",
            explanation: "Scaled Dot-Product Attention formula."
          }
        ]
      }
    }
  },
  {
    id: "concept_4",
    title: "5. Multi-Head Attention",

    codeSnippet: `class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads
        self.num_heads = num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.attn = ScaledDotProductAttention(self.d_k)

    def forward(self, x, mask=None, kv=None):
        # x: (batch, seq_len, d_model) for queries
        # kv: if None, self-attention; else (batch, kv_seq, d_model)
        if kv is None:
            kv = x
        batch_size, seq_len, d_model = x.size()
        kv_len = kv.size(1)

        Q = self.W_q(x)  # (batch, seq, d_model)
        K = self.W_k(kv)
        V = self.W_v(kv)

        # reshape to (batch, heads, seq, d_k)
        def split_heads(t):
            return t.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        Q, K, V = map(split_heads, (Q, K, V))

        # attention
        out, attn = self.attn(Q, K, V, mask=mask)  # (batch, heads, seq, d_k)

        # concat heads
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        out = self.W_o(out)
        return out, attn`,
    subConcepts: [],
    explanations: {
      beginner: {
        motivation: "One attention pattern might miss some relationships. Multiple “heads” let the model look at the sentence in different ways at the same time.",
        definition: "Multi-head attention runs self-attention several times with different learned projections, then combines the results."
      },
      advanced: {
        motivation: "Multi-head attention projects inputs into multiple learned subspaces, performs attention in parallel, and concatenates the results; this increases expressivity while keeping per-head dimension smaller.",
        definition: "The final output is $\\mathrm{MHA}(X) = \\mathrm{Concat}(\\mathrm{head}_1, \\dots, \\mathrm{head}_H)W^O$ where each head performs attention with lower-dimensional queries/keys/values."
      }
    }
  },
  {
    id: "concept_5",
    title: "6. Feed-Forward Networks (FFN)",

    codeSnippet: `class PositionwiseFFN(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.activation = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.activation(self.fc1(x)))`,
    subConcepts: [],
    explanations: {
      beginner: {
        motivation: "After mixing information between words (attention), we need to process each word individually to extract more complex meaning from it.",
        definition: "The Feed-Forward Network is a simple two-layer neural network applied to every token separately and identically. It expands the dimension (e.g., 512 → 2048) and then projects it back."
      },
      advanced: {
        motivation: "Position-wise FFNs implement deep, shared, token-wise nonlinearity, increasing the representational rank of the model.",
        definition: "FFNs can be viewed as key-value memories where the keys are the first layer weights and values are the second layer weights. Equation: $\\mathrm{FFN}(x) = W_2 \\, \\sigma(W_1 x + b_1) + b_2$",
        equations: [
          {
            latex: "\\mathrm{FFN}(x) = W_2 \\, \\sigma(W_1 x + b_1) + b_2",
            explanation: "Feed-Forward Network equation."
          }
        ]
      }
    }
  },
  {
    id: "concept_6",
    title: "7. Add & Norm (Residuals & Layer Normalization)",

    codeSnippet: `class AddNorm(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, sublayer_out):
        return self.norm(x + sublayer_out)`,
    subConcepts: [],
    explanations: {
      beginner: {
        motivation: "As we add more layers, training gets harder and numbers can get too big or too small. We need a way to keep things stable and remember the original input.",
        definition: "“Add” (Residual Connection) adds the input to the output of a layer to preserve information. “Norm” (Layer Normalization) keeps the numbers in a reasonable range for stable training."
      },
      advanced: {
        motivation: "Residual streams allow gradients to flow through many layers (mitigating vanishing gradients); LayerNorm normalizes feature-wise statistics per position to stabilize training dynamics.",
        definition: "LayerNorm: $\\mathrm{LN}(h) = \\gamma \\frac{h - \\mu}{\\sqrt{\\sigma^2 + \\epsilon}} + \\beta$ with $\\mu, \\sigma$ computed over the feature dimension. The output is $x + \\mathrm{SubLayer}(x)$.",
        equations: [
          {
            latex: "\\mathrm{LN}(h) = \\gamma \\frac{h - \\mu}{\\sqrt{\\sigma^2 + \\epsilon}} + \\beta",
            explanation: "Layer Normalization equation."
          }
        ]
      }
    }
  },
  {
    id: "concept_7",
    title: "8. Encoder Block / Encoder Stack",

    codeSnippet: `class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.addnorm1 = AddNorm(d_model)
        self.ffn = PositionwiseFFN(d_model, d_ff)
        self.addnorm2 = AddNorm(d_model)

    def forward(self, x, src_mask=None):
        attn_out, _ = self.self_attn(x, mask=src_mask)
        x = self.addnorm1(x, attn_out)
        ffn_out = self.ffn(x)
        x = self.addnorm2(x, ffn_out)
        return x

class Encoder(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, num_layers):
        super().__init__()
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, num_heads, d_ff) for _ in range(num_layers)]
        )

    def forward(self, x, src_mask=None):
        for layer in self.layers:
            x = layer(x, src_mask)
        return x  # memory`,
    subConcepts: [],
    explanations: {
      beginner: {
        motivation: "One layer of attention isn’t enough. Stacking multiple layers lets the model build more abstract understanding of the input sentence.",
        definition: "An encoder block is: multi-head self-attention → Add & Norm → FFN → Add & Norm. The encoder stack repeats that block several times."
      },
      advanced: {
        motivation: "The encoder stack composes multiple self-attention + FFN transformations, forming a deep residual network where each layer can attend to arbitrary receptive fields and implement complex hierarchical transformations.",
        definition: "All encoder layers share the same structural pattern but have independent parameters; the final encoder output is a sequence of contextualized representations used as keys/values in decoder cross-attention.",
        equations: [
          {
            latex: "H = \\mathrm{AddNorm}(X^{(\\ell)}, \\mathrm{MHA}(X^{(\\ell)}))",
            explanation: "Encoder layer first sub-layer."
          },
          {
            latex: "X^{(\\ell+1)} = \\mathrm{AddNorm}(H, \\mathrm{FFN}(H))",
            explanation: "Encoder layer second sub-layer."
          }
        ]
      }
    }
  },
  {
    id: "concept_8",
    title: "9. Cross-Attention",

    codeSnippet: `class MultiHeadAttention(nn.Module):
    # ... (same as Self-Attention)
    def forward(self, query, key, value, mask=None):
        # In Cross-Attention:
        # query comes from Decoder (Target)
        # key, value come from Encoder (Source)
        ...
        return output`,
    subConcepts: [],
    explanations: {
      beginner: {
        motivation: "The decoder knows what it has generated so far, but it needs to look at the original input sentence to translate it correctly.",
        definition: "Cross-Attention lets the decoder 'look back' at the encoder's output. It uses the decoder's current state as the Query, and the encoder's output as Keys and Values."
      },
      advanced: {
        motivation: "Cross-attention is the mechanism for conditioning the generation on the source context, enabling sequence-to-sequence alignment.",
        definition: "It computes attention where $Q = X_{dec}W^Q$ and $K, V = X_{enc}W^K, X_{enc}W^V$. This allows the decoder to attend to relevant parts of the source sequence based on its current generation state.",
        equations: [
          {
            latex: "\\mathrm{Attention}(Q, K, V) = \\mathrm{softmax}\\left(\\frac{QK^T}{\\sqrt{d_k}}\\right)V",
            explanation: "Standard attention equation, but Q comes from decoder, K/V from encoder."
          }
        ]
      }
    }
  },
  {
    id: "concept_9",
    title: "10. Decoder Block / Decoder Stack",

    codeSnippet: `def subsequent_mask(size):
    # Mask out (set to 0) future positions
    # returns (1, size, size) for broadcasting
    attn_shape = (1, size, size)
    mask = torch.tril(torch.ones(attn_shape)).bool()
    return mask  # True = keep, False = block

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.addnorm1 = AddNorm(d_model)
        self.addnorm2 = AddNorm(d_model)
        self.ffn = PositionwiseFFN(d_model, d_ff)
        self.addnorm3 = AddNorm(d_model)

    def forward(self, y, memory, tgt_mask=None, memory_mask=None):
        self_attn_out, _ = self.self_attn(y, mask=tgt_mask)
        y = self.addnorm1(y, self_attn_out)

        cross_out, _ = self.cross_attn(y, mask=memory_mask, kv=memory)
        y = self.addnorm2(y, cross_out)

        ffn_out = self.ffn(y)
        y = self.addnorm3(y, ffn_out)
        return y

class Decoder(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, num_layers):
        super().__init__()
        self.layers = nn.ModuleList(
            [DecoderLayer(d_model, num_heads, d_ff) for _ in range(num_layers)]
        )

    def forward(self, y, memory):
        seq_len = y.size(1)
        tgt_mask = subsequent_mask(seq_len).to(y.device)
        for layer in self.layers:
            y = layer(y, memory, tgt_mask=tgt_mask)
        return y`,
    subConcepts: [],
    explanations: {
      beginner: {
        motivation: "When generating a sentence, the model shouldn’t peek at future words; it should only use words it has already produced plus the encoded input.",
        definition: "A decoder block has: masked self-attention, encoder–decoder attention, FFN, with Add & Norm around each. The decoder stack repeats this multiple times."
      },
      advanced: {
        motivation: "Causal masks enforce autoregressive factorization; cross-attention lets the decoder perform content-based addressing over encoder states, analogous to traditional sequence-to-sequence attention mechanisms.",
        definition: "The causal mask is an upper-triangular $-\\infty$ mask applied to attention logits; cross-attention uses queries from the decoder and keys/values from the encoder, enabling soft alignment between source and target tokens.",
        equations: [
          {
            latex: "H_1 = \\mathrm{AddNorm}(Y^{(\\ell)}, \\mathrm{MHA}_{\\text{mask}}(Y^{(\\ell)}))",
            explanation: "Masked self-attention sub-layer."
          },
          {
            latex: "H_2 = \\mathrm{AddNorm}(H_1, \\mathrm{MHA}_{\\text{cross}}(H_1, M))",
            explanation: "Cross-attention sub-layer."
          }
        ]
      }
    }
  }
];

export const bertConcepts = [
  {
    id: "bert_0",
    title: "Big Picture: BERT",

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
    title: "Big Picture: Latent Diffusion",

    subConcepts: [
      { id: "latent_sub_0_1", label: "Concept 1" },
      { id: "latent_sub_0_2", label: "Concept 2" },
      { id: "latent_sub_0_3", label: "Concept 3" }
    ],
    explanations: {
      beginner: {
        motivation: "Motivation for Latent Diffusion...",
        definition: "Definition of Latent Diffusion...",
        toyExample: {
          description: "Toy example for Latent Diffusion:",
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




