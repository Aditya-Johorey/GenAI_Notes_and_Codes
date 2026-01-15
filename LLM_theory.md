## What is an LLM?
> An LLM is a pobabilistic model trained to predict the next word based on patterns in massive text data.

`eg: A Horse is an ___ ? ____ -> animal`

* LLMs by themselves dont understand context
    * So they cannot be a search engine by nature
    * Not AGI
* LLMs speak and understand tokens

### Then what are Tokens?
> A token is a small piece of text that the LLM actually reads and writes.

**Note:** A word is not necessarily a token. They fundamental units that can be words, subwords, or even characters, that text is broken down into for machines to process, understand, and analyze.

For eg:

TEXT | No of Tokens
---|---
Yes | 1 
Tokenization | 2 ("Token", "ization")
Hello! | 2 ("Hello", "!")
""  | 1 (Whitespace)

**Why they matter?**
  * Cost (APIs charge per token)
      * 4k tokens ≈ ~3,000 words
      * 16k tokens ≈ long conversations
  * Context window (memory limit)
  * Truncation (old messages disappear)

* LLMs never see *“sentences”*. They only see **tokens** in a *sequence*.
* Tokens are formed by the process of **Tokenization**.

### Ugh! What is Tokenization ?
>The process of breaking raw text into smaller, manageable units called tokens

Types:
1. **Word-Based:** Splits text at every space or punctuation mark (e.g., "Let's go" → ["Let", "'s", "go"]).
2. **Character-Based:** Breaks text into every single letter or symbol (e.g., "Hi" → ["H", "i"]). This is highly robust to typos but creates very long sequences.
3. **Subword:** Used by models like GPT-4 and BERT, this method breaks rare words into smaller common chunks (e.g., "unhappiness" → ["un", "happi", "ness"]).
   
Workflow: 
1. **Normalization:** The raw text is cleaned by standardizing it (e.g., converting to lowercase, removing extra whitespaces, or fixing encoding issues).
2. **Segmentation:** A tokenizer applies specific rules or algorithms to split the text into discrete units.
3. **Mapping to IDs:** Each unique token is assigned a numerical integer ID from the model's fixed vocabulary.
4. **Encoding:** The final sequence of these IDs is what is actually fed into the neural network for processing.

### Where does probability fit in the equation?
When you give an LLM a prompt, it looks at the sequence and calculates a probability distribution over its entire vocabulary.

For eg: 

`
"The capital of France is..."
Probabilities: Paris (98%), Lyon (1%), a (0.5%), etc.
`

These scores are called Logits. These are passed through a Softmax function which squashes them into probabilities that add up to 100% (1.0).

**Sampling (Helps LLM be more creative with expression):**
The model doesn't always pick the #1 highest probability token. That would make it repetitive and boring. We use "sampling" to control this: 

1. Temperature: High temperature makes the distribution "flatter," giving lower-probability tokens a better chance (increasing creativity). Low temperature makes it "sharper," focusing only on the top choice.
2. Top-P (Nucleus Sampling): The model only considers the top tokens whose cumulative probability adds up to a certain threshold (e.g., 0.9).

But this might sometimes lead to **Hallucinations.**

> A "hallucination" happens when the model generates a high-probability sequence that is factually incorrect. This occurs because the model prioritizes statistical patterns (how words usually follow each other) over actual database lookups.


