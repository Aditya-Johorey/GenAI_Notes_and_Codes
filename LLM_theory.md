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
___
## How to train this "BlackBox"?
Training LLMS work in 3 stages sequencially.

1. Pretraining
2. Fine-Tuning
3. RLHF (Reinforcement Learning with Human Feedback)

### PreTraining (KinderGarten for LLMs)
> It is the massive, computationally expensive stage where a model learns the basic patterns, grammar, and facts of human language by reading a huge chunk of the internet.

The model is fed trillions of tokens from diverse sources like Common Crawl, Wikipedia, and GitHub. During this phase, the data is unlabelled.
Objective: Next Token Prediction.
Outcomes:
1. Forms basic Grammar and Patterns
2. Learns Global Facts
3. Logic and Reasoning

At this stage, a raw, biased, unsupervised base model is prepared, pretty informative and has deep understanding of statistical patterns in the language, but unable to take and execute orders properly yet.
The nature of this base model, depends majorly on the data fed to it.

### Fine Tuning (College Grad in making)
> Fine-tuning teaches it how to behave and follow specific instructions.

Supervised Fine-Tuning (SFT):
High quality datasets are curated with the format of Ideal prompts and responses.

```
Input: "Write a poem about a robot."
Target Output: [A high-quality, human-written poem.]
Result: The model learns the format of a helpful assistant.
```

At this stage, the model develops:
1. Sense of tone
2. Text Formatting
3. Domain Expertise
4. Instruction Obedience

Pretraining is actually a costly opperation. A shortcut to perform it is by performing Low-Rank Adaptation (LoRA) on the base model.
* It only updates a tiny, lightweight "adapter" that sits on top of the base model.
* It only trains about 0.01% to 1% of the model's total parameters.
* No forgetting since the base model is *frozen*.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# 1. Load the original "Frozen" Base Model
base_model_id = "meta-llama/Llama-2-7b-hf" 
model = AutoModelForCausalLM.from_pretrained(base_model_id, device_map="auto")

# 2. Load the tiny LoRA Adapter (the "specialization" layer)
# This 'adapter_id' is usually just a few megabytes
adapter_id = "your-username/llama-2-7b-lora-medical"
model = PeftModel.from_pretrained(model, adapter_id)

# 3. Use it like a normal model
tokenizer = AutoTokenizer.from_pretrained(base_model_id)
inputs = tokenizer("Symptoms of a cold include:", return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=50)

print(tokenizer.decode(outputs[0]))

```

### RLHF (Expericed Professional)
> The final "polishing" phase that turns a smart AI into a helpful, safe, and conversational assistant

It is done in 3 steps:
1. **Sampling:** The model generates multiple different responses to the same prompt (e.g., one is short, one is long, one is slightly rude).
2. **Human Ranking:** Human trainers look at these options and rank them (e.g., "Response B is better than Response A"). This captures subjective qualities like "helpfulness" or "tone" that math alone can't define.
3. **The Reward Model:** A separate, smaller AI (the Reward Model) is trained to learn these human preferences. It becomes a digital "judge" that can score the main LLM's outputs instantly.

Outcomes:
* Safety
* Develop Tone & Style
* Truthfulness (Penalizes Halucinations)

### Summary and Comparrison

| Aspect      | Pretraining    | Fine-tuning       | Prompting      |
| ----------- | -------------- | ----------------- | -------------- |
| Who does it | Big companies  | Companies / teams | You            |
| Cost        | Millions $$$   | Thousands $$$     | Free           |
| Purpose     | Learn language | Learn behavior    | Control output |
| Knowledge   | Yes (patterns) | No                | No             |
| Flexibility | Fixed          | Medium            | Very high      |

### Why RAG is better?
> RAG (Retrieval-Augmented Generation) is like an "open-book exam." Instead of relying solely on its memory (pre-training), the model looks up specific, fresh information from an external source before answering.


Working:
1. **Retrieve:** When you ask a question, the system searches your private documents (PDFs, databases, or live websites) for relevant snippets.
2. **Augment:** It sticks those snippets into your prompt as "context."
3. **Generate:** The LLM reads the context and writes an answer based only on that information.

Benefits:
* Lesser Hallucinations
* Up-to-Date
* Ensures Privacy

Popular Tools:
* LangChain: The most popular framework for building RAG pipelines.
* LlamaIndex: Specialized in connecting "data sticks" to LLMs.
* Vector Databases: Tools like Pinecone or Chroma store your documents as "embeddings" (numbers) so the AI can find them instantly.


