## Controlling Output Quality

### What "Output Quality" Actually Means

Before diving into techniques, it is worth defining what quality actually means in the context of LLM outputs — because it is not one thing, it is several.

A high-quality output is:
- **Accurate** — factually correct and grounded in the task
- **Relevant** — answers what was actually asked, nothing more
- **Consistent** — produces similar structure and style across multiple runs
- **Appropriately formatted** — structured in a way that is immediately usable
- **Calibrated in tone** — matches the audience and context
- **Concise** — no unnecessary padding, hedges, or repetition

Most prompting mistakes cause failures in one or more of these dimensions. The techniques in this section each target specific failure modes. Understanding *why* each technique works makes you a far better prompt engineer than simply memorizing the techniques themselves.

---

### 1. Role Prompting (Persona Conditioning)

Assigning the model a role is one of the fastest ways to shift output quality across multiple dimensions simultaneously. A well-chosen role activates a cluster of behaviors — vocabulary, reasoning depth, tone, caution level — in a single sentence.

```
You are a senior UX researcher with 10 years of experience
conducting user interviews for enterprise software products.
```

This one line implicitly tells the model to:
- Use research terminology accurately
- Frame findings in terms of user needs, not opinions
- Apply structured thinking before drawing conclusions
- Avoid overclaiming based on limited data

**Why it works at a technical level:** During pretraining, the model absorbed millions of documents written by, about, and in the voice of specific professional roles. Naming a role activates the statistical patterns associated with that identity — the vocabulary it uses, the way it structures arguments, the level of hedging it applies, the assumptions it makes about the reader.

**The difference between a weak and strong role:**

| Weak | Strong |
|---|---|
| "You are an expert." | "You are a board-certified emergency physician explaining a diagnosis to a frightened patient." |
| "You are a writer." | "You are a New Yorker staff writer known for sharp, economical prose and dry wit." |
| "You are a teacher." | "You are a high school physics teacher explaining concepts to students who struggle with math anxiety." |

The weak versions give the model almost nothing to work with. The strong versions specify expertise, context, audience relationship, and communication style — all in one sentence.

**Role stacking** is an advanced variation where you assign the model multiple complementary roles:

```
You are both a data scientist and a science communicator.
Your job is to explain technical findings accurately while
making them accessible to a non-technical executive audience.
```

This is useful when the task requires expertise in one domain but communication skills from another.

---

### 2. Audience Targeting

This is one of the most commonly skipped components of a prompt, and its absence is responsible for a huge proportion of outputs that are technically correct but practically useless.

The model does not know who will read its output unless you tell it. Without that information, it defaults to a generic "educated adult" assumption — which may be completely wrong for your situation.

```
Explain how compound interest works.

Audience: A 16-year-old who has just received their first paycheck
and is thinking about opening a savings account.
```

versus

```
Explain how compound interest works.

Audience: A CFO evaluating whether to restructure the company's
debt obligations ahead of a rate change.
```

The topic is identical. The output should be entirely different. Without the audience specification, the model picks one arbitrarily.

**Audience targeting controls:**
- Vocabulary complexity and jargon level
- Assumed prior knowledge
- Depth of explanation
- Tone and formality
- Use of analogies vs technical precision
- Length — experts need less background, beginners need more

**A practical framework for defining your audience:**

```
Audience:
- Who they are: [role or identity]
- What they already know: [relevant background]
- What they do not know: [gaps to fill]
- What they will do with this output: [downstream use]
```

---

### 3. Constraints and Guardrails

Constraints are instructions about what the model must **not** do, or hard limits it must stay within. They are underused because they feel negative — but they are one of the most powerful quality levers available.

**Why constraints improve quality:** Without them, the model optimizes for completeness and fluency, which in practice means verbose, hedge-heavy, over-explained output. Constraints force the model to prioritize — and prioritization is where quality comes from.

```
Constraints:
- Maximum 150 words
- No bullet points — write in flowing prose
- Do not use the phrases "it's worth noting" or "in conclusion"
- Do not suggest solutions — only identify problems
- Avoid passive voice
```

**Types of constraints and what they fix:**

| Constraint type | Example | What it prevents |
|---|---|---|
| Length | "Under 100 words" | Padding and repetition |
| Style | "No jargon" | Inaccessible language |
| Format | "Plain text only" | Unwanted markdown in plain contexts |
| Scope | "Only address pricing, not features" | Topic drift |
| Negative | "Do not recommend surgery as a first option" | Specific failure modes |
| Knowledge | "Do not reference events after 2022" | Hallucination from uncertain data |
| Ethical | "Do not make claims about competitor products" | Legal or brand risk |

**Negative constraints** deserve special attention. Telling the model explicitly what not to do is often more effective than describing what to do, because it directly closes off the failure path.

```
Do not begin your response with "Certainly!" or "Great question!"
Do not summarize what you are about to say before saying it.
Do not add a closing paragraph restating your main points.
```

These three constraints alone eliminate the most common sources of LLM padding.

**Guardrails for safety-critical outputs:**

When outputs will be used in high-stakes contexts — medical, legal, financial — guardrails prevent the model from overstepping:

```
If you are uncertain about any fact, say so explicitly.
Do not present opinions as facts.
Always recommend the user consult a qualified professional
before acting on this information.
```

---

### 4. Output Formatting and Schema Control

Defining the structure of the output is not just an aesthetic choice — it is what makes outputs reliable, parseable, and immediately usable. This becomes especially critical when outputs feed into automated systems, APIs, or downstream processes.

**Prose formatting:**

```
Structure your response as follows:
1. A one-sentence summary of the main finding
2. Three supporting points, each in its own paragraph
3. A single recommended next action
```

**JSON schema control:**

```
Return your answer as a JSON object with exactly these keys:

{
  "headline": "string, max 10 words",
  "sentiment": "positive | negative | neutral",
  "confidence": "number between 0 and 1",
  "key_topics": ["array", "of", "strings"]
}

Return only the JSON. No preamble, no explanation, no markdown fences.
```

The instruction "return only the JSON" is critical. Without it, the model frequently adds a sentence before or after the JSON, which breaks any parser expecting pure JSON.

**Markdown control:**

LLMs default to heavy markdown formatting — bold text, headers, bullet points everywhere. This is great in a chat interface but terrible if the output goes into a plain text email, a database field, or a voice system.

```
Format your response as plain text only.
Do not use markdown, bullet points, bold, italics, or headers.
Write in natural paragraphs.
```

**Table formatting:**

```
Present your comparison as a markdown table with these columns:
| Feature | Option A | Option B | Winner |
```

**Structured extraction:**

When pulling specific information from a document, defining a schema prevents the model from paraphrasing or reinterpreting:

```
Extract the following fields from the contract below.
If a field is not present, write "Not specified."

- Party A name:
- Party B name:
- Contract start date:
- Contract end date:
- Payment terms:
- Governing law:
```

---

### 5. Iterative Refinement (Generate → Critique → Revise)

This is one of the most underused techniques for dramatically improving output quality, and it mirrors how skilled human writers actually work. Almost no professional produces a final draft in one pass — they draft, critique, and revise.

You can instruct the model to do the same within a single prompt or across multiple turns.

**Single-prompt version:**

```
Step 1: Write a first draft of the email.
Step 2: Critique your own draft. Identify what is weak, vague,
or could be misinterpreted.
Step 3: Rewrite the email incorporating your critique.
Output only the final revised version.
```

**Multi-turn version:**

Turn 1: Generate the draft
Turn 2: "What are the three weakest parts of what you just wrote?"
Turn 3: "Now rewrite it fixing those three weaknesses."

**Why this works:** The model's first pass is fast and pattern-driven. The critique pass forces it to evaluate rather than generate — a different cognitive mode that catches errors the generation pass glosses over. The revision pass then has explicit targets to fix, producing a measurably better output than the original.

This technique is especially effective for:
- Long-form writing (articles, reports, proposals)
- Code (generate → identify bugs → fix)
- Arguments (draft → steelman the opposing view → strengthen)
- Plans (create → identify risks → revise)

---

### 6. Sampling and Temperature Control

Temperature and sampling parameters control how much randomness is introduced during token selection. Understanding these gives you a lever that operates independently of your prompt wording.

**Temperature** divides all token scores before converting them to probabilities.

- **Temperature = 0** → fully deterministic. The model always picks the highest-probability token. Same input produces identical output every run. Use for factual extraction, classification, structured data tasks.
- **Temperature = 0.7** → sensible default for most tasks. Some variety, mostly coherent.
- **Temperature = 1.0** → model's raw distribution. More creative, less predictable.
- **Temperature > 1.0** → increasingly chaotic. Useful for brainstorming, creative divergence. Risky for anything requiring accuracy.

**Top-P (Nucleus Sampling)** cuts off the long tail of low-probability tokens. Instead of adjusting scores, it simply removes any token outside the top P% of cumulative probability.

- **Top-P = 0.1** → only the very highest-probability tokens survive. Extremely conservative.
- **Top-P = 0.9** → the standard setting. Rare outliers are excluded but reasonable variety remains.
- **Top-P = 1.0** → no cutoff, all tokens eligible.

**How temperature and top-p interact:**

Think of temperature as adjusting the shape of the probability hill — making it sharper or flatter. Think of top-P as drawing a fence around the top of that hill, excluding everything below a certain height. They work together, and most production systems tune both.

**Practical guidance:**

| Task type | Temperature | Top-P |
|---|---|---|
| Data extraction, classification | 0.0 – 0.2 | 0.1 – 0.5 |
| Summarization, Q&A | 0.3 – 0.5 | 0.7 – 0.9 |
| General writing, emails | 0.5 – 0.7 | 0.9 |
| Creative writing, brainstorming | 0.8 – 1.0 | 0.95 – 1.0 |

**In chat interfaces** like Claude or ChatGPT, you cannot set these directly. But you can approximate the effect through your prompt:

```
Generate ten very different headline options. Prioritize variety
over polish — I want genuinely distinct angles, not variations
on the same idea.
```

This nudges the model toward higher-variance output even without touching temperature settings.

---

### 7. Priming with Partial Outputs

A lesser-known but highly effective technique is beginning the model's response for it. Instead of asking for an output and waiting, you provide the opening tokens of the answer yourself, and the model completes from there.

```
User: Summarize the key risk factors from this report.
Assistant: The three most critical risk factors identified in the report are:
1.
```

By starting the response yourself, you:
- Lock in the format before the model makes any choices
- Prevent preamble ("Sure! Here is a summary of the key risk factors...")
- Force a numbered list structure
- Signal the expected level of formality

This works because the model treats your partial output as already-generated tokens and simply continues the sequence. It cannot "go back" and change what you wrote.

**Practical uses:**
- Forcing JSON to start with `{` — eliminates all preamble
- Starting a list with `1.` — guarantees a numbered list
- Beginning with a specific tone word — locks in voice from the first token

---

### 8. Anchoring with Reference Material

When you need the model to stay close to specific content — a document, a dataset, a set of facts — explicitly anchoring it to that material dramatically improves accuracy and reduces hallucination.

```
Answer the following question using only the information in the
document below. If the answer is not in the document, say
"This information is not available in the provided material."

Document:
[paste your content here]

Question: What were the main causes of the revenue decline in Q2?
```

This technique:
- Prevents the model from filling gaps with invented information
- Forces citations back to the source material
- Makes the model's confidence level visible — it cannot confidently answer what isn't there

The refusal instruction ("say X if not found") is critical. Without it, the model will hallucinate a plausible-sounding answer rather than admit uncertainty.

---

### Putting It All Together

These techniques are not mutually exclusive — the best prompts layer several of them simultaneously. A production-grade prompt for a content moderation system might combine:

- A role (you are a content moderation specialist)
- An audience definition (flagging decisions will be reviewed by a legal team)
- Constraints (classify only, do not explain your reasoning in prose)
- A JSON schema (return structured output only)
- A few-shot example (here is how to handle an ambiguous case)
- An anchoring instruction (base your decision only on the policy document below)
- A refusal condition (if the content does not clearly violate policy, classify as "review required")

Each layer adds a dimension of control. Together they produce output that is consistent, accurate, structured, and safe to use downstream — which is the actual goal of output quality control.
