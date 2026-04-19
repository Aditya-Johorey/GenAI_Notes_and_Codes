# Foundations of Prompt Engineering

Why AI interaction sucks?

* Most people treat a prompt like a Google search - type something in, get something back.
* That mental model is wrong ❌
* A prompt is not a question. It is **a probability steering document**.

LLMS actually dont
-
* **Understand the truth:**
  
  > When you send text to an LLM, the model does not read it, think about it, and reply.
  
* Read databases
* Reason properly
> A prompt is instructions + context, wrapped with constraints and supported by examples, that shape how the model predicts the next token.

what this implies is that:
* A prompt is not communicating meaning to AI, its shaping its statistical distribution.
* AI is incapable of understanding intent - it finds the most statistically plausible continuation of your words.
* Write words that make **the right answer the most likely continuation**, and you get **good output**.
  Write words that leave **many equally plausible continuations**, and you get **inconsistent, generic, or wrong output**.

How LLMs work?
-
* LLMS processes your words as **a sequence of tokens** and uses them to **calculate** which token is **most likely** to come next
  and then the one after that — and so on until it decides to stop.
* **Every single word** you write **shifts those probabilities**.
* A **vague prompt** produces a **wide probability distribution**, meaning the model has many equally valid paths forward.
  A **precise prompt narrows that distribution**, pushing the model toward exactly the output you want.

What LLMS Actually Do?
-
LLMs Dont:
* Understand truth or check facts against a database
* Reason the way humans do
* Remember previous conversations (unless given a memory system)
* Know what you meant — only what you wrote

LLMs do:
* Predict the statistically most probable next token given all previous tokens
* Reflect patterns from their training data extraordinarily well
* Follow structural patterns (JSON, bullet lists, arguments) very reliably
* Generate fluent, confident-sounding text even when factually wrong

_**This is why prompting is engineering, not conversation. You are configuring a system, not talking to a person.**_

Components of a Well-Built Prompt
-
> * A professional prompt is not one long sentence. It is a **structured document made of distinct components**, each doing a specific job. 
> * Omitting any one of them forces the model to guess — and **guessing introduces variance**, which **means inconsistent outputs**.

**Note: Variance describes how spread out that distribution is**
* **Low variance** means the **probability is concentrated**
  * one or two tokens are overwhelmingly likely, and the model picks predictably from a narrow range.
  * You get consistent, deterministic-feeling outputs.
* **High variance** means the **probability is spread** thin across many tokens
  * lots of options have similar scores
  * The model could reasonably go in many directions, so small changes in your prompt, or just running it again, can produce very different outputs.
_**This is directly controlled by temperature**_

A vague prompt introduces artificial variance. The model has no strong signal about which direction to go, so many continuations become equally probable. A precise, well-structured prompt reduces that variance before temperature even comes into play — it makes the right answer statistically dominant, so the model converges on it reliably regardless of sampling settings.
> Prompting is essentially the act of **reducing variance before generation** begins.
> Temperature **controls variance during generation**.


<img width="1440" height="1102" alt="image" src="https://github.com/user-attachments/assets/801d596b-76a5-42a1-ba06-2a7e742eca89" />

### Role
The role tells the model who it should behave as. This is sometimes called **persona conditioning.**
```
You are a senior financial analyst at a hedge fund.
```
#### Why it works?
During pretraining, the model read millions of documents written by, about, and in the voice of financial analysts.

Activating that persona shifts the model's internal probability distribution toward the vocabulary, reasoning style, risk framing, and caution levels associated with that identity.

The role controls:

- **Tone** — formal vs conversational, cautious vs bold
- **Vocabulary** — technical jargon vs plain language
- **Depth of reasoning** — surface-level vs expert-level analysis
- **Risk tolerance** — a doctor hedges differently than a comedian
- **Use of caveats** — a lawyer adds disclaimers a marketer would not

_**Note: A role is not decoration. It is a compression of dozens of implicit instructions into one sentence.**_

#### Why is the Role Important?
- The model defaults to a generic helpful assistant persona — which is a blend of average writing from its training data.
- Useful for casual tasks, but too generic for professional or specialized outputs.

#### Common Mistakes:
* Making the role too vague.
  - `"You are an expert"` gives the model nothing to work with.
  - `"You are a board-certified cardiologist explaining a diagnosis to a worried patient"` activates a very specific behavioral profile.

### Task
> The task is the specific action you want the model to perform. It must be clear, direct, and unambiguous.

```
Write a 3-paragraph executive summary of the attached quarterly report.
```
The task answers: what **exactly** should the model produce?

Weak tasks leave the model to choose its own scope:

- ❌ "Help me with my report" — what kind of help? Summary? Edit? Rewrite?
- ✅ "Rewrite the introduction to be more concise, under 80 words"

Strong tasks include:
- A clear verb (write, summarize, classify, translate, extract, compare)
- A defined scope (one paragraph, five bullet points, a JSON object)
- A specific deliverable (a subject line, a function, a plan)

_**One task per prompt is a useful rule of thumb. When you give the model two tasks in one prompt, it often underperforms on both. Break complex goals into sequential prompts.**_

### Context
> Context provides the background information the model needs to understand **why the task matters** and _**who it's for**_.

```
Context: I am presenting to a board of non-technical executives who
have no AI background. They are skeptical of automation initiatives.
```

#### Context controls:

- **Audience calibration** — who will read this output?
- **Situational relevance** — what constraints exist in the real world?
- **Stakes** — is this a casual blog post or a legal document?
- **Prior knowledge** — what does the audience already know?

#### The hallucination connection
> Many AI hallucinations happen not because the model lacks knowledge, but because the model lacks context and invents a plausible scenario.

**Providing context grounds the model in your specific situation and dramatically reduces invented content.**

#### What should context section contain?
A good context section answers:

- Who is the intended audience?
- What is the purpose of this output?
- What situation is this being used in?
- What does the reader already know?

### Constraints
> Constraints tell the model what it must not do, or strict limits it must stay within. They shrink the solution space.

```
Constraints:
- Maximum 200 words
- No bullet points
- Do not use technical jargon
- Do not recommend specific products
```

Type | Example
--- | ---
Length | "Under 100 words", "exactly 5 bullet points"
Style | "Formal tone", "conversational, no jargon"
Format | "Only plain text, no markdown"
Knowledge | "Do not reference events after 2023"
Ethical | "Do not make claims about competitor products"
Negative | "Do not suggest surgery as a first option"

#### Why constraints are important?
Without constraints, the model optimizes for completeness and fluency — which often means verbose, over-explained output full of hedges and caveats. Constraints force it to make choices, which produces tighter, more useful outputs.

During pretraining, the model read vast amounts of human writing — articles, essays, textbooks, forum answers. The pattern in most of that writing is: say more, cover all angles, hedge your claims, don't leave things out. That is what "good writing" looked like in the training data. THe model has been trained to cover all the knowledge gaps and give a complete outlook on a topic.

So when you give the model a task with no constraints, it defaults to that pattern. It writes the way a cautious, thorough writer would:

- It adds phrases like "it's worth noting that..." and "however, it depends on..."
- It covers edge cases you didn't ask about
- It qualifies every claim with "generally speaking" or "in most cases"
- It wraps up with a summary paragraph restating everything it just said

None of this is wrong — it's just the model doing what got rewarded in training: produce fluent, complete-sounding text.

**Constraints break this default.** When you say "under 80 words", the model cannot pad. It is forced to decide what actually matters and cut everything else. When you say "no caveats", it cannot hedge. It has to commit.

Think of it like this — if you ask someone "explain machine learning" with no limits, they'll talk for ten minutes covering everything. If you say "explain it in two sentences to a 10-year-old", they're forced to find the sharpest possible version of the idea.

**Negative constraints (what not to do)** are often more powerful than positive ones. Telling the model to avoid a specific failure mode directly prevents that failure, whereas hoping it won't happen without instruction is unreliable.

_**Constraints don't restrict quality. They force prioritization — which is where quality actually comes from.**_

### Examples
> Examples are the single most powerful lever for controlling output quality. Showing the model what good output looks like is more effective than describing it.

```
Here is the format I want:

Product: AirPods Pro
Review: "Excellent noise cancellation, but the case scratches easily."
Sentiment: Mixed
Key issue: Durability

Now do the same for: [new review]
```

Examples work because the model is a **pattern-completion engine**. When you **show it a pattern, it extends that pattern**. This is more reliable than explaining the pattern in words, because the model may interpret your words differently than you intend.

```
Positive and negative examples together are especially powerful:
Good response: "Based on the data, sales increased 12% in Q3."
Bad response: "Sales went up a lot recently, which is great!"

The good response cites specific figures. The bad one is vague and adds unnecessary enthusiasm.
```

Showing what to avoid alongside what to produce dramatically reduces the chance of the model drifting toward the bad pattern.

### Output Format
> The output format defines the structure of the response.
This is especially important when the output will be used programmatically — fed into another system, parsed by code, or inserted into a document.

```
Return your answer as a JSON object with exactly these keys:
{
  "summary": "...",
  "risk_level": "low | medium | high",
  "recommended_action": "..."
}
```

Output format controls:

- Structure — prose, JSON, markdown, table, numbered list
- Length — word count, number of items, number of paragraphs
- Order — which sections come first
- Completeness — must all fields be filled in?

When building AI-powered applications or automations, the output format is not optional — it is what makes the output machine-readable.

### Putting It Together: A Complete Prompt
Here is what these six components look like assembled into a real prompt:
```
Role:
You are an enthusiastic startup coach who specializes in helping
first-time founders find clarity and confidence in their ideas.

Task:
Write a motivational one-page business pitch for a young founder
who wants to launch a sustainable water bottle brand.

Context:
The founder is 22 years old, just graduated, has no funding yet,
but has a clear passion for reducing single-use plastic waste.
The pitch will be shared with potential angel investors at a
casual networking event — not a formal boardroom setting.

Constraints:
- Maximum 200 words
- Energetic and optimistic tone
- No corporate jargon
- Do not mention lack of funding or experience as weaknesses
- Focus on vision and impact, not financials

Output format:
Plain text with three short sections:
1. The Problem
2. The Vision
3. The Ask
```

Every component is pulling its weight here:
- The role sets an encouraging, coaching energy. 
- The task is specific — one page, one brand, one founder. 
- The context tells the model exactly who this person is and where the pitch lands. 
- The constraints prevent the model from defaulting to cautious, hedge-heavy business writing. 
- The output format locks in a clean three-part structure so the output is immediately usable.

Categorising Prompts on the Basis of Iterations
-

When you write a prompt, you have a choice: explain what you want in words, or show the model what you want through examples.

The number of examples you provide defines the type of prompting you are doing — and that choice has a direct impact on output quality and consistency.
This is not about how many times you run the prompt. It is about how many examples you include inside the prompt itself before asking the model to do the actual task.

Think of it like teaching someone a new task:

- You could describe the task and hope they get it
- You could show them one example and then ask them to try
- You could walk them through several examples before handing it over

### Zero-Shot Prompting
> Zero-shot prompting means giving the model a task with no examples at all. You describe what you want and trust the model to figure out the rest from its training.

```
Classify the sentiment of this sentence as Positive, Negative, or Neutral.

Sentence: "The delivery was late but the product itself was great."
```
The model has seen enough sentiment classification during pretraining that it can handle this without examples. No demonstration needed.

When it works well:

- Tasks the model has seen countless times in training — summarization, translation, basic classification, brainstorming
- When the output format is simple and universally understood
- When speed matters and you do not have time to craft examples
- Early exploration — when you are figuring out what the model can do

When it breaks down:

- Niche or specialized tasks the model has little training exposure to
- When you have a very specific output style or structure in mind
- When the task involves subtle judgment calls the model cannot infer
- When consistency across many runs matters

**The hidden risk of zero-shot:** The model fills every gap with its own assumptions. If those assumptions match yours, great. If they do not, you get output that is **technically correct** but **not what you wanted** — and **you may not even realize why.**

### One-Shot Prompting
> One-shot prompting gives the model exactly one example before the real task. That single example communicates format, tone, depth, and style far more efficiently than a paragraph of instructions.

```
Convert the following customer complaint into a calm, professional
support response.

Example:
Complaint: "Your app crashed and I lost three hours of work. This is unacceptable."
Response: "We're truly sorry to hear about your experience. Losing work due to
an unexpected crash is incredibly frustrating, and we understand how serious
this is. Our team is actively investigating this issue. In the meantime, please
reach out to support@company.com and we'll make this right."

Now do the same for:
Complaint: "I've been waiting two weeks for a refund and nobody is responding to my emails."
Response:
```

One example showed the model:

- The tone (calm, empathetic, not defensive)
- The structure (acknowledge → validate → action → resolution)
The length (three sentences approximately)
- The vocabulary (professional but human)

None of that was explained in words. The example communicated it all implicitly.

**When it works well:**

- You have a specific voice or format that is hard to describe in words
- The task involves stylistic judgment — tone, register, personality
- You want consistent output across many inputs
- The task is simple enough that one demonstration is sufficient

**When it breaks down:**

- The task has multiple distinct patterns or edge cases one example cannot cover
- The example you choose is unrepresentative — the model overfits to it
- The task requires the model to understand a rule, not just copy a pattern

**The one-shot trap:** If your single example has any quirk — an unusual word, an unexpected structure — the model may treat that quirk as intentional and replicate it everywhere. Choose your example carefully.

### Few Shot Prompting
> Few-shot prompting provides multiple examples — typically between two and five — before the real task. Each additional example teaches the model something new: a pattern, an edge case, a boundary condition.

```
Role: You are a support operations specialist.
Task: Classify each incoming ticket by Category and Priority (1-5).

Priority guide:
- Priority 5: System is completely down, users cannot work
- Priority 3: Feature broken, but a workaround exists
- Priority 1: Cosmetic feedback or minor suggestions

---
Input: "I love the new dashboard design! Could you add a dark mode option?"
Output: Category: Feature Request | Priority: 1 | Reason: Cosmetic improvement, no workflow impact.

---
Input: "The export to PDF button is greyed out. I need it for a client meeting
in an hour — I'm copy-pasting for now."
Output: Category: Functional Bug | Priority: 3 | Reason: Feature broken but workaround exists.

---
Input: "URGENT — our entire team gets a 404 when trying to log in.
Nobody can access the system."
Output: Category: Critical Outage | Priority: 5 | Reason: Complete loss of access for all users.

---
Input: "Hi, I tried resetting my password but the link in the email just
loads a blank white page. I've tried three times."
Output:
```

Three examples taught the model:

- What Priority 1 looks like in practice
- What Priority 3 looks like in practice
- What Priority 5 looks like in practice
- The exact output format to use every time
- How to write a one-line reason

**When it works well:**

- Complex classification or labeling tasks with multiple categories
- Tasks where the pattern is subtle and hard to explain in words
- When output consistency across many runs is critical
- Niche or domain-specific tasks the model has less training exposure to
- When you need to teach the model a custom logic or rubric

**When it breaks down:**

- Your examples are inconsistent with each other — the model gets confused about what the rule actually is
- You use too many examples and bloat the prompt unnecessarily — two good examples beat five mediocre ones
- The examples do not cover the edge cases that actually appear in production

**Choosing the Right Approach**
The decision is not always obvious. Here is a practical framework:

- Start with zero-shot. If the output is good enough, stop there. Adding examples costs tokens and time.
- Move to one-shot when zero-shot output has the right content but the wrong format, tone, or structure. One example is usually enough to fix stylistic problems.
- Move to few-shot when one-shot is inconsistent — when different inputs produce structurally different outputs, or when the task has multiple distinct cases the model needs to distinguish between.

#### The Quality of Examples Matters More Than the Quantity**

A common mistake is thinking more examples automatically means better output. It does not. Three well-chosen examples that each demonstrate something different will outperform six examples that all show the same pattern.

When selecting examples for few-shot prompts, ask:

- Does each example teach the model something the others do not?
- Do my examples cover the range of inputs the model will actually see?
- Are my examples consistent with each other — do they all follow the same underlying rule?
- Would a human, reading only these examples, understand exactly what I want?

If the answer to all four is yes, your examples are ready.

#### Positive and Negative Examples Together

One underused technique is pairing a good example with a bad one and explaining the difference. This is especially effective when the failure mode is subtle.

```
Good output: "Revenue increased 14% year-over-year, driven primarily
by enterprise subscriptions."

Bad output: "Revenue went up a lot this year, which is really exciting
for the whole team!"

The good output cites specific figures and stays objective.
The bad output is vague and adds unnecessary emotional commentary.

Now summarize the following financial result in the same style as
the good output:
```

Showing what to avoid, alongside what to aim for, closes the gap between your intent and the model's interpretation far more effectively than words alone.
