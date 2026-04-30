## Advanced Prompting Patterns

### Why These Patterns Matter

The techniques covered in earlier sections — role prompting, chain-of-thought, hallucination control — all operate within a single prompt and a single model response. They make individual outputs better.

Advanced prompting patterns operate at a different level. They are about **architecture** — how you connect prompts together, how you give models access to external tools and information, how you coordinate multiple models working in parallel, and how you build systems that can operate autonomously over extended tasks.

This is where prompt engineering stops being about crafting clever instructions and starts being about designing intelligent systems. The patterns in this section are the building blocks of every serious AI application being built today — from customer service agents to research assistants to automated data pipelines.

---

### 1. Retrieval-Augmented Generation (RAG)

RAG is the most important pattern for building reliable, up-to-date, domain-specific AI applications. It solves the two biggest limitations of raw LLMs in one architecture: knowledge cutoffs and hallucination risk.

**The core idea:**

Instead of asking the model to answer from memory, you first retrieve relevant information from an external source, inject that information into the prompt as context, and then ask the model to reason over that context to produce an answer.

```
[User question]
     ↓
[Search your knowledge base for relevant content]
     ↓
[Inject retrieved content into the prompt]
     ↓
[Model reasons over retrieved content]
     ↓
[Grounded, accurate answer]
```

The model becomes a reasoning engine over your data, not a knowledge oracle drawing on uncertain memory.

**The prompting layer of RAG:**

Most explanations of RAG focus on the retrieval infrastructure — vector databases, embedding models, similarity search. But the prompting layer is equally important and often poorly designed.

```
System: You are a helpful assistant for Acme Corporation.
Answer questions using only the context provided below.
If the answer is not in the context, say:
"I don't have that information in our knowledge base.
Please contact support@acme.com for further help."
Do not use any knowledge outside the provided context.

Context:
[Retrieved document chunks inserted here]

User: [Question]
```

**What makes a good RAG prompt:**

- **Explicit grounding instruction** — tell the model to use only the provided context
- **Explicit refusal instruction** — tell the model what to say when the answer is not in the context
- **Source attribution** — instruct the model to indicate which part of the context it is drawing from
- **Tone and persona** — define how the model should communicate, not just what it should say

**RAG prompt with source attribution:**

```
Answer the user's question using only the documents provided.
After your answer, list the document titles you used as sources.
If multiple documents contributed to the answer, list all of them.
If the answer is not in the documents, say so clearly.

Documents:
[Document 1 — Title: Product Manual v3.2]
[content]

[Document 2 — Title: FAQ Updated June 2024]
[content]

User question: [question]
```

**Chunking strategy and its effect on prompting:**

How you split your documents before storing them affects what gets retrieved and therefore what the model sees. Large chunks give more context but retrieve less precisely. Small chunks retrieve precisely but may miss surrounding context. The prompting implication: always instruct the model to acknowledge when retrieved context seems incomplete.

```
The context below may be excerpts from larger documents.
If the provided context seems incomplete or cut off,
say so and answer only what the context clearly supports.
```

**Hybrid RAG — combining retrieval with model knowledge:**

Sometimes you want the model to use both retrieved content and its own training knowledge, with clear separation:

```
Answer the question using the provided documents as your
primary source. You may supplement with your own knowledge
only for general background information that helps interpret
the documents. Clearly distinguish between information from
the documents and information from your own knowledge.
Use [Document] and [General knowledge] tags to label each.
```

---

### 2. Tool Use Prompting (Function Calling)

Tool use is what transforms a language model from a text generator into an agent that can interact with the world. Instead of answering entirely from text, the model can decide to call external functions — search the web, run calculations, query a database, send an email, create a calendar event — and use the results to inform its response.

**The mental model:**

The model reads your prompt and, instead of immediately generating a text response, it outputs a structured instruction to call a specific tool with specific parameters. Your system executes that tool call and returns the result. The model then reads the result and either calls another tool or generates a final response.

**Designing tool use prompts:**

```
You are a helpful assistant with access to the following tools:

1. search_web(query: string) → returns top search results
   Use when: the user asks about current events, recent information,
   or anything that may have changed since your training.

2. get_weather(location: string, date: string) → returns weather forecast
   Use when: the user asks about weather conditions.

3. calculate(expression: string) → evaluates a math expression
   Use when: the user asks for numerical calculations.

4. send_email(to: string, subject: string, body: string) → sends an email
   Use when: the user explicitly asks to send an email.

Rules:
- Use a tool whenever it would produce a more accurate answer
  than relying on your own knowledge.
- Never guess at information a tool could retrieve accurately.
- After receiving a tool result, incorporate it naturally
  into your response — do not just repeat the raw output.
- Ask for clarification before calling send_email —
  confirm the recipient and content with the user first.
```

**The decision logic prompt:**

A critical part of tool use prompting is teaching the model when to use tools and when not to:

```
Before responding, decide:
- Can I answer this accurately from my own knowledge? → Answer directly
- Would a tool produce a more accurate or current answer? → Use the tool
- Does this require real-time data (prices, weather, news)? → Always use a tool
- Is this a calculation with specific numbers? → Always use the calculator
- Am I uncertain about any fact in my answer? → Use search to verify
```

**Chaining tool calls:**

Many real tasks require multiple tool calls in sequence, where each call's output informs the next:

```
User: Book me a flight from Mumbai to Delhi next Friday,
      and add it to my calendar.

Model reasoning:
1. search_flights(origin="Mumbai", destination="Delhi", date="next Friday")
   → Returns available flights

2. get_user_preferences() → Returns preferred airline, seat preference

3. book_flight(flight_id="...", passenger_details="...")
   → Returns booking confirmation

4. create_calendar_event(title="Flight to Delhi", date="...", details="...")
   → Confirms calendar entry

5. Generate final response summarizing what was booked
```

**Safety constraints for tool use:**

When tools have real-world consequences — sending emails, making purchases, deleting files — always build confirmation steps into the prompt:

```
Before executing any action that cannot be undone
(sending emails, making purchases, deleting data),
summarize what you are about to do and ask the user
to confirm with "yes" before proceeding.
```

---

### 3. Prompt Chaining

Prompt chaining is the practice of breaking a complex task into a sequence of prompts where the output of each prompt becomes the input to the next. Each prompt in the chain does one thing well, and the chain as a whole accomplishes something no single prompt could do reliably.

**Why chaining beats single complex prompts:**

A single prompt asking for everything at once forces the model to juggle multiple sub-tasks simultaneously — and it underperforms on all of them. Chaining lets each prompt do one focused job, producing higher quality at every stage.

**A basic chain structure:**

```
Prompt 1 — Research:
"Given this topic, identify the five most important questions
a reader would want answered. Output only the questions as a list."

Output → feeds into →

Prompt 2 — Outline:
"Given these five questions: [output from Prompt 1]
Create a structured outline for an article that answers all of them.
Output only the outline."

Output → feeds into →

Prompt 3 — Draft:
"Using this outline: [output from Prompt 2]
Write the full article. Follow the outline exactly.
Aim for 800 words."

Output → feeds into →

Prompt 4 — Edit:
"Edit this article for clarity, concision, and flow:
[output from Prompt 3]
Remove any padding. Tighten every paragraph.
Output the revised article only."
```

**Conditional chaining:**

Chains do not have to be linear. You can build decision points where the next prompt depends on the output of the current one:

```
Prompt 1: Classify this customer message as:
          - Complaint → route to Prompt 2A
          - Question → route to Prompt 2B
          - Compliment → route to Prompt 2C

Prompt 2A (Complaint handler):
"The customer has a complaint: [message]
Write an empathetic response that acknowledges the issue,
apologizes sincerely, and offers a concrete resolution."

Prompt 2B (Question handler):
"The customer has a question: [message]
Search the knowledge base and provide a clear, direct answer."

Prompt 2C (Compliment handler):
"The customer sent a compliment: [message]
Write a warm, genuine thank-you response that does not feel
corporate or scripted."
```

**Transformation chains:**

Chains are especially powerful for transforming content through multiple stages:

```
Raw data → Clean and structure → Analyze → Summarize → Format for report
```

Each stage does one transformation cleanly. The final output is the product of five focused operations, not one sprawling prompt trying to do everything.

**Error handling in chains:**

Robust chains include validation steps between stages:

```
After each prompt in the chain, run a validation prompt:
"Does this output meet the requirements of the next stage?
Requirements: [list requirements]
If yes, output: PASS
If no, output: FAIL — [specific reason]"

If FAIL → re-run the previous prompt with additional constraints
If PASS → proceed to the next stage
```

---

### 4. Memory Patterns

LLMs have no memory between conversations. Every new session starts from zero. Memory patterns are techniques for giving models persistent context — making them aware of history, preferences, and prior decisions across multiple interactions.

**Why memory matters for workflows:**

Without memory, every interaction with an AI assistant is like meeting a stranger. With memory, the model can build on previous conversations, remember user preferences, track ongoing projects, and maintain consistency across sessions.

**Four types of memory:**

**In-context memory** — the simplest form. You include relevant history directly in the prompt:

```
Previous conversation summary:
- User is building a Python FastAPI application
- We have already set up authentication using JWT
- The next task is implementing the database layer with PostgreSQL
- User prefers async code and type hints throughout

Current request: [new message]
```

In-context memory is simple and reliable but limited by the context window. Long histories get truncated.

**External memory** — relevant past information is stored in a database and retrieved when needed, similar to RAG:

```
[Retrieve relevant past interactions based on current query]
[Inject retrieved memory into prompt as context]
[Model responds with awareness of relevant history]
```

This scales to unlimited history but requires retrieval infrastructure.

**Summary memory** — instead of storing raw conversation history, you periodically summarize it into a compressed representation:

```
After every 10 exchanges, run this prompt:
"Summarize the key decisions, preferences, and context
from the conversation so far in under 200 words.
Focus on information that would be useful in future interactions."

Store this summary and inject it at the start of future sessions.
```

**Structured memory** — store specific facts about the user or project in a structured format that gets updated as new information emerges:

```
User profile (updated automatically):
{
  "name": "Priya",
  "role": "Product Manager",
  "current_project": "Mobile app redesign",
  "preferences": {
    "communication_style": "direct and concise",
    "output_format": "bullet points for action items"
  },
  "decisions_made": [
    "Chose React Native over Flutter",
    "Target launch: Q3 2025"
  ]
}
```

The model reads this at the start of each session and updates it at the end.

---

### 5. Multi-Agent Prompting

Multi-agent systems use multiple AI models — or multiple instances of the same model — each with a specific role, working together to accomplish a task that is too complex for a single model to handle well.

**Why multiple agents beat one:**

A single model trying to be a researcher, writer, critic, and fact-checker simultaneously underperforms on all four. Specialized agents each doing one job produce better results — and they can check each other's work.

**The basic multi-agent pattern:**

```
Agent 1 — Researcher:
"Your only job is to find and summarize relevant information
on the following topic. Do not write, analyze, or evaluate.
Only research and present findings.
Topic: [topic]"

Agent 2 — Writer:
"Using only the research provided below, write a first draft.
Do not research or fact-check — only write.
Research: [output from Agent 1]"

Agent 3 — Critic:
"Review the draft below. Your only job is to find weaknesses:
logical gaps, unsupported claims, unclear passages, poor structure.
Do not rewrite — only critique.
Draft: [output from Agent 2]"

Agent 4 — Editor:
"Revise the draft based on the critique below.
Address every point raised. Output the final version only.
Draft: [output from Agent 2]
Critique: [output from Agent 3]"
```

**Debate pattern — multiple agents with opposing views:**

```
Agent A — Advocate:
"You are arguing strongly in favor of [position].
Make the strongest possible case. Do not acknowledge weaknesses."

Agent B — Skeptic:
"You are arguing against [position].
Identify every flaw, risk, and counterargument you can find."

Agent C — Synthesizer:
"You have read the arguments for and against [position]:
For: [Agent A output]
Against: [Agent B output]
Produce a balanced, nuanced assessment that incorporates
the strongest points from both sides."
```

This pattern produces more balanced, rigorous analysis than asking a single model to "consider multiple perspectives" — because each agent is fully committed to its role.

**Verification pattern — agents checking each other:**

```
Agent 1 — Generator:
"Answer the following question as accurately as possible."

Agent 2 — Verifier:
"You are a fact-checker. Review the answer below for:
- Factual accuracy
- Logical consistency
- Unsupported claims
Flag every issue you find. Be ruthless.
Answer to verify: [Agent 1 output]"

Agent 3 — Resolver:
"An answer was generated and then fact-checked.
Revise the answer to address every issue the fact-checker raised.
Original answer: [Agent 1 output]
Fact-check findings: [Agent 2 output]"
```

**Orchestrator pattern:**

For complex workflows, one agent acts as an orchestrator that breaks down the task and delegates to specialist agents:

```
Orchestrator prompt:
"You are managing a team of specialist agents to complete
the following project: [project description]

Your available agents are:
- Research Agent: finds and summarizes information
- Writing Agent: produces drafted content
- Code Agent: writes and reviews code
- Review Agent: checks quality and accuracy

Break this project into tasks. For each task, specify:
1. Which agent should handle it
2. What input they need
3. What output they should produce
4. In what order tasks should be completed"
```

---

### 6. The Meta-Prompt Pattern

A meta-prompt is a prompt that generates other prompts. Instead of manually crafting prompts for every task, you describe what you need and let the model design the optimal prompt for you.

```
You are an expert prompt engineer.
I need to accomplish the following task: [describe your task]
The output will be used for: [describe the downstream use]
The model that will receive this prompt is: [model name]

Design the optimal prompt for this task. Include:
- Role specification
- Clear task definition
- Relevant constraints
- Output format specification
- Any examples that would improve reliability

Output only the prompt itself, ready to use.
```

**Meta-prompts for iteration:**

```
Here is a prompt I have been using: [your prompt]
Here is the output it produces: [example output]
Here is what is wrong with the output: [specific problems]

Redesign the prompt to fix these problems while preserving
what is working. Explain what you changed and why.
```

**Why meta-prompts matter for non-technical users:**

For people who are not professional prompt engineers, meta-prompts lower the barrier significantly. Instead of knowing how to craft a perfect prompt from scratch, you describe your goal in plain language and let the model translate that into a well-structured prompt. You then review and refine the generated prompt rather than building it from zero.

---

### 7. The Skeleton-of-Thought Pattern

Skeleton-of-thought is a technique where the model first generates a structural outline — the skeleton — and then fills in each section in parallel or sequence. This produces more coherent long-form outputs than generating everything in a single pass.

```
Step 1 — Generate the skeleton:
"Create a detailed outline for [document type] on [topic].
Include main sections and 2-3 bullet points per section
indicating what each should cover. Output only the outline."

Step 2 — Flesh out each section:
"Using the outline below as your structure, write the full
content for Section [X] only. Do not write other sections.
Be thorough and specific for this section.
Outline: [full outline]
Write Section [X]:"

[Repeat Step 2 for each section]

Step 3 — Integrate:
"Below are individually written sections for a [document type].
Edit them into a coherent whole. Ensure smooth transitions
between sections, consistent tone throughout, and remove
any repetition.
[All sections concatenated]"
```

**Why this beats single-pass generation for long documents:**

Single-pass generation of long content drifts — the model loses track of its opening arguments by the time it reaches the conclusion, tone shifts between sections, and later sections sometimes contradict earlier ones. The skeleton-of-thought approach gives the model a map before it starts writing, and each section is written with the full map in mind.

---

### 8. The Persona Ensemble Pattern

Persona ensemble uses multiple expert personas within a single prompt to analyze a problem from different angles simultaneously — without the overhead of separate agent calls.

```
Analyze the following business decision from three perspectives:

As a CFO: Focus on financial risk, cash flow implications,
and return on investment. Be conservative and numbers-focused.

As a CMO: Focus on brand impact, customer perception,
market positioning, and growth opportunity.

As an Operations Director: Focus on implementation complexity,
resource requirements, timeline, and operational risk.

After all three perspectives, synthesize a recommendation
that balances all three viewpoints.

Decision to analyze: [describe the decision]
```

This is lighter than a full multi-agent setup and works well for decisions that genuinely benefit from multiple professional lenses without requiring separate model calls.

---

### 9. Dynamic Few-Shot Selection

Standard few-shot prompting uses the same fixed examples for every query. Dynamic few-shot selection chooses examples at runtime based on what the current query is most similar to — producing more relevant demonstrations for each specific case.

**The concept in a prompt:**

```
I am going to give you a query and a bank of examples.
Select the two examples from the bank that are most similar
to the query, and use only those as demonstrations.

Example bank:
[Example 1: query type A]
[Example 2: query type B]
[Example 3: query type A]
[Example 4: query type C]
[Example 5: query type B]

Current query: [query]

First, identify which examples are most relevant to this query.
Then use those examples as the pattern for your response.
```

In production systems, dynamic few-shot selection is typically handled programmatically — embedding the query, finding the most similar examples in a vector database, and injecting only those examples into the prompt. But the prompting principle is the same: relevance beats quantity.

---

### Putting the Patterns Together

These patterns are not alternatives to choose between — they are layers that combine into complete AI systems.

A production customer support system might combine:

- **RAG** to ground responses in product documentation
- **Tool use** to check order status in a live database
- **Memory** to remember the customer's previous interactions
- **Prompt chaining** to route different query types through specialized handlers
- **Multi-agent verification** to check responses before sending
- **Conditional logic** to escalate to a human when confidence is low

Each pattern solves a specific problem. Together they produce a system that is accurate, consistent, context-aware, and safe to deploy at scale.

The skill of advanced prompt engineering is knowing which patterns to reach for, how to combine them, and how to design the connections between them — which is exactly what building real AI workflows requires.
