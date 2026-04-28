## Reasoning & Thinking Control

### Why Reasoning Control Matters

LLMs do not think before they speak. They speak in order to think.

This is the single most important thing to understand about reasoning in language models. When a model generates a response, it does not silently work through the problem and then write the answer. It generates tokens sequentially — and each token it produces becomes context that influences what comes next. The "thinking" and the "speaking" are the same process.

This has a profound implication: **the more reasoning tokens you give the model before it commits to an answer, the better that answer will be.** Every technique in this section exploits this property in a different way. They all do the same thing at a fundamental level — give the model more generated context to reason with before locking in a final response.

When you ask a model a hard question and demand an immediate answer, you are asking it to jump directly to the highest-probability completion of your prompt — which is often a pattern match, not genuine inference. When you ask it to reason step by step first, you are forcing it to generate intermediate tokens that constrain and guide what comes next. The answer that follows that reasoning is measurably more accurate.

---

### 1. Chain-of-Thought Prompting

Chain-of-thought (CoT) is the foundational reasoning technique. It is also the simplest: you ask the model to show its work before giving an answer.

The original formulation from research was almost embarrassingly simple:

```
Think step by step.
```

Adding those four words to a prompt reliably improves accuracy on multi-step problems — math, logic, planning, analysis — because it prevents the model from pattern-matching straight to a plausible-sounding answer.

**Why it works mechanically:** When the model writes "Step 1: First I need to identify...", those words become tokens in the context. The next step it generates is conditioned on those tokens — it cannot contradict what it just wrote without incoherence. The reasoning chain acts as a set of rails that guide the model toward a consistent conclusion.

**Without chain-of-thought:**
```
User: A bat and a ball cost $1.10 together. The bat costs $1 more
than the ball. How much does the ball cost?

Model: The ball costs 10 cents.
```

This is wrong. The intuitive pattern-match answer is 10 cents but the correct answer is 5 cents. The model jumped to the most statistically common response without actually solving the problem.

**With chain-of-thought:**
```
User: A bat and a ball cost $1.10 together. The bat costs $1 more
than the ball. How much does the ball cost? Think step by step.

Model: Let me call the ball's price X.
The bat costs $1 more than the ball, so the bat costs X + $1.
Together they cost $1.10, so: X + (X + $1) = $1.10
2X + $1 = $1.10
2X = $0.10
X = $0.05

The ball costs 5 cents.
```

The reasoning chain forced the model to build the algebra before committing to a number.

**Visible vs hidden chain-of-thought:**

In a chat interface, showing the reasoning is fine — often it is valuable. But in production systems, you often want clean outputs without exposing the internal reasoning process. You can instruct the model to reason internally but output only the conclusion:

```
Think through this problem step by step before answering.
Once you have reasoned it through, output only the final answer
with no explanation.
```

Or split it across two prompts — one to reason, one to extract the conclusion.

**When to use it:**
- Multi-step math or logic problems
- Complex classification with multiple criteria
- Any task where the answer depends on getting intermediate steps right
- Debugging — when you need to understand why the model reached a conclusion

**When not to use it:**
- Simple factual retrieval where reasoning adds nothing
- When you are charged by the token and need efficiency
- When the reasoning itself would confuse or mislead the end user

---

### 2. Zero-Shot Chain-of-Thought vs Few-Shot Chain-of-Thought

There are two flavors of chain-of-thought prompting and they serve different purposes.

**Zero-shot CoT** simply appends a reasoning instruction to your prompt:

```
Solve this problem. Think step by step.
```

No examples provided. The model figures out what "step by step" means for this task on its own. This works surprisingly well for many problems and requires minimal prompt engineering effort.

**Few-shot CoT** provides complete worked examples — problem, reasoning chain, and answer — before the real task:

```
Example 1:
Problem: If a train travels 60 miles per hour for 2.5 hours,
how far does it travel?
Reasoning: Distance = speed × time. Speed = 60 mph, time = 2.5 hours.
60 × 2.5 = 150 miles.
Answer: 150 miles.

Example 2:
Problem: A store sells apples for $0.50 each. If I buy 8 apples
and pay with a $5 bill, how much change do I receive?
Reasoning: Cost = 8 × $0.50 = $4.00. Change = $5.00 - $4.00 = $1.00.
Answer: $1.00.

Now solve this:
Problem: A car uses 1 gallon of fuel every 32 miles. How many
gallons does it need to travel 200 miles?
Reasoning:
```

The worked examples teach the model both the reasoning format and the level of detail expected. Few-shot CoT consistently outperforms zero-shot CoT on harder problems, at the cost of a larger prompt.

**When to use which:**

| | Zero-shot CoT | Few-shot CoT |
|---|---|---|
| Setup effort | Minimal | Medium |
| Works best for | Medium-complexity problems | Hard, multi-step problems |
| Token cost | Low | Higher |
| Consistency | Good | Better |

---

### 3. Task Decomposition and Stepwise Prompting

Complex tasks fail when approached as a single prompt because the model tries to do everything at once and does everything shallowly. Decomposition breaks the task into an ordered sequence of sub-tasks, each of which gets the model's full attention.

**The failure mode of single-prompt complexity:**

```
Write a complete market analysis for a new electric scooter
startup, including competitive landscape, customer personas,
pricing strategy, go-to-market plan, and financial projections.
```

The model will produce something for each of these — but each section will be shallow, generic, and disconnected from the others. It is optimizing to cover the ground, not to go deep on any of it.

**Decomposed version:**

```
We are building a market analysis for a new electric scooter startup.
Complete these steps in order. Do not move to the next step until
the current one is finished.

Step 1: Ask me five clarifying questions about the business before
proceeding. Wait for my answers.

Step 2: Based on my answers, identify and describe three distinct
customer personas with specific demographics, behaviors, and needs.

Step 3: Analyze the competitive landscape. Identify four existing
competitors and compare them across price, range, target market,
and key differentiators.

Step 4: Recommend a pricing strategy with justification based on
the personas and competitive landscape you identified.

Step 5: Draft a 90-day go-to-market plan with specific actions,
not general advice.
```

Each step produces a complete, deep output that becomes input context for the next step. The final result is dramatically better than a single-prompt attempt.

**Why this works:**
- The model goes deep at each stage rather than spreading thin across all stages
- Each sub-output constrains and informs the next — the personas from Step 2 shape the competitive analysis in Step 3
- Human steering is built in — you can redirect at any step if the output is off
- Errors are caught early and corrected before they propagate

**Within a single prompt:**

You can also decompose within one prompt using explicit sequencing instructions:

```
First, identify the three core assumptions in the argument below.
Then, for each assumption, assess whether it is well-supported
or questionable.
Finally, write a one-paragraph rebuttal targeting the weakest assumption.
Do not skip ahead — complete each step before moving to the next.
```

The instruction "do not skip ahead" matters. Without it, the model sometimes collapses steps together and produces shallower output.

---

### 4. Self-Consistency Prompting

Self-consistency is a technique where you ask the model to solve the same problem multiple times using different approaches, then identify the most consistent answer across those attempts.

```
Solve the following problem using three completely different
reasoning approaches. Show your work for each approach separately.
Then compare your three answers and identify which one you are
most confident in and why.

Problem: [your problem here]
```

**Why this works:** Any single reasoning chain can go wrong — a flawed assumption early on propagates through the entire chain. When you run multiple independent chains, random errors in one chain are unlikely to appear in all others. The answer that appears most consistently across chains is most likely to be correct.

This is the LLM equivalent of asking three experts to independently solve a problem and then comparing their answers — not because any one of them is unreliable, but because consensus is more trustworthy than a single opinion.

**Where it shines:**
- Mathematical problems where different solution paths should converge
- Logic puzzles with multiple valid solution approaches
- Risk assessment — different frameworks should identify the same major risks
- Strategic decisions — different analytical lenses should point toward the same conclusion

**Where it is overkill:**
- Simple factual questions
- Creative tasks where multiple valid answers exist by design
- Time-sensitive situations where the cost of multiple reasoning chains is too high

**A practical shortcut:** In multi-turn conversations, you can approximate self-consistency by asking the model to challenge its own first answer:

```
Now argue against the conclusion you just reached.
What is the strongest case for the opposite position?
```

If the model cannot construct a strong counterargument, the original answer is more reliable. If it easily dismantles its own reasoning, the answer needs revision.

---

### 5. Self-Critique and Iterative Reflection

Self-critique is a structured generate → evaluate → revise loop. After producing an initial output, you prompt the model to evaluate that output against specific criteria, then revise based on the evaluation.

**Basic three-step pattern:**

```
Step 1 — Generate:
Write a first draft of [task].

Step 2 — Critique:
Review the draft you just wrote. Identify:
- The three weakest or most generic parts
- Any claims that lack sufficient support
- Anything that could be misinterpreted

Step 3 — Revise:
Rewrite the draft, specifically addressing each weakness
you identified. Output only the final revised version.
```

**Why this outperforms a single-pass prompt:** Generation and evaluation are different cognitive operations. The generation pass optimizes for fluency and completeness — it produces something that sounds right. The evaluation pass applies explicit criteria — it identifies what is actually wrong. These two operations improve each other when sequenced.

A model that generates and evaluates in the same pass compromises both. Separating them lets each step do its job fully.

**Targeted critique prompts:**

You can make the critique step more powerful by specifying exactly what to look for:

```
Review the proposal you just wrote through the eyes of a
skeptical CFO. What would they object to? What numbers would
they question? What risks did you understate?
```

```
Read your explanation as if you are a complete beginner
encountering this topic for the first time. Where would
you get confused? What terms did you use without defining?
What did you assume the reader already knows?
```

```
Check your code for: off-by-one errors, unhandled edge cases,
missing error handling, and any assumptions about input format
that might not hold.
```

The more specific the critique criteria, the more useful the revision.

**Iterative reflection across multiple turns:**

For high-stakes outputs, you can run multiple rounds:

Turn 1: Generate draft
Turn 2: Critique for content accuracy
Turn 3: Revise based on critique
Turn 4: Critique for tone and clarity
Turn 5: Final revision

Each pass targets a different quality dimension, producing a final output that has been pressure-tested from multiple angles.

---

### 6. Least-to-Most Prompting

Least-to-most prompting solves complex problems by starting with the simplest possible version and layering on complexity progressively. Each simpler version becomes the foundation for the next level.

This mirrors how good teachers actually explain difficult concepts — they do not start with the full complexity, they build up to it.

**Structure:**

```
Let's solve this step by step, starting simple.

First, solve this simpler version of the problem: [simplified version]
Now use that result to solve: [slightly harder version]
Now use that result to solve the full problem: [original problem]
```

**Example — building up a programming concept:**

```
Step 1: Write a function that returns the largest number
from a list of three numbers.

Step 2: Now generalize it to work with a list of any length.

Step 3: Now modify it to return the top three largest numbers,
not just the largest one.

Step 4: Now make it work efficiently on a list of ten million numbers.
```

Each step's output is the starting point for the next. The model never has to make the full leap from simple to complex — it makes a series of small, manageable steps.

**Where it works best:**
- Mathematical induction-style problems
- Programming tasks that increase in complexity
- Teaching and explanation — building understanding progressively
- Any problem where the full complexity is too much to handle in one jump

---

### 7. ReAct (Reason + Act)

ReAct is a prompting framework designed for **agentic tasks** — situations where the model needs to interact with external tools, search the web, run code, or query databases in order to answer a question. It was introduced in a 2022 paper by researchers at Princeton and Google.

The core insight is that language models perform far better on real-world tasks when they can interleave reasoning with action, rather than reasoning in isolation or acting blindly without reasoning.

**The three primitives of ReAct:**

Every step in a ReAct loop is one of three types:

**Thought** — the model's internal reasoning about the current state. What does it know? What is it missing? What should it do next? This is never sent to a tool — it only shapes the next action.

**Action** — a call to an external tool. This could be a web search, a calculator, a database query, a code interpreter, an API call — anything with a defined input and output.

**Observation** — the result returned by the tool, appended to the context. The model now has new information to reason with.

The loop continues until the model's Thought step concludes it has enough information to produce a final answer.

**A ReAct prompt template:**

```
You are an agent that can use tools to answer questions.

Available tools:
- search(query) → returns top web results for the query
- calculator(expression) → evaluates a mathematical expression
- lookup(term) → returns a definition or factual summary

For each step, output in this exact format:
Thought: [your reasoning about what to do next]
Action: [tool name and input]
Observation: [the tool's result — this will be filled in by the system]

When you have enough information, output:
Final Answer: [your complete response]

Question: What is the GDP of France divided by its population,
and how does that compare to Germany?
```

**What a ReAct loop looks like in practice:**

```
Thought: I need France's GDP and population to calculate GDP per capita,
then the same for Germany to compare.
Action: search("France GDP 2024")
Observation: France's GDP in 2024 was approximately $3.1 trillion.

Thought: Now I need France's population.
Action: search("France population 2024")
Observation: France's population is approximately 68 million.

Thought: GDP per capita for France = $3.1T / 68M ≈ $45,588.
Now I need Germany's figures.
Action: search("Germany GDP and population 2024")
Observation: Germany GDP: $4.5 trillion, Population: 84 million.

Thought: Germany GDP per capita = $4.5T / 84M ≈ $53,571.
I have everything I need.
Final Answer: France's GDP per capita is approximately $45,600,
compared to Germany's $53,600 — meaning Germany's per-capita
output is about 17% higher than France's.
```

**Why ReAct outperforms pure reasoning or pure tool use:**

Pure chain-of-thought (no tools) hallucinates facts the model does not know. The model will confidently invent a GDP figure rather than admit it does not have current data.

Pure tool use (no reasoning between calls) fires tools randomly and cannot integrate results meaningfully — it has no strategy for what to search for next.

ReAct combines both: the Thought step decides which tool to call and why, and the Observation updates the reasoning rather than being discarded. The model has a strategy, executes it, and updates based on what it finds.

**Key limitations to understand:**

- **Context window pressure** — each Thought-Action-Observation triple consumes tokens. Long loops can hit the model's context limit.
- **Error propagation** — a bad Observation early in the loop can misdirect all subsequent reasoning. Garbage in, garbage out still applies.
- **No backtracking** — standard ReAct is linear. If the model goes down a wrong path, it cannot undo previous actions. Extensions like Tree of Thoughts add branching capability.
- **Tool quality is a ceiling** — the agent is only as good as the tools available to it. A great reasoning loop with a bad search tool produces bad answers.

---

### 8. Tree of Thoughts

Tree of Thoughts (ToT) extends chain-of-thought by allowing the model to explore **multiple reasoning paths simultaneously** rather than committing to a single linear chain. It treats problem-solving as a search through a tree of possible reasoning steps.

```
Consider three different approaches to solving this problem.
For each approach:
1. Outline the first two steps
2. Evaluate whether this approach is promising or likely to fail
3. If promising, continue; if not, abandon it

After exploring all three, select the most promising path
and complete the solution.
```

**Why this matters:** Standard chain-of-thought is linear — if the model takes a wrong turn at step 2, every subsequent step is built on that error. Tree of Thoughts lets the model recognize dead ends and backtrack, just as a human expert would try one approach, realize it is not working, and try something else.

**Where it is most valuable:**
- Complex planning problems with multiple possible approaches
- Creative challenges where the first idea is rarely the best
- Mathematical proofs where the right proof strategy is not obvious
- Strategic decisions where different frameworks lead to different conclusions

**The tradeoff:** Tree of Thoughts generates significantly more tokens than linear chain-of-thought. It is a precision tool for hard problems, not a default approach.

---

### 9. Constraint-First Reasoning

Constraint-first reasoning lists all rules, requirements, and boundaries **before** beginning to reason about the problem. This forces the model into a bounded solution space from the start, rather than generating a solution and then checking whether it violates constraints afterward.

**Standard approach (constraint-last):**
```
Write a product recommendation for noise-cancelling headphones.
The recommendation must be under 100 words, avoid brand names,
and not make any price claims.
```

**Constraint-first approach:**
```
Before beginning, note these hard constraints:
- Maximum 100 words
- No brand names
- No price claims
- Focus only on sound quality and comfort

With those constraints firmly in mind, write a product
recommendation for noise-cancelling headphones.
```

The difference is subtle but meaningful. In the constraint-last version, the model begins generating and then tries to honor constraints it read at the end. In the constraint-first version, the constraints shape the generation from the very first token.

**When it matters most:**
- Legal or compliance-sensitive outputs where violations are costly
- Outputs with hard length limits that must not be exceeded
- Any task where constraint violations require a full rewrite rather than a minor edit

---

### 10. Socratic Prompting

Socratic prompting instructs the model to guide the user toward an answer through questions, rather than simply providing the answer directly. The name comes from Socrates, whose teaching method consisted of relentless questioning rather than lecturing — on the belief that people already hold the seeds of knowledge and the teacher's job is to draw it out through dialogue.

**The key insight:** The act of answering a question changes how you think — more than being told the answer does. When a model asks "where in your code is the variable being modified?", you trace the variable yourself. That tracing is the learning. Being told "your increment is in an unreachable branch" skips the tracing entirely and produces surface-level understanding that fades quickly.

**Prompt patterns that trigger Socratic behavior:**

```
Before answering, ask me one question to understand my actual goal.
```

```
Do not tell me the answer. Guide me to it through questions.
```

```
Ask me what I already know about this topic before explaining it.
```

```
After I respond, challenge any assumptions I am making.
```

```
Once I have solved this, ask me to reflect on what I learned.
```

**The five modes of Socratic prompting:**

**Clarifying** — surface what the user actually needs before answering. Prevents the failure mode of answering a well-formed question that is not the real question.

```
User: Help me improve my writing.
Model: Before I suggest anything, can you tell me what kind of writing
you want to improve, and what you feel is currently holding it back?
```

**Guided discovery** — the model never gives the answer. It only asks questions that narrow the user's thinking one step at a time until they arrive themselves. Powerful for debugging, math, and logic.

```
User: My for-loop is not working.
Model: What output are you currently getting, and what did you expect?
[after answer] Where in the loop do you think the value is being changed?
[after answer] What happens if you print the variable at the start of each iteration?
```

**Challenging assumptions** — targets the premises behind a claim or decision. The model does not argue — it asks what supports the belief.

```
User: I need to rewrite our entire codebase in a new language.
Model: What specific problems are you experiencing with the current codebase
that you believe a language change would solve?
```

**Concept building** — checks what the user already knows before layering in new ideas. Avoids the over-explanation trap (wasting time on things they know) and the under-explanation trap (jumping past their gap).

```
User: Explain machine learning to me.
Model: To make sure I pitch this at the right level — what is your
current understanding of how traditional software works?
```

**Reflection** — happens after a task is complete. Asking the user to articulate what happened and why consolidates a solved problem into a transferable principle.

```
Model: Now that you have fixed the bug — what was the root cause,
and how would you catch this kind of issue earlier next time?
```

**When to use Socratic prompting vs when not to:**

Use it when the goal is learning, decision quality, or eliciting the user's own reasoning. The process of getting to the answer matters as much as the answer itself.

Do not use it when the user just needs a fast, accurate answer. "What is the capital of France?" should not be met with "What do you already know about European capitals?" The technique only has value when understanding — not just information — is the goal.

---

### Choosing the Right Technique

These techniques are not competing alternatives — they are complementary tools. A sophisticated prompt might combine several:

```
[Constraint-first] Here are the rules for this analysis: ...

[Task decomposition] Complete these three steps in order:
Step 1: ...
Step 2: ...
Step 3: ...

[Chain-of-thought] For each step, show your reasoning before
giving your conclusion.

[Self-critique] After completing all three steps, identify
the weakest part of your analysis and revise it.
```

The right combination depends on the task:

| Task type | Recommended techniques |
|---|---|
| Multi-step math or logic | Chain-of-thought, Self-consistency |
| Complex planning | Task decomposition, Tree of Thoughts |
| Tool use and research | ReAct |
| Teaching and learning | Socratic prompting, Least-to-most |
| High-stakes writing | Self-critique, Iterative reflection |
| Constrained outputs | Constraint-first reasoning |
| Hard problems with no obvious approach | Tree of Thoughts, Self-consistency |

The underlying principle across all of them remains the same: **LLMs think by speaking. Give them more tokens to think with before they commit to an answer, and the answer gets better.**
