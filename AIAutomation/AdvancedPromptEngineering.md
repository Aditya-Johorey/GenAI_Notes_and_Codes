# Topic 1: Advanced Prompt Engineering for Automation
### AI Automation Course — Full Lesson Content

---

> **Topic overview**
> Your students already know how to write a prompt. This topic upgrades that skill for a completely different context: not a one-off chat with an AI, but a repeatable, automated system that runs without you watching it. By the end of this topic, students will write prompts that work the 1,000th time as reliably as the first.

**Estimated teaching time:** 3–4 hours
**Format:** Teach lesson → Short exercise → Move to next lesson
**Prerequisite:** Basic–intermediate prompt writing (assumed)

---

## Lesson 1: Why your current prompts won't survive automation

### What we're covering
The difference between a "chat prompt" and an "automation prompt" — and why that difference matters enormously.

---

### Lesson content

You've probably had this experience: you write a prompt in ChatGPT or Claude, it gives you a great answer, and you feel like a prompt engineering pro. Then you try to use that same prompt inside an automated workflow — and suddenly it starts giving you inconsistent, unpredictable results.

**This isn't bad luck. It's a structural problem.**

When you're chatting with an AI directly, you are the safety net. If the AI goes off-track, you correct it. If it misunderstands, you clarify. If it gives a weird format, you ask it to redo it. You are constantly in the loop, steering the output.

When you build an automation, you step out of that loop entirely. The prompt runs at 3am. It handles 500 inputs at once. There's no one to correct it when it goes sideways. The prompt itself has to do all the steering — every single time.

This is the fundamental shift in mindset this topic is about: **moving from prompts that work once to prompts that work reliably at scale.**

Think of it like the difference between giving directions to someone standing next to you versus writing printed instructions for someone who has never met you, in a country where English isn't the first language, and who will follow your instructions literally — no matter how strange the outcome seems. Your instructions need to be airtight before you hand them over.

---

### The three failure modes of automation prompts

**Failure Mode 1: Vagueness**
"Write a summary of this email."
→ Sometimes 2 sentences. Sometimes 10. Sometimes a bullet list. Sometimes a paragraph. Your downstream workflow breaks because it expected a specific format.

**Failure Mode 2: Missing constraints**
"Reply to the customer."
→ The AI might apologise when it shouldn't, promise a refund when you didn't authorise one, or write in a tone that doesn't match your brand. You never said it couldn't — so it did.

**Failure Mode 3: No error handling**
"Extract the invoice amount from this document."
→ What if the document is a blank page? What if the amount is in a foreign currency? What if there are two amounts? A chat prompt ignores these edge cases. An automation prompt needs to handle them explicitly.

---

> 💡 **Deep-dive callout: The reliability gap**
> Research on LLM behaviour in production systems consistently shows that small changes in input — a slightly different email, a document with an unusual format, an unexpected language — can cause significant changes in output if the prompt isn't built to absorb variation. The technical term for this is *prompt brittleness*. Building prompts for automation means designing for brittleness from the start.

---

### ✏️ Exercise 1
Take a prompt you currently use regularly (in ChatGPT, Claude, or any AI tool). Write it down exactly as you use it.

Now identify:
- Where is it vague? (Where could the AI interpret it differently each time?)
- What constraints are missing? (What could it do that you haven't explicitly forbidden?)
- What edge cases does it ignore? (What unusual inputs would break it?)

You don't need to fix it yet. Just diagnose it.

---

## Lesson 2: Instruction Hierarchy — building prompts with layers

### What we're covering
How to structure your prompts with a clear chain of command so the AI always knows what matters most.

---

### Lesson content

Here's something most people don't realise: when you write a prompt, the AI isn't reading it the way you'd read a paragraph. It processes everything at once and has to figure out, from context, which instructions are most important.

If your prompt is a long block of text with instructions buried inside examples buried inside caveats — the AI will get confused about what to prioritise. And when an AI is confused about priorities, it starts making its own decisions about what matters. That's dangerous inside an automation.

**Instruction Hierarchy** is a way of layering your prompt so that priority is always clear. Think of it like an organisational chart: the senior instructions sit at the top and override everything below them. Junior instructions fill in the details but can never contradict what's above.

---

### The four layers of a well-structured automation prompt

**Layer 1 — Role and purpose (who the AI is and why)**
This anchors the entire prompt. It tells the AI what kind of entity it's supposed to be in this context and what its job is.

*Example:*
> "You are a customer support assistant for a UK-based software company. Your only job is to help customers resolve technical issues with our product. You do not give pricing advice, make promises about refunds, or discuss competitors."

Notice what this does: it defines scope immediately. The AI knows what it is, what it does, and — critically — what it doesn't do.

**Layer 2 — Task instruction (what to do with this specific input)**
This is the actual job for this particular automation. It comes after the role so the AI already knows who it is before being told what to do.

*Example:*
> "Read the customer email below. Identify the main issue they are experiencing. Then write a reply that acknowledges their problem, provides a step-by-step solution if possible, or escalates to a human agent if the issue is beyond your scope."

**Layer 3 — Format constraints (what the output must look like)**
This is where most people forget to be specific — and where automations break.

*Example:*
> "Your reply must:
> - Begin with a greeting using the customer's name if provided
> - Be no longer than 150 words
> - End with: 'If this doesn't resolve your issue, reply to this email and a human agent will assist you within 24 hours.'
> - Never use bullet points in the reply itself"

**Layer 4 — Edge case handling (what to do when things go wrong)**
This is the layer that separates professional automations from amateur ones.

*Example:*
> "If the email is not in English, reply in the same language as the customer. If you cannot confidently identify the customer's issue, do not guess — instead reply asking them to describe the problem in more detail. If the email appears to be spam or irrelevant, respond with exactly: SKIP"

---

> 💡 **Deep-dive callout: Why "SKIP" matters**
> Notice that last instruction — telling the AI to respond with exactly the word SKIP for irrelevant emails. This is *structured output design*. When your prompt runs inside a workflow tool like Make or Zapier, the next step in the workflow can check for the word SKIP and route that email differently — perhaps into a spam folder. This is how prompts and workflows talk to each other. A prompt that gives unpredictable outputs breaks the workflow downstream. A prompt that gives reliable, predictable outputs is the foundation of everything.

---

### ✏️ Exercise 2
Take the prompt you diagnosed in Exercise 1. Rewrite it using the four-layer structure:
- Layer 1: Role and purpose
- Layer 2: Task instruction
- Layer 3: Format constraints
- Layer 4: Edge case handling

Keep each layer clearly separated. You can use a blank line between them.

---

## Lesson 3: System Constraints — writing prompts that hold their shape

### What we're covering
How to set boundaries that stop an AI from drifting, improvising, or doing things you haven't authorised — especially when the automation runs unattended.

---

### Lesson content

Imagine hiring a contractor to paint your living room. You say "paint it white." You come home and half the room is white, half is cream, and they've also repainted the ceiling because they thought it looked dull. They were trying to be helpful. But they went beyond what you asked.

AI models do the same thing. They are trained to be helpful, and "helpful" sometimes means they'll add things you didn't ask for, make assumptions to fill gaps, or soften your instructions when they think it sounds better. In a conversation, you can redirect this. In an automation, it accumulates silently.

**System constraints are the guardrails that keep an AI within the boundaries you've set**, even when it might naturally want to wander.

---

### Types of constraints to build into your prompts

**Behavioural constraints** — what the AI must and must not do
> "Do not apologise more than once in a single reply."
> "Never offer discounts or refunds unless the customer email explicitly contains the word 'refund'."
> "Do not add information that is not present in the source document."

**Scope constraints** — what topics or actions are off-limits
> "Only respond to questions about our product. If the customer asks about anything unrelated, politely tell them this is outside your scope."
> "Do not generate content about competitors under any circumstances."

**Format constraints** — hard rules about structure
> "Always output valid JSON. Never include explanatory text outside the JSON structure."
> "Respond only in the language the input is written in."
> "Your output must be exactly three sentences. Not two. Not four."

**Tone and voice constraints** — how the AI should communicate
> "Write in a warm, direct tone. Avoid corporate jargon. Never use the phrase 'I apologise for any inconvenience.'"
> "Match the reading level of a 14-year-old. Use short sentences. Avoid technical terminology."

---

> 💡 **Deep-dive callout: Positive vs negative constraints**
> Research into prompt design shows that telling an AI what *to do* is generally more reliable than telling it what *not to do* — especially for complex tasks. "Reply in plain English" tends to work better than "Don't use jargon." However, for safety-critical automation (anything involving money, legal statements, or sensitive data), negative constraints are essential as a second layer. Use both: tell the AI what to do, then explicitly forbid the most dangerous deviations.

---

### The "test the edges" mindset

Once you've written your constraints, mentally throw the worst possible inputs at your prompt:
- What if the input is in a different language?
- What if it's empty?
- What if it contains offensive content?
- What if it's completely irrelevant to the task?
- What if it's ambiguous — could mean two different things?

Every one of these scenarios will eventually arrive inside a live automation. The question is whether your prompt handles them gracefully or falls apart.

---

### ✏️ Exercise 3
Design a system-constrained prompt for one of the following scenarios (pick the one closest to your work):

**Option A:** An AI that reads incoming job applications and categorises them as Strong / Review / Not Suitable based on criteria you define.
**Option B:** An AI that reads customer reviews and extracts: sentiment (positive/negative/neutral), main topic, and a one-sentence summary.
**Option C:** An AI that takes raw meeting notes and produces a structured summary with: decisions made, actions assigned, and open questions.

Your prompt must include all four layers from Lesson 2 and at least three explicit constraints from this lesson.

---

## Lesson 4: Iteration and Evaluation — how to know if your prompt is actually working

### What we're covering
A repeatable process for testing, scoring, and improving prompts — because guessing isn't good enough when automations run at scale.

---

### Lesson content

Most people test a prompt like this: run it once, it looks okay, ship it. Then two weeks later something breaks and they can't figure out why.

Professional prompt engineering — especially for automation — requires a more structured approach. You need to know not just "does this work?" but "how reliably does this work, across a wide range of real inputs?"

This is called **prompt evaluation**, and it's one of the most important skills in this entire course.

---

### The prompt testing loop

**Step 1 — Build a test set**
Collect 10–20 real examples of the inputs your automation will receive. Don't cherry-pick the easy ones. Deliberately include:
- Normal inputs (the typical case)
- Edge case inputs (unusual, ambiguous, or incomplete)
- Adversarial inputs (ones designed to trip up the AI — offensive content, off-topic messages, inputs in unexpected languages)

**Step 2 — Define what "correct" looks like**
Before running any tests, write down the ideal output for each input in your test set. This is your benchmark. Without this, you're just reading outputs and hoping they feel right — which is not evaluation.

**Step 3 — Run the prompt against every test input**
Do this methodically. Don't just run the obvious ones. The edge cases are where your automation will fail in production.

**Step 4 — Score each output**
For each test input and its output, score it on a simple scale:
- ✅ Correct — output matches expected output
- ⚠️ Partially correct — right direction, wrong format or missing detail
- ❌ Wrong — incorrect, off-topic, or broken format

**Step 5 — Identify patterns in failure**
If 3 out of 10 outputs are wrong, look at *what they have in common*. Is it a format issue? A scope issue? An edge case you didn't anticipate? Patterns tell you exactly which part of your prompt needs adjustment.

**Step 6 — Change one thing at a time**
This is critical. When you find a problem, resist the urge to rewrite the whole prompt. Change one specific thing — add a constraint, clarify an instruction, add an edge case handler — and retest. If you change five things at once, you won't know which one fixed it (or broke something else).

---

> 💡 **Deep-dive callout: The 90% rule**
> A commonly used benchmark in professional automation is that a prompt should produce acceptable output on at least 90% of realistic test inputs before it goes into production. Below that threshold, the automation will generate enough errors to create more work than it saves. For high-stakes automations (anything that sends communications, moves data, or triggers payments), raise that bar to 95%+. The remaining exceptions should be routed to a human review step — which you'll learn how to build in Topic 5.

---

### ✏️ Exercise 4
Take the prompt you built in Exercise 3. Create a mini test set of 6 inputs:
- 2 normal inputs
- 2 edge case inputs
- 2 adversarial inputs

Run each through your prompt (in ChatGPT, Claude, or whatever AI tool you use). Score each output. Identify the most common failure. Write one specific change to your prompt to address that failure. Retest.

---

## Lesson 5: Program-Aided Language Models (PAL) — making AI work with logic, not guesswork

### What we're covering
How to get AI to delegate calculation and logic to external tools rather than doing it in its head — where it often gets it wrong.

---

### Lesson content

Here's something worth knowing about how AI models work: they're very good at language, but they're not calculators. When you ask an AI to work out a complex multi-step calculation — say, "calculate the total cost of this order after tax, discount, and shipping" — it's not actually computing the answer. It's *predicting* what the answer should look like based on patterns in its training data. Most of the time it's right. Sometimes it's convincingly, confidently wrong.

This is a serious problem in automations that handle numbers, dates, logic conditions, or any step-by-step reasoning where errors compound.

**Program-Aided Language Models (PAL)** is the approach that solves this. Instead of asking the AI to calculate the answer itself, you ask it to *describe the steps needed* — and then hand those steps off to a tool that actually executes the logic precisely.

---

### The PAL principle in plain English

Think of it like this: you have a brilliant analyst who understands business problems deeply and can structure any problem into a clear plan — but they're bad at arithmetic. You don't fire them. You pair them with a calculator. They say what to calculate; the calculator does the calculating.

PAL is the same idea. The AI understands the problem and figures out *what logic to apply*. A separate tool (a spreadsheet formula, a code interpreter, a calculator step in your workflow tool) executes that logic with precision.

---

### How this applies to no-code automation (without writing any code)

In no-code tools like Make or n8n, this principle shows up naturally through the way you chain steps:

**Without PAL thinking:**
> Prompt: "Read this sales report and calculate which product had the highest profit margin this month."
→ AI does all the maths in its head. Errors possible.

**With PAL thinking:**
> Step 1 — Prompt: "Read this sales report. Extract the revenue and cost figures for each product into a structured table. Output as JSON with fields: product_name, revenue, cost."
> Step 2 — Workflow tool: Use a formula step to calculate profit margin for each product (revenue minus cost, divided by revenue).
> Step 3 — Prompt: "Here is a table of products and their profit margins: [table]. Write a one-paragraph summary identifying the top performer and any notable trends."

The AI handles language and structure. The workflow tool handles arithmetic. The result is accurate every time.

---

### Where PAL thinking applies in your automations
- Any automation involving numbers, percentages, or financial figures
- Any automation that needs to compare dates or deadlines
- Any automation that routes decisions based on conditions ("if score is above X, do Y")
- Any automation that ranks or scores multiple items

---

> 💡 **Deep-dive callout: The original research**
> PAL was formally defined by researchers at Carnegie Mellon University in 2022. Their finding: by having the LLM generate structured reasoning steps rather than final answers, and delegating execution to a Python interpreter, accuracy on mathematical and logical tasks improved dramatically — outperforming models several times larger that tried to reason in plain text. The principle translates directly to no-code workflows: let AI do what it's good at (understanding, structuring, summarising), and let deterministic tools do what they're good at (calculating, comparing, routing).

---

### ✏️ Exercise 5
Look at the automation use case you've been building throughout this topic. Identify one step where the AI is being asked to do logic or calculation that could instead be handled by your workflow tool.

Redesign that step using the PAL principle:
1. What does the AI extract or structure?
2. What does the tool calculate or route?
3. What does the AI do with the result?

Write the three-step breakdown.

---

## Lesson 6: Meta-Prompting — using AI to improve your own prompts

### What we're covering
How to use AI as a partner in building and refining your prompts — turning a slow manual process into a fast feedback loop.

---

### Lesson content

At this point in the topic, you've learned how to build a properly layered, constrained, testable prompt. There's one more technique that will dramatically speed up your workflow as an automation builder.

**Meta-prompting** means using an AI to help you write, evaluate, and improve prompts.

If normal prompting is "give me the answer," meta-prompting is "help me build a better question." You're not asking the AI to do a task — you're asking it to think about how to do the task more effectively. This is particularly powerful because AI models have been trained on enormous amounts of prompt engineering content. They know common failure patterns. They can spot ambiguity faster than most humans. And they can generate and test multiple prompt variations in seconds.

---

### Three ways to use meta-prompting in your practice

**Use 1: Prompt generation**
Describe what you want your automation to do and ask the AI to draft a prompt for it.

*Example meta-prompt:*
> "I'm building an automation that reads incoming customer support emails and produces a structured output with three fields: issue category, urgency level (high/medium/low), and a one-sentence summary. The automation runs without any human in the loop. Write me a robust prompt for this step that includes a role definition, clear task instructions, output format, and edge case handling."

The AI will give you a strong starting draft. Your job is then to review, test, and refine it — not accept it blindly.

**Use 2: Prompt critique**
Paste your existing prompt and ask the AI to identify weaknesses.

*Example meta-prompt:*
> "Here is a prompt I'm using in an automated workflow: [paste your prompt]. Analyse it for weaknesses. Specifically: where is it ambiguous, what edge cases does it not handle, and where might it produce inconsistent outputs at scale? Give your analysis as a bullet list."

This is like having an expert peer reviewer for every prompt you write.

**Use 3: Prompt variation testing**
Ask the AI to generate multiple versions of a prompt with different approaches, so you can test which performs best.

*Example meta-prompt:*
> "Write three different versions of a prompt that extracts action items from meeting notes. Version 1 should use a strict numbered list format. Version 2 should use a JSON structure. Version 3 should use a conversational paragraph. I'll test all three and pick the one that works best in my workflow."

---

> 💡 **Deep-dive callout: The prompt library habit**
> Professional automation builders maintain a prompt library — a document or database where they store every tested, working prompt along with its use case, test score, and version history. Meta-prompting accelerates the building of this library. When you've used AI to generate, critique, and refine a prompt until it scores 90%+ on your test set, that prompt becomes a reusable asset. A freelance automation builder with a well-stocked prompt library can build new client automations in hours rather than days — because the hard thinking has already been done.

---

### ✏️ Exercise 6
Take the full prompt you've built throughout this topic. Run it through a meta-prompt critique:

1. Paste it into Claude or ChatGPT with this instruction:
   *"Analyse this automation prompt for weaknesses. Where is it ambiguous? What edge cases does it miss? What constraints should be added? Give specific suggestions."*

2. Review the critique. Implement the two most important suggestions.

3. Rerun your test set from Exercise 4 with the updated prompt. Did the score improve?

---

## Topic 1 Capstone Project

**Project title: The Reliable Prompt Stack**

---

## Topic Summary — What students now know

| Concept | What it means in practice |
|---|---|
| Automation vs chat prompts | Prompts in workflows need to be airtight — no human in the loop to correct them |
| Instruction Hierarchy | Layer your prompts: role → task → format → edge cases |
| System Constraints | Define hard limits on what the AI can and cannot do |
| Iteration & Evaluation | Test against a real test set; score, identify patterns, improve one thing at a time |
| PAL thinking | Let AI structure the problem; let tools execute the logic |
| Meta-Prompting | Use AI to draft, critique, and improve your own prompts |

---

*Next topic: Agentic Orchestration — how AI agents plan, reason, and act across multiple steps*
