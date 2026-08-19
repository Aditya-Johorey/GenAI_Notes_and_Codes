# Multi-Agent Orchestration: How AI Agents Work Together

**Audience:** Non-technical students already familiar with the basics of AI agents (they know what an agent is, roughly how it perceives/decides/acts, and have probably used one).
**Format:** Single class session, ~75 minutes, lecture + discussion + short design activity.
**Goal:** By the end, students should be able to explain *why* orchestration is needed, describe the main coordination patterns in plain language, and sketch a simple multi-agent system for a real scenario.

---

## Learning Objectives

By the end of this class, students will be able to:
1. Explain why a single agent often isn't enough for complex tasks.
2. Define "orchestration" and name the role an orchestrator plays.
3. Describe at least three coordination patterns (sequential, hierarchical, parallel/decentralized) using real-world analogies.
4. Identify the key design tradeoffs: communication, memory, task splitting, error handling, and human oversight.
5. Sketch a basic multi-agent design for a given scenario.

---

## Timing Overview

| Segment | Time |
|---|---|
| 1. Hook & framing | 10 min |
| 2. Core concepts | 20 min |
| 3. Real-world patterns & examples | 15 min |
| 4. Design tradeoffs | 15 min |
| 5. Group activity | 10 min |
| 6. Wrap-up & takeaways | 5 min |

---

## 1. Hook & Framing (10 min)

**Opening question to the class:**
"You already know what one AI agent can do — book a flight, answer a question, write code. But what happens when the job is too big or too varied for one agent to handle well?"

**Analogy to open with — the kitchen.**
A single home cook can make a meal alone. A restaurant kitchen, by contrast, has a head chef, a line cook for grills, one for sauces, a pastry chef, and an expediter who calls out orders and makes sure everything reaches the table at the same time, hot and correct. No single cook is "smarter" — the kitchen works because roles are divided and someone coordinates the handoffs.

**Bridge to today's topic:**
Multi-agent orchestration is the "expediter" problem for AI: how do you take several specialized agents and combine their work into one reliable outcome?

**Discussion prompt (cold call, 2–3 students):**
"Think of a task you'd never hand to just one person. Why not?" (Expected answers: too much work, needs different skills, needs checking/oversight.) Draw the parallel to agents explicitly.

---

## 2. Core Concepts (20 min)

### Why one agent isn't always enough
Walk through three limits of a single agent, in plain language:
- **Scope limit** — one agent juggling ten different jobs (research, writing, fact-checking, formatting) tends to do all of them adequately rather than any of them well.
- **Context limit** — an agent can only "hold in mind" so much information at once; long, multi-part tasks overflow that.
- **Reliability limit** — a single agent making every decision has no one checking its work; mistakes compound silently.

### What "orchestration" means
Define plainly: **orchestration is the system that decides which agent does what, when, and how their outputs get combined.** The word is deliberate — like a conductor who doesn't play an instrument but ensures the violins, brass, and percussion come in at the right time and stay in balance.

Introduce two roles that recur across systems:
- **Worker agents** — specialists that do one kind of task (e.g., a "search agent," a "coding agent," a "summarizer").
- **Orchestrator (or "manager") agent** — decides task order, routes work, and reconciles results. Sometimes this is itself an AI agent; sometimes it's simpler fixed logic (a "if X then route to Y" script).

**Key distinction to land:** orchestration is a *coordination problem*, not a *capability* problem. You can have excellent individual agents and still get bad results from poor coordination — just like an all-star sports team with no game plan.

---

## 3. Real-World Coordination Patterns (15 min)

Present each pattern with a plain analogy, then a real applied example. Sketch each on the board as a simple diagram of boxes and arrows — no code needed.

### A. Sequential / Pipeline
**Analogy:** an assembly line. Each station does one step, then passes the product down the line.
**Example:** A content-creation pipeline — a research agent gathers facts → a writing agent drafts → an editing agent polishes → a fact-checking agent verifies claims before publishing.
**Tradeoff to flag:** if one station is slow or wrong, everything downstream is delayed or wrong too.

### B. Hierarchical / Manager–Worker
**Analogy:** a film production — a director (manager) breaks the script into scenes and assigns them to different crews (workers), then reviews and assembles the final cut.
**Example:** A "manager" agent takes a complex customer request ("plan my trip and handle the visa paperwork"), splits it into sub-tasks, and dispatches them to a flights agent, a hotels agent, and a documents agent — then merges their answers into one itinerary.
**Tradeoff to flag:** the manager becomes a bottleneck and a single point of failure; if it makes a bad plan, all the workers execute a bad plan well.

### C. Parallel / Decentralized (peer agents)
**Analogy:** a group project where teammates work on separate sections at the same time and then reconcile at the end, without one person bossing the others.
**Example:** Several agents independently research different vendors for a purchasing decision, then a final "debate" or voting step reconciles conflicting recommendations.
**Tradeoff to flag:** faster, but reconciling disagreements between peer agents is genuinely hard — who wins when two agents disagree?

**Quick class check-in:** ask students to match each pattern to a scenario you read aloud (e.g., "a hospital triage system," "a group essay," "a factory line") — thumbs vote, no need to call on individuals.

---

## 4. Design Tradeoffs (15 min)

Frame this as: "Once you decide *who* does *what*, five practical questions decide whether the system actually works."

1. **Communication** — How do agents share information? (Direct messages? A shared "whiteboard" document? Through the orchestrator only?) Analogy: a group chat vs. everyone only talking to the team lead.
2. **Shared memory / context** — What does each agent need to know about what the others have already done, so work isn't duplicated or contradicted?
3. **Task decomposition** — Splitting a big goal into the *right-sized* sub-tasks is itself hard — too coarse and agents get overwhelmed, too fine and coordination overhead eats the benefit.
4. **Failure handling** — What happens when one agent gives a bad or wrong answer? Does the system catch it, retry, or ask a human? Analogy: a hospital's second-opinion process.
5. **Human oversight** — Where does a person check in? Fully autonomous systems are riskier as the stakes rise (compare: an agent drafting a marketing email vs. an agent executing a financial trade).

**Emphasize this line explicitly:** more agents is not automatically better. Every added agent adds coordination cost — more places for miscommunication, more latency, more cost. Good orchestration design is about *minimum sufficient complexity*, not maximum agent count.

---

## 5. Group Activity (10 min)

Split into small groups (3–4 students). Give each group **one** scenario card:

- Automating a small business's customer support inbox
- Planning and executing a company's social media campaign for a week
- Reviewing job applications and scheduling interviews
- Managing a household's grocery shopping and meal planning

**Task (5 min):** On paper or whiteboard, sketch:
- What 2–4 worker agents would you create, and what does each do?
- Which pattern (sequential / hierarchical / parallel) fits best, and why?
- Where would you put a human check-in point, and why there?

**Share-out (5 min):** 2 groups present briefly; instructor highlights good tradeoff reasoning, not just "correct" answers — there often isn't one right design.

---

## 6. Wrap-Up & Key Takeaways (5 min)

Recap in one slide / board summary:

- Orchestration = deciding *who does what, when, and how it's combined* — it's a coordination problem, not just a capability problem.
- Three common patterns: **sequential** (assembly line), **hierarchical** (manager + workers), **parallel** (peers who reconcile).
- Every design has real tradeoffs in communication, shared memory, task-splitting, failure handling, and human oversight.
- More agents ≠ better. The goal is the simplest system that reliably gets the job done.

**Exit question (write on an index card, hand in):**
"Name one task in your own life or work you'd design as hierarchical rather than sequential, and say why."

---

## Glossary (for slide or handout)

- **Agent** — a system that perceives input, makes a decision, and takes action toward a goal.
- **Orchestration** — the coordination layer that routes tasks between agents and combines their outputs.
- **Orchestrator / manager agent** — the agent (or logic) responsible for assigning and sequencing work.
- **Worker / specialist agent** — an agent focused on one narrow task.
- **Pipeline (sequential)** — agents work one after another, each depending on the previous one's output.
- **Hierarchical (manager–worker)** — a manager agent decomposes a task and delegates to workers, then merges results.
- **Parallel / decentralized** — agents work simultaneously and independently, then reconcile.
- **Human-in-the-loop** — a checkpoint where a person reviews or approves before the system proceeds.

---

## Optional Case Studies (for further reading or a follow-up class)

- Customer support triage systems that route tickets between a classification agent, a knowledge-base agent, and an escalation agent.
- Multi-agent coding assistants where a planning agent breaks a feature request into sub-tasks for separate coding and testing agents.
- Research assistant tools where a "lead" agent assigns different sub-questions to search agents and synthesizes their findings.

*(Have students find one current news example of a company using multiple coordinated AI agents, and bring it to the next class — good bridge to a discussion of real deployed systems.)*

---

## Instructor Notes

- Keep all analogies concrete (kitchen, film crew, assembly line, group project) — avoid technical jargon like "API calls," "message passing protocols," or specific frameworks; this audience needs the *concept*, not the implementation.
- The board diagrams (boxes + arrows) do more work than any slide text — draw the three patterns live rather than showing them pre-made.
- Watch the "more agents = better" instinct in the activity; it's the most common design mistake students will propose, and correcting it live is a good teaching moment.
- If time is short, cut the case studies section first — the activity and wrap-up should never be rushed, since that's where the concepts actually get applied.
