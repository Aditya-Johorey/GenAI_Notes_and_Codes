# TIER 1 — FOUNDATION
## AI Agent Development for Non-Coders | Weeks 1–2

**Goal of this Tier:** By the end of these two weeks, every student — regardless of technical background — should be able to explain what makes something an "agent" instead of an "automation," describe the five parts every agent needs, and write a system prompt that gives an AI a real job.

---

# TOPIC 1: Agents vs. Automations
### The Mental Shift from "Trigger → Action" to "Reason → Decide → Act"

## Why We Start Here

Your students already know n8n. That's actually a problem we need to solve before it becomes a habit. Most people who learn automation first will try to build an "agent" that's really just a fancier automation — and it will look impressive, but it won't be able to handle anything unexpected. We need to break that pattern on Day 1.

## The Core Idea, In Plain Language

Think about the difference between a **vending machine** and a **shop assistant**.

- A **vending machine** (this is an automation) works like this: you press B4, and it *always* gives you the item in slot B4. It doesn't think. It doesn't know if you're thirsty or hungry, if the drink is expired, or if you actually wanted something else. It just follows the same fixed steps every single time. Press this button → get this exact result. No variation, no judgment.

- A **shop assistant** (this is an agent) works differently. You say, "I want something cold and not too sweet." The assistant *thinks*: What do we have that's cold? Which of those aren't too sweet? Is the customer possibly diabetic — should I ask? They make a judgment call based on the situation, and they can handle a request nobody trained them on word-for-word.

**An automation follows a path. An agent makes a decision.**

## Bringing This Back to n8n (Bridging Their Existing Knowledge)

Your students have built n8n workflows before, so use that as the bridge:

| n8n Automation (What They Know) | AI Agent (What We're Building Toward) |
|---|---|
| "If a new email arrives, send a Slack message" | "Read this email, decide if it's urgent, and respond appropriately — or escalate to a human if unsure" |
| Fixed steps that run the same way every time | Steps that change depending on the situation |
| Breaks if something unexpected happens (a missing field, a weird input) | Can reason through the unexpected and still produce a sensible outcome |
| You (the builder) made every decision in advance | The AI makes some decisions *in the moment*, based on instructions you gave it |

**Key teaching line to repeat often:**
> "In automation, YOU decide everything ahead of time. In an agent, you decide the *boundaries*, and the AI decides *within* them."

## A Simple Real-World Example to Use in Class

**Automation version (n8n style):**
"When a customer fills out our contact form, send them a template email that says 'Thanks, we'll respond in 24 hours.'"
→ Every single person gets the exact same email. No matter what they wrote.

**Agent version:**
"When a customer fills out our contact form, read what they wrote. If it's a simple question you can answer (like store hours or return policy), answer it immediately in a warm and helpful tone. If it's a complaint or something you're unsure about, forward it to a human with a one-line summary of the issue."

Ask your class: *What's different here?* Guide them to notice:
1. The agent **reads and understands** before acting (automation doesn't "understand" anything)
2. The agent **makes a choice** between two different paths based on judgment, not a fixed rule
3. The agent knows **its own limits** ("if unsure, forward to a human") — this is a concept called an *escalation path*, and it's one of the most important safety habits in agent building. We'll return to this in Tier 3.

## Classroom Analogy Bank (Use Whichever Lands With Your Group)

- **Recipe vs. Chef** — A recipe (automation) gives exact steps: 2 cups flour, bake at 350°F. A chef (agent) can taste the sauce and adjust the seasoning without being told exactly how much salt to add.
- **GPS with Fixed Route vs. GPS with Live Traffic Rerouting** — One repeats the same instructions no matter what. The other reroutes based on what's actually happening on the road *right now*.
- **A Form Letter vs. A Personal Assistant** — A form letter says the same thing to everyone. A personal assistant reads the situation and responds differently to different people.

## Common Confusion to Watch For

Non-technical students sometimes think "AI-powered" automatically means "it's an agent." Clarify:

> "Just because ChatGPT wrote the message doesn't make it an agent. If you're still telling it exactly what to say every single time with no room for judgment, that's still just an automation — you've just used AI to write the *content* instead of making the *decision*."

Technical students sometimes swing the other way and think agents need to be "fully autonomous" with no rules. Clarify:

> "A good agent isn't a robot with no rules — it's an employee with clear boundaries and the freedom to use judgment inside them."

## In-Class Activity (15–20 minutes)

Break students into small mixed-ability groups (pair a technical student with a non-technical one). Give each group a business scenario card:

- A restaurant taking reservations
- A freelance graphic designer handling client inquiries
- An HR team fielding employee questions about leave policy
- A small e-commerce store answering shipping questions

**Task:** Write one version of the task as an *automation* (fixed steps) and one version as an *agent* (judgment-based). No AI tools needed yet — just paper or a doc. This trains the *thinking* before the *building*.

---

# TOPIC 2: Anatomy of an Agent
### The Five Parts Every Agent Needs — Brain, Memory, Tools, Persona, Trigger Loop

## Why This Topic Matters

This is the "checklist" topic. Once students internalize these five parts, they'll be able to look at *any* no-code tool — n8n, Voiceflow, Relevance AI, Custom GPTs — and immediately understand what they're looking at, because every single agent-building tool is really just giving you a different interface for the same five ingredients.

## The Analogy: Building a New Employee

Tell your students to imagine they're hiring and training a brand-new employee who has never worked at their company before. What does that employee need before they can do the job well?

1. **Intelligence** — the ability to think and respond (their brain)
2. **Job Description & Personality** — what's their role, how should they behave (their persona/instructions)
3. **Notes & History** — remembering what happened yesterday, or five minutes ago (their memory)
4. **Access to Do Things** — a company email, a company laptop, access to the CRM (their tools)
5. **A Reason to Start Working** — do they wait by the phone? Check email every hour? Start when a customer walks in? (their trigger)

This is *exactly* what an AI agent needs. Let's go through each one.

---

### 1. The Brain (The LLM — Large Language Model)

**In simple terms:** This is the "thinking engine" — the actual AI model (like GPT-4, Claude, or Gemini) that reads information and generates a response.

**What to teach non-technical students:**
> "You don't need to know how this works under the hood — same way you don't need to know how a car engine works to drive one. What you DO need to know is that different 'brains' have different strengths: some are faster and cheaper but less capable of complex reasoning, some are slower and more expensive but much smarter. Choosing the right brain for the job is a decision you'll make in every project."

**Practical framing for later lessons:** In Tier 3 (Cost & Token Economics), they'll learn to actually choose between models. For now, they just need to know: *the brain is swappable, and picking the brain is one of the decisions you make when building the agent.*

---

### 2. Persona & Instructions (Who the Agent Is)

**In simple terms:** This is the written explanation of who the agent is, what its job is, how it should speak, and what it should never do.

**Analogy:** This is the *job description and training manual* combined. If you hired a new receptionist and gave them zero instructions, they wouldn't know if they should be formal or casual, whether to make jokes, or what to do if someone asks something outside their job.

We will cover this in deep detail in Topic 3 (this is the "programming" of a no-code agent) — for now, just introduce it as one of the five ingredients.

---

### 3. Memory (What the Agent Remembers)

**In simple terms:** Memory is what allows an agent to remember things — either within one single conversation, or across many conversations over time.

**Two types to teach clearly, with an analogy:**

- **Short-term memory** = like a conversation with a stranger on a train. They remember what you said five minutes ago in *this* conversation, but once you get off the train, it's gone.
- **Long-term memory** = like your family doctor, who has your full medical history on file and remembers your visit from two years ago, even though you haven't spoken since.

**Why this matters practically:** An agent with only short-term memory will greet a returning customer like a total stranger every time. An agent with long-term memory can say "Welcome back — last time you asked about our return policy, did that get resolved?" This massively changes how "smart" the agent *feels* to a user, even if the underlying brain is exactly the same.

*(We go deep on how to actually build this — no-code — in Tier 2, Topic 6.)*

---

### 4. Tools (What the Agent Can DO, Not Just Say)

**In simple terms:** Tools are the actions an agent is allowed to take in the real world — sending an email, updating a spreadsheet, booking a calendar slot, looking something up online.

**Analogy:** Without tools, an agent is like a very smart person locked in a room with no phone, no computer, and no way to leave — they can only talk to you through a slot in the door. Tools are what let the agent actually reach out and *do* something, not just describe what someone else should do.

**Important distinction to hammer home:**
> "An agent that can only chat is a chatbot. An agent that can chat AND take action — send that email, update that spreadsheet, book that meeting — is what makes it genuinely useful instead of just impressive."

*(We build actual tool connections hands-on in Tier 2, Topic 5.)*

---

### 5. Trigger / Loop (What Starts the Agent, and What Keeps It Going)

**In simple terms:** This is what makes the agent spring into action — and whether it does its job once and stops, or keeps checking and working continuously.

**Examples to use in class:**
- A **message-based trigger**: the agent wakes up when someone sends it a message (like a chatbot on a website)
- A **schedule-based trigger**: the agent wakes up every morning at 9 AM to check for new orders
- A **event-based trigger**: the agent wakes up whenever a new row is added to a spreadsheet, or a new form is submitted

**Analogy:** This is like asking, "How does this employee know when to start working — do they wait for the phone to ring, check email every hour, or is there someone knocking on the door?"

Your students already understand this concept well from n8n — n8n workflows always start with a "trigger" node. Point this out explicitly:
> "This is the one part of agent-building you already have real experience with. It works exactly the same way."

---

## The Full Checklist (Give This as a Handout)

Whenever a student looks at ANY agent-building tool, teach them to ask these five questions:

1. **Brain** — Which AI model is powering this, and is it the right one for the task?
2. **Persona** — What instructions define who this agent is and how it behaves?
3. **Memory** — Does it need to remember just this conversation, or remember people/data over time?
4. **Tools** — What real-world actions does it need to be able to take?
5. **Trigger** — What starts it working, and does it run once or keep going?

**Teaching tip:** Have students apply this checklist to something familiar and non-technical — like Amazon's Alexa, or a customer service chatbot they've used on a website. This makes the abstract concept concrete before they build anything themselves.

---

# TOPIC 3: System Prompt & Persona Design
### The "Programming" of a No-Code Agent

## Why This Is the Most Important Skill in the Entire Course

Here's the framing to give students on Day 1 of this topic, word for word if useful:

> "In traditional software, you program a computer with code. In no-code AI agent building, you 'program' the agent with *words*. Your system prompt IS your code. If you get sloppy with it, your agent will behave unpredictably — the exact same way sloppy code produces bugs. The good news? You already know how to write clear instructions in English. That skill translates directly."

This reframes prompt writing from "a soft skill" to "the actual technical skill of this course" — which matters especially for non-technical students who may otherwise feel like they're not doing "real" building.

## What Is a System Prompt, Really?

**Simple definition:** A system prompt is a set of instructions given to the AI *before* any conversation starts, that shapes everything about how it behaves — invisible to the end user, but governing every response.

**Analogy: The Actor and the Script**

Imagine hiring an actor for a play. Before they walk on stage, you give them a briefing:
- "You're playing a strict but fair headteacher."
- "You care deeply about the students but you don't tolerate excuses."
- "You never break character, even if the audience asks you something weird."
- "If someone asks you about something outside the play's story, redirect them back to the scene."

The actor then improvises *within* that character for the whole performance. The system prompt is that briefing. The AI is the actor. Every response it gives during the conversation is "in character" based on what you told it beforehand.

## The Five Building Blocks of a Strong System Prompt

Teach students to structure every system prompt with these five components. This gives non-technical students a repeatable formula instead of a blank page.

### Block 1: Role (Who Are You?)
Tell the AI exactly what role it's playing.

*Weak example:* "You are a helpful assistant."
*Strong example:* "You are a friendly, knowledgeable customer support agent for a small independent bookstore called Chapter & Verse."

**Teaching point:** Vague roles produce vague, generic-sounding responses. Specific roles produce responses that actually sound like they belong to *this* business.

### Block 2: Tone & Personality (How Do You Sound?)
Describe the communication style clearly enough that two different people reading it would picture the same voice.

*Weak example:* "Be professional."
*Strong example:* "Speak warmly and casually, like a knowledgeable friend, not a corporate script. Use simple language. Avoid sounding robotic or overly formal. It's okay to use light humor occasionally."

**Classroom exercise:** Have students describe a person they know well (a friend, relative, favorite teacher) in 3–4 sentences focused purely on *how that person talks*. Then have them realize: that's exactly the kind of detail a tone instruction needs.

### Block 3: Boundaries (What Should You NEVER Do?)
This is the safety net — and the piece non-technical students most often forget, because in everyday communication we rarely have to spell out what NOT to do.

*Examples to give:*
- "Never make promises about refunds — always direct refund questions to a human team member."
- "Never discuss competitor products."
- "Never provide medical, legal, or financial advice, even if asked directly."
- "If you don't know the answer, say so clearly instead of guessing."

**Key teaching line:**
> "An agent without boundaries isn't 'more free' — it's more dangerous. Boundaries are what make an agent trustworthy enough to actually deploy."

### Block 4: Behavior Rules / Decision Logic (What Do You Do in Specific Situations?)
This is where you connect back to Topic 1 — this is literally where "reasoning" gets defined.

*Example:*
"If a customer asks a simple factual question (store hours, return policy, shipping times), answer directly and confidently.
If a customer expresses frustration or files a complaint, respond with empathy first, then say you're connecting them with a team member.
If you're unsure whether you can answer something accurately, say so honestly rather than guessing."

**Teaching point:** This block is where students design the *decision tree* in plain English — no flowchart software needed, just clear if/then language.

### Block 5: Examples (Show, Don't Just Tell) — "Few-Shot Prompting"
Give the AI 1–3 examples of an ideal exchange. This is one of the highest-leverage techniques in the entire course because AI models are excellent at pattern-matching from examples, often better than following abstract rules alone.

*Example to show in class:*

```
Customer: "Do you ship internationally?"
Agent: "Great question! Yes, we ship worldwide 🌍. 
Shipping usually takes 7–14 business days depending on 
your location, and costs are calculated at checkout. 
Anything else I can help with?"
```

**Teaching point:** "This one example teaches the AI your tone, your format, your level of detail, and even that you use emojis sparingly — all without writing a single explicit rule about any of it. Examples do the heavy lifting that instructions alone can't."

## Putting It All Together: A Full Worked Example

Build this live in class, adding one block at a time so students see the prompt "grow":

```
ROLE:
You are Sam, the virtual assistant for Chapter & Verse, an independent 
bookstore in town. You help customers with questions about hours, 
book recommendations, orders, and store events.

TONE:
Speak warmly and conversationally, like a well-read friend who works 
at the store. Keep responses concise — 2 to 4 sentences unless more 
detail is genuinely needed. Occasional light humor is welcome.

BOUNDARIES:
Never make promises about specific refund amounts or timelines — 
direct these to a human team member at support@chapterandverse.com. 
Never recommend books you don't have information about — it's okay 
to say "I'm not sure, let me find out" instead of guessing.

BEHAVIOR RULES:
- If asked about store hours, location, or general policies, answer directly.
- If asked for a book recommendation, ask 1–2 clarifying questions about 
  genre or mood before suggesting titles.
- If a customer seems upset or is making a complaint, acknowledge their 
  frustration first, then offer to connect them with a team member.
- If you don't know something, say so honestly rather than inventing an answer.

EXAMPLE EXCHANGE:
Customer: "Can you recommend a book for my mom? She loves mysteries."
Agent: "I love this mission! Does she prefer something cozy and 
lighthearted, or twisty and dark? That'll help me point you to the 
right shelf 📚"
```

Walk through this line by line and ask the class: *"Which block does this sentence belong to, and why does it matter?"* This reinforces the framework actively rather than passively.

## Common Mistakes to Warn Students About

1. **Being too vague** — "Be helpful and professional" gives the AI almost nothing to work with. Specificity is what separates a generic-sounding bot from one that feels genuinely useful.
2. **Forgetting boundaries entirely** — Students (especially non-technical ones excited to just "make it work") often skip Block 3 completely, then are surprised when the agent promises things it shouldn't.
3. **Writing instructions instead of showing examples** — Long paragraphs of rules are weaker than 1–2 well-chosen examples. Teach the phrase: *"Show, don't just tell."*
4. **Over-engineering on day one** — Reassure non-technical students it's fine to start simple (Role + Tone) and layer in Boundaries, Behavior Rules, and Examples once the basic version works. Prompt design is iterative, not one-shot.

## In-Class Activity (25–30 minutes)

Students write a full 5-block system prompt for one of these (or their own idea):
- A study buddy agent for students preparing for exams
- A fitness check-in agent for a personal trainer's clients
- A local restaurant's reservation and FAQ assistant

Then, in pairs, they **test it live** by pasting it into ChatGPT or Claude as a system/custom instruction and role-playing as a customer — trying deliberately to "break" their partner's agent (asking something outside its boundaries, being rude, asking a trick question) to see if the prompt holds up.

**This activity doubles as an early, gentle introduction to Tier 3's testing concepts — plant that seed now.**

---

## Tier 1 Wrap-Up: Connecting to Project 1

By the end of these two weeks, students should be ready to build **Project 1**: a single-purpose agent with no tools and no memory — just a well-designed system prompt — using a tool like Custom GPTs or n8n's AI Agent node.

**Recap the three big ideas before moving to Tier 2:**
1. An agent *decides*; an automation *follows*.
2. Every agent is made of the same five parts: Brain, Persona, Memory, Tools, Trigger.
3. The system prompt is where students do their real "programming" — and a strong one has five clear blocks: Role, Tone, Boundaries, Behavior Rules, and Examples.
