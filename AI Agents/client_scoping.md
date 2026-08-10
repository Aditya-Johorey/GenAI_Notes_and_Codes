# Client Scoping & Use-Case Discovery

## Start With a Story, Not a Definition

Before teaching any framework, tell students this scenario — it's the fastest way to make the lesson land:

> "A freelancer gets hired to build a customer support AI agent. The client says 'we need something that answers customer questions.' The freelancer gets excited, spends two weeks building a slick chatbot that answers FAQs about products. They demo it proudly. The client says: 'This is nice, but... our actual problem is that customers email us asking to *cancel orders*, and we can't process cancellations fast enough. We're losing money on refund disputes. This doesn't do that.'
>
> Two weeks. Gone. Not because the freelancer built badly — because they built the wrong thing, confidently."

This is the lesson in one story: **enthusiasm about the tech is not the same as understanding the problem.** For non-technical students especially, this matters because they may feel extra pressure to "prove" their technical competence quickly — and rushing to build is often how that pressure shows up. Reassure them: *the client isn't hiring you because you can build fast. They're hiring you because you can solve their problem. Scoping IS the skill.*

---

## Reframe What "Scoping" Actually Is

Non-technical students often hear "scoping" and think it's a formal, corporate, intimidating process — something consultants do with clipboards. Reframe it simply:

**Scoping is just a structured conversation where you make sure you understand the problem before you touch anything.**

That's it. It's closer to what a good doctor does before prescribing anything — ask questions, understand symptoms, understand the patient's life — than it is to a technical audit. Students don't need to know how AI models work to be excellent at this. In fact, this is a skill many non-technical people are *naturally better at* than technical people, because it's about listening, not building.

---

## The Core Discovery Questions — Taught With "Why" and "What It Sounds Like"

Give students not just the question, but the *reasoning* and a sample of what a real answer might sound like — so they can recognize when they've actually gotten a useful answer versus a vague one.

### 1. "Walk me through what happens today, step by step, when [the situation] occurs."

**Why this question first:** Clients describe their problem in headlines ("we need faster customer support"). This question forces them into specifics, and specifics are where the real project lives.

**What a vague answer sounds like (red flag):** "We just handle it as it comes in."
→ Teach students to gently push further: *"Sure — but walk me through an actual example from last week. What's the very first thing that happens?"*

**What a useful answer sounds like:** "Okay, so a customer emails support@ourcompany.com. Sarah checks that inbox twice a day. She reads the email, checks our order system to see the order status, then manually types a reply. If it's a refund request, she has to also log into a separate finance tool to check if we've already refunded them, then message our finance person on Slack to approve it."

Now the student knows: there are two systems involved, a human bottleneck (Sarah, twice a day), and a manual approval step. That's a real project shape — not "build a chatbot."

**Teaching point:** The step-by-step answer is often 3-5x messier than the client's opening description. That messiness is the goldmine.

---

### 2. "What part of that process wastes the most time, or causes the most mistakes?"

**Why this matters:** Clients often lead with what excites them about AI ("I saw this cool demo") rather than what actually hurts them. This question redirects to pain, not novelty.

**Teaching point for non-technical students:** You don't need to evaluate *whether AI can fix the pain point* yet — that's a later step, possibly with help from more technical framework (Tier 2). Right now your only job is to find out **where it hurts**. Resist the urge to start problem-solving out loud mid-question. Just listen and take notes.

**Follow-up prompt to teach students:** "How much time would you say that takes per week?" — this turns a vague complaint into a rough number, which becomes useful later for showing the client ROI ("this used to take 5 hours/week, now it takes 30 minutes").

---

### 3. "What would 'this is working well' look like to you, three months from now?"

**Why this question is critical — and often skipped:** Without this, "success" is whatever the client feels like in the moment, which can shift. This question forces a concrete, checkable definition *before* any work starts.

**Teach students to listen for two kinds of answers:**
- **Vague (needs follow-up):** "I'd just feel like it's helping."
  → Push gently: *"What would you actually see or measure? Fewer emails in the inbox? Faster replies? Fewer mistakes?"*
- **Concrete (what you want):** "Sarah isn't spending her mornings on this anymore. Customers get a reply within an hour instead of a day. And refund approvals don't need a Slack message anymore."

**Teaching point:** Write this answer down *in the client's own words*. You'll use it again later — both in the scope document and when you eventually show them the finished project ("Remember you said success would look like X? Here's how we're tracking against that.").

---

### 4. "What should this NOT do, or never be allowed to do?"

**Why non-technical students should love this question:** This is where you don't need any technical knowledge at all — you just need to ask and listen. The client will often *volunteer* the exact guardrails needed.

**What answers commonly sound like:**
- "It should never issue a refund on its own — only suggest one for a human to approve."
- "It should never tell a customer their order is going to arrive on a specific date if we're not sure — that's caused problems before."
- "It shouldn't just make something up if it doesn't know the answer."

**Teaching point — connect this explicitly to later material:** Tell students plainly: *"Everything the client says here becomes a rule you'll write directly into the agent's instructions later. You're not designing the guardrails yourself out of thin air — the client is handing them to you right now, in this conversation. Your job is to capture it accurately."* This makes an abstract future task (writing a system prompt) feel concrete and doable, because they'll already have the raw material.

---

### 5. "What tools/systems does your team already use day to day?"

**Why this question matters for non-technical students specifically:** You do NOT need to know how to build integrations with every tool a client might mention. Your job here is just **inventory-taking** — like a nurse asking "what medications are you currently on?" before a doctor decides on treatment.

**What to listen for and write down, plainly, without needing to understand the tech underneath:**
- What email platform do they use? (Gmail? Outlook?)
- What do they use to track orders, customers, or tickets? (Shopify? A spreadsheet? Salesforce?)
- Where do they communicate internally? (Slack? Teams?)
- Is anything already "connected" to anything else, or is it all manual copy-paste between systems?

**Reassurance to give students:** "You don't have to know *how* to connect to Salesforce right now. You just need to know it exists in their workflow. Whether and how to connect it is a separate decision you'll make later, possibly with help from a technical framework or a more experienced collaborator."

---

## The Trap: Feature Creep From Excitement

### Why This Happens (Explain the Psychology)

Once non-technical clients see what AI can plausibly do, their brain starts generating ideas in real time — this is a *good* sign (they're engaged!) but a *dangerous* one for the project. Explain it to students like this:

> "Imagine you take someone to a buffet who's never seen one before. They don't calmly pick one plate. They get excited and start piling on everything, because it's all right there and it all looks great. Client conversations about AI are the same. The moment they see the first thing is possible, their brain starts asking 'okay, but could it also...?' That's not a bad client. That's just human excitement. Your job is to be the calm one."

### The Skill: Redirect Without Rejecting

Teach students this is **not** about telling the client "no." It's about being the person in the room who protects the project's focus. Give them exact phrases to use, because non-technical students especially benefit from scripts they can lean on in the moment rather than having to improvise under social pressure:

- *"I love that idea — let's write it down so we don't lose it. Can I park it as a 'Phase 2' idea while we make sure Phase 1 is rock solid first?"*
- *"That's a great addition. Just so I'm scoping this properly — is that something you need on day one, or would it be okay as something we add once the first version is working well?"*

**Teaching point:** This technique does three things at once — it makes the client feel heard (not shut down), it protects the freelancer's timeline and budget, and it *plants the seed for a second paid engagement later*, without ever having to awkwardly pitch "more work" out of nowhere.

### The Two-Column Trick

Teach students to literally keep two running lists during and after every discovery call:

| Phase 1 (Building Now) | Future Ideas (Parked) |
|---|---|
| Auto-draft replies to refund questions | Auto-handle shipping delay questions |
| Flag refund requests for human approval | Connect to Salesforce for order history |
| | Auto-send review requests after resolution |

This simple visual habit — even done on paper or in a notes app — is often the single biggest thing that keeps a freelance project from spiraling in scope.

---

## Turning Discovery Into a Scope Document

### Why a Document, Not Just Notes

Explain to non-technical students that this document isn't about looking professional (though it does that too) — it's a **shared memory** that protects both people in the relationship from the natural human tendency to remember conversations differently over time. Nobody is lying when they say "I thought it would also do X" three weeks later — memory just drifts. The document exists so nobody has to rely on memory.

### The Five Sections, Explained Simply

**1. The specific problem being solved (one sentence)**
Teach students to write this as if explaining to a friend who wasn't in the meeting. Example: *"Sarah spends 2+ hours a day manually replying to refund emails and cross-checking a separate finance tool."*

**2. What the agent will do (a short, bulleted list — not vague)**
Teach students to avoid vague verbs like "help with" or "assist." Instead use concrete, checkable actions:
- ❌ "Helps with customer emails"
- ✅ "Drafts a reply to incoming refund request emails within 5 minutes of receipt"
- ✅ "Flags the request for Sarah's approval before sending anything"

**3. What the agent will explicitly NOT do**
Straight from discovery question #4. Example:
- "Will not issue refunds automatically"
- "Will not respond to non-refund-related emails"
- "Will not guess at delivery dates"

**4. What "success" looks like, in the client's own words**
Literally copy-paste or closely paraphrase what they said in question #3. This matters — using their language (not yours) means there's no room for them to later say "that's not what I meant."

**5. What tools/data the agent needs access to**
The plain inventory list from question #5.

### The Non-Confrontational Superpower This Gives Students

Roleplay this moment with students so they feel it before it happens to them for real:

> **Client (three weeks in):** "Wait, I thought it would also handle shipping delay questions?"
>
> **Student (calm, not defensive):** "Totally hear you — I actually have that written down in our 'Future ideas' list from our first call, since we agreed Phase 1 was just refund requests. Want me to scope that out as a Phase 2 addition once this is live and working well?"

**Teaching point to land at the end:** Notice this response has zero conflict in it. The student isn't arguing, isn't defensive, isn't saying "you're wrong." The document does the disagreeing *for* them, calmly, in writing, from an earlier date. This is exactly why the document is protection — not paperwork.
