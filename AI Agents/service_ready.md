# TIER 4 — PROFESSIONAL / SERVICE-READY
## Week 8 | Goal: Turn technical skill into a deployable product or service

---

# TOPIC 11: Data Privacy & Security Basics

## Why This Topic Exists at All

> "Everything you've built so far has been tested by you, on your own data. The moment you deploy this for a real business, you're handling *other people's* information — their customers, their employees, their private data. Getting this wrong isn't just a bad review. It can be a legal problem for your client, and for you."

## The Core Rule, Stated Simply

> "The safest data is the data you never collected in the first place. Before storing anything, ask: does this agent actually need to remember this, or am I just storing it because it was easy to?"

## What "Personal Data" Actually Means (No Legal Jargon)

Teach students to recognize personal data by example rather than definition:
- Name, email, phone number, physical address
- Payment or financial details
- Health information
- Anything that could identify a specific individual, even indirectly (an order number tied to a name, a chat transcript with someone's story in it)

**Practical filter to give students:** "If a data breach involving this information would embarrass or harm the person it belongs to, treat it as sensitive — full stop."

## What NOT to Store — A Practical Checklist

| Category | Rule |
|---|---|
| **Payment details** | Never store raw card numbers or CVVs in a Sheet, database, or agent memory — always use a payment processor's own secure checkout/tokenization, never pass this through the agent |
| **Passwords / login credentials** | Never ask the agent to collect or store these, even "temporarily" |
| **Sensitive personal topics** | Health, legal issues, immigration status — if the agent must discuss these, don't log full transcripts into a general-purpose spreadsheet everyone on the team can see |
| **Full chat transcripts by default** | Log only what's operationally needed (was it resolved, what category was it) rather than defaulting to "store everything, just in case" |

## GDPR-Lite Awareness (What Freelancers Actually Need to Know)

Frame this honestly — students are not being trained as lawyers:

> "You don't need to become a privacy lawyer. You need to know the handful of principles that show up in almost every privacy law worldwide, so you build good habits by default and know when to tell a client 'you should get proper legal advice on this part.'"

**The four principles worth teaching, in plain language:**

1. **Collect only what you need** — don't add a "date of birth" field to a form just because it might be useful someday.
2. **Tell people what you're doing with their data** — a simple line like "Your message will be used to answer your question and may be reviewed by our team" is often enough at the small-business scale.
3. **Let people ask to have their data deleted** — build in a simple way for a client's customers to request this (even just "email us and we'll remove your records" is a starting point for a small business).
4. **Don't keep data forever** — build in a habit of clearing out old logs/transcripts periodically instead of letting a Sheet grow indefinitely.

**The line to give students for client conversations:**
> "I build with privacy-conscious defaults, but I'm not a lawyer — if you're handling health data, financial data, or operating in a heavily regulated industry, please get that reviewed by someone qualified." Saying this proactively builds trust, not doubt.

## Practical n8n Habits to Teach

- **Restrict who can see the Google Sheet/database the agent writes to** — not the whole company, just whoever actually needs it
- **Avoid putting sensitive data in n8n workflow names, node names, or execution logs** that a wider team might have access to
- **Use environment variables/credentials for API keys** — never paste an API key directly into a node's visible text field or a shared workflow
- **When testing with real client data, use a copy or sample, not the live production data**, wherever possible

## Classroom Activity (15 minutes)

Give students a completed Project 2/3 agent and have them audit it with this checklist: What personal data does this agent currently collect? Is any of it stored somewhere more people can see than necessary? Is there anything being logged "just in case" that isn't actually being used for anything? Have them remove or restrict at least one thing they find.

---

# TOPIC 12: Deployment & UX for Failure States

## Reframing Deployment

> "A brilliant agent that only works inside your n8n testing window isn't a product yet — it's a prototype. Deployment is the step that turns 'I built something' into 'someone else can actually use this.'"

## The Main Deployment Channels

| Channel | What It's Good For | n8n Mechanism |
|---|---|---|
| **Website embed / chat widget** | Customer support, lead capture on a business's own site | Webhook Trigger receiving messages from an embedded chat widget, responding back through the same webhook |
| **WhatsApp** | Businesses where customers already message on WhatsApp (very common outside the US) | WhatsApp Business API connection triggering the workflow on incoming messages |
| **Slack** | Internal tools — HR assistants, internal knowledge bases, team-facing agents | Slack Trigger/node, already covered in Topic 5 |
| **Email** | Agents that respond to inbound inquiries without needing a live chat interface | Email Trigger (IMAP) node |

**Teaching point:** The AI Agent node itself doesn't change between these — only the trigger and the "send response" node change. Reinforce that students already know how to build the core agent; deployment is just choosing the right doorway for it.

## Designing for the Moments It Fails (The Part Most Builders Skip)

> "Ask any experienced builder what separates an agent a client loves from one they quietly stop using, and it's rarely the happy path — it's what happens the one time in twenty when the agent doesn't know the answer. If that moment feels broken or unhelpful, the client remembers that moment, not the nineteen good ones."

### Designing a Good "I Don't Know" Response

**Bad:** "I'm sorry, I don't understand your request."
**Better:** "That's outside what I can help with directly — but I've flagged it for our team, and someone will follow up within [timeframe]. Is there anything else I can help with in the meantime?"

**Teaching point:** A good failure response does three things: acknowledges the limitation honestly, tells the person what happens next, and doesn't leave them stuck. This directly reuses the fallback response and escalation mechanics from Tier 3, Topic 8 — deployment is where those mechanics finally get seen by a real end user, so it's worth re-testing them specifically in the deployed context, not just inside n8n's test chat window.

### The "First 5 Seconds" Rule

Teach students to think about what a brand-new user sees before they've asked anything:
- Does the chat widget clearly say what the agent can and can't help with, so people don't ask something wildly out of scope?
- Is there a visible way to reach a human immediately, for anyone who doesn't want to deal with a bot at all?

**Practical addition:** A simple opening message like *"Hi! I'm [Name], here to help with [specific things]. If you'd rather speak to a person, just say so anytime."* — this single line prevents a large share of frustrated "let me talk to a real person" conversations, by setting expectations honestly upfront rather than pretending to be all-capable.

## Classroom Activity (20 minutes)

Have students deploy their Project 2/3 agent to at least one real channel (a simple website embed is usually fastest for a classroom setting) and have a partner interact with it as a first-time user with zero context. Debrief: did the opening message set clear expectations? When the agent hit its limits, did the failure feel graceful or jarring?

---

# TOPIC 13: Client Scoping & Use-Case Discovery

## Why This Is "The #1 Skill Freelancers Skip"

> "The most common reason a freelance AI project fails isn't a bad build — it's building the wrong thing well. Someone gets excited about the tech, jumps straight to building, and only later discovers the client actually needed something completely different. Scoping is the unglamorous conversation that prevents that."

## The Core Discovery Questions (Give Students a Script)

Teach this as a repeatable interview framework, not a checklist to read verbatim — the goal is understanding, not box-ticking.

1. **"Walk me through what happens today, step by step, when [the situation this agent would handle] occurs."**
   → This surfaces the *actual* current process, which is often messier and more manual than the client initially describes.

2. **"What part of that process wastes the most time, or causes the most mistakes?"**
   → This finds the real pain point — often different from what the client leads with.

3. **"What would 'this is working well' look like to you, three months from now?"**
   → Forces a concrete definition of success before any building starts, and gives students something measurable to point back to later.

4. **"What should this NOT do, or never be allowed to do?"**
   → This is where Tier 3's Boundaries and guardrail concepts get scoped directly from the client's own words — invaluable for building the system prompt later.

5. **"What tools/systems does your team already use day to day?"**
   → Directly informs which platform and tool connections make sense (Tier 2, Topic 4's decision framework), rather than picking a tool because the student likes it.

## The Trap to Warn Students About: Feature Creep From Excitement

> "Once a client sees what's possible, they'll often start adding ideas mid-conversation — 'oh, and could it also do X, and Y, and maybe Z too?' Your job isn't to say yes to everything they get excited about. It's to help them see which one thing, done well, actually solves their real problem first."

**Practical technique:** Teach students to write down every extra idea a client mentions, but explicitly separate them into "Phase 1 (what we're building now)" and "Future ideas (parked for later)." This protects the timeline and the budget, and it gives students a natural, non-awkward way to open a second engagement later.

## Turning Discovery Into a Scope Document

Teach students to leave every discovery conversation with a short written scope covering:
- The specific problem being solved (one sentence)
- What the agent will do (a short, bulleted list — not vague)
- What the agent will explicitly NOT do (from question 4 above)
- What "success" looks like, in the client's own words
- What tools/data the agent needs access to

**Teaching point:** This document isn't bureaucracy — it's protection for both sides. If the client later says "I thought it would also do X," the scope document is the calm, non-confrontational way to say "that wasn't in what we agreed to build — let's talk about adding it as a next phase."

## Classroom Activity (25 minutes)

Pair students up. One plays a small business owner (given a business type and a vague, slightly-too-broad idea like "I want an AI to handle customer service"), the other conducts a scoping interview using the five questions above. The "client" should occasionally throw in a scope-creep idea mid-conversation, to give the interviewer practice handling it. End with the interviewer producing a short written scope document.

---

# TOPIC 14: Packaging & Pricing AI Agent Services

## Why Pricing Deserves Its Own Topic

> "Knowing how to build an agent and knowing how to charge for one are completely different skills — and most technically capable people underprice their work badly the first few times, either because they anchor on their own low running costs, or because they copy a pricing model that doesn't actually fit what they're selling."

## The Pricing Models, Explained Simply

Teach these as a "ladder" students climb as they gain experience and confidence, not as options to pick randomly:

| Model | How It Works | When to Use It |
|---|---|---|
| **Hourly** | Charge per hour of work | Weakest option long-term — it penalizes you for getting faster with practice, and clients can't budget predictably |
| **Fixed project fee** | One agreed price for a defined scope | Good starting point once a student has a clear scope document (Topic 13) — client gets predictability, student is protected by the written scope |
| **Productized package** | A pre-defined offer at a fixed price (e.g., "FAQ Agent Setup — $X") | Best once a student has built the same type of agent 2–3 times and can standardize it |
| **Setup fee + monthly retainer** | One-time build fee, plus an ongoing monthly amount for hosting, monitoring, and small updates | The most common real-world default for small-business AI agent work, and the one to teach as the primary model |
| **Usage/outcome-based** | Price tied to volume (conversations handled) or results (leads qualified) | More advanced — needs reliable measurement in place first |

**The model to teach as the default recommendation:** a hybrid of a one-time setup fee plus a modest ongoing retainer. This is consistently the standard structure in the current market — a setup fee plus a monthly retainer is the default for local-business agents: a one-time fee for the build and integration work, then a flat monthly amount for hosting, monitoring, and iteration.

## Why the Retainer Matters (Not Just as Extra Income)

Explain the *reasoning* behind the retainer, not just the number:
> "An agent isn't a website you build once and forget. It needs monitoring (is it still answering well?), occasional updates (the client's policies changed, so the knowledge base needs updating), and it has an ongoing cost to run (the AI provider bills per message). The retainer covers that ongoing reality — it's not just extra profit, it's what keeps the agent actually working six months from now instead of quietly breaking and nobody noticing."

## Realistic Numbers to Discuss (With the Caveat That These Shift)

Present this as "current market ballpark, useful for anchoring a conversation" rather than a fixed rulebook — pricing varies hugely by region, client size, and complexity, and these figures should be re-checked periodically since the market moves fast:

- Custom-built single-purpose agents (a lead qualifier, a support bot) typically run roughly $1,500–$5,000 to build, plus $300–$800/month to run.
- Multi-agent workflows orchestrating several specialized agents run roughly $5,000–$25,000 to build, plus $1,000–$3,000/month.
- For freelancers/small consultants specifically, a common range is $300–$1,500/month per client for a deployed agent handling a specific workflow.

**Important context to give students:** people routinely underestimate the ongoing running cost by two to five times — hidden costs come from the model's internal reasoning, agent retries, and the context stuffed in for retrieval (RAG). This directly ties back to Tier 3's cost estimation lesson. Teach students to build their retainer pricing from an actual estimated token cost (Topic 9's method) plus their own margin, not a guessed number.

## Packaging: Turning One Build Into a Repeatable Offer

**Teaching point for students wanting to freelance seriously:** After building 2–3 similar agents (e.g., several FAQ/support bots for different small businesses), encourage students to notice the repeatable pattern and turn it into a named package — "Small Business FAQ Agent Setup" — with a fixed price and a clear scope. This is faster to sell than a fully custom quote every time, and it's how a one-off freelance gig starts turning into an actual small business.

## Building a Portfolio / Case Study (Even From a Class Project)

Teach students to document every project — including class projects — the same way they'd document real client work:
- The problem (one sentence)
- What was built (a short description, plus a screenshot of the n8n workflow)
- The result, even if simulated ("reduced average response time from X to Y" or "answered 90% of test questions accurately from the knowledge base")

**Teaching point:** A freelancer with three well-documented projects, even simulated ones from this course, looks far more credible to a first real client than someone who can only say "I took a course."

## In-Class Activity: The One-Page Proposal (Feeds Directly Into the Capstone)

Using the scope document built in Topic 13's activity, have students write a one-page proposal including:
1. The problem being solved (from the scope doc)
2. What will be built (bulleted, specific)
3. Pricing (using the setup fee + retainer model, with a rough cost estimate shown using Tier 3's method)
4. Timeline (a realistic build schedule, broken into phases if scope creep ideas were parked earlier)

This proposal is the direct template for the Capstone Project deliverable.

---

# 🔨 CAPSTONE PROJECT — Full Client-Ready Agent

## The Brief

Each student scopes, builds, tests, and deploys one full agent for a real or simulated client, applying everything from all four tiers.

## Step-by-Step Deliverables

1. **Scoping (Topic 13):** A written scope document from a discovery conversation (real or role-played)
2. **Build (Tiers 1–2):** A full system prompt (5 blocks), at least one tool connection, and a RAG knowledge base
3. **Reliability (Tier 3):** At least one guardrail (loop limit, fallback response, or human-in-the-loop checkpoint) and a rough cost estimate for running it
4. **Privacy check (Topic 11):** A short audit confirming what data is collected, where it's stored, and who can access it
5. **Deployment (Topic 12):** Deployed to at least one real channel, with a tested, graceful failure response
6. **Proposal (Topic 14):** A one-page client-facing proposal — problem, solution, pricing, timeline

## Assessment Criteria (Suggested Rubric)

| Criterion | What to Look For |
|---|---|
| **Scope clarity** | Is it obvious what the agent does and doesn't do, based on the scope document? |
| **Build quality** | Does it apply the Tier 1–2 fundamentals cleanly — a strong system prompt, working tool connections, accurate RAG answers? |
| **Reliability** | Does it have at least one real guardrail, and does the student understand *why* they chose it? |
| **Privacy awareness** | Can the student explain what data is collected and justify that it's the minimum necessary? |
| **Real-world readiness** | Would a non-technical business owner be comfortable putting this in front of their actual customers? |
| **Business framing** | Is the proposal priced realistically, using an actual cost estimate rather than a guess? |

**Closing note for students finishing the Capstone:**
> "What you're holding at the end of this isn't a class project — it's a portfolio piece and a pricing conversation you already know how to have. That's the actual finish line of this course: not 'I can build an agent,' but 'I can scope, build, price, and deliver one for someone who's paying me to get it right.'"
