# AI Agent Curriculum — Basic to Advanced

### 🟢 TIER 1 — FOUNDATION (Weeks 1–2)
*Goal: Students understand what an agent is and ship one simple working agent.*

| # | Topic | Why Here |
|---|---|---|
| 1 | **Agents vs. Automations** — the mental shift from "trigger→action" to "reason→decide→act" | Prevents them from just rebuilding n8n workflows and calling it an agent |
| 2 | **Anatomy of an Agent** — brain, memory, tools, persona, trigger loop | Gives them a reusable mental checklist |
| 3 | **System Prompt & Persona Design** — role, tone, boundaries, few-shot examples | This *is* the "programming" in no-code agent building |

**🔨 Project 1:** Build one single-purpose agent with no tools, no memory, no integrations — just a well-designed system prompt (e.g., a customer FAQ responder or a personal journaling coach) using Custom GPTs or n8n's AI Agent node. **Goal: ship something real in week 2.**

---

### 🟡 TIER 2 — CORE BUILD SKILLS (Weeks 3–5)
*Goal: The agent can now act, remember, and know things.*

| # | Topic | Why Here |
|---|---|---|
| 4 | **No-Code Agent Platforms Deep Dive** — n8n AI Agent, Relevance AI, Voiceflow, Lindy, Zapier Agents, Custom GPTs | Now that they know *what* an agent needs, they can evaluate tools intelligently |
| 5 | **Connecting Tools & APIs (No-Code)** — Sheets, CRM, Slack, email, webhooks | Gives the agent hands |
| 6 | **Memory & Context Management** — short-term vs. long-term, no-code vector DBs | Gives the agent continuity |
| 7 | **RAG (No-Code)** — feeding an agent your own documents/knowledge base | Gives the agent expertise — this is the single highest-value skill for client work |

**🔨 Project 2:** Upgrade Project 1's agent — add a tool connection (e.g., logs data to Sheets or sends Slack alerts) **and** a knowledge base (RAG) so it answers from real documents. This is their first "sellable" agent.

---

### 🟠 TIER 3 — ADVANCED SYSTEMS (Weeks 6–7)
*Goal: Reliability, safety, and complexity — the difference between a demo and a deployable product.*

| # | Topic | Why Here |
|---|---|---|
| 8 | **Testing, Debugging & Guardrails** — edge cases, prompt injection defense, fallback responses, human-in-the-loop checkpoints | Must come *before* multi-agent complexity, or errors compound |
| 9 | **Cost & Token Economics** — estimating API costs, choosing models by cost/performance, rate limits | Non-negotiable if they're building for clients — nobody wants a surprise $2,000 OpenAI bill |
| 10 | **Multi-Agent Orchestration** — manager/specialist agent patterns, agent-to-agent handoff | Now they have the guardrails and cost sense to do this safely |

**🔨 Project 3:** Build a 2–3 agent system with a manager agent delegating to specialists (e.g., a content pipeline: research agent → writer agent → editor/QA agent), with guardrails and a cost estimate attached.

---

### 🔴 TIER 4 — PROFESSIONAL / SERVICE-READY (Week 8)
*Goal: Turn technical skill into a deployable product or service offering — this is what your original list was missing entirely.*

| # | Topic | Why It's Essential |
|---|---|---|
| 11 | **Data Privacy & Security Basics** — handling client/user data, what NOT to store, compliance basics (GDPR-lite awareness) | Corporate/freelance students will be handling other people's data — this protects them legally |
| 12 | **Deployment & UX for Failure States** — embedding on websites, WhatsApp/Slack bots, and designing graceful failure ("I don't know" responses, escalation to human) | A technically perfect agent with a bad failure UX still fails in the real world |
| 13 | **Client Scoping & Use-Case Discovery** — how to interview a client/business to find where an agent adds value, defining scope before building | The #1 skill freelancers skip — they build cool tech nobody needed |
| 14 | **Packaging & Pricing AI Agent Services** — productized offers, one-time build vs. retainer, portfolio/case study building | Converts "I can build agents" into "I have a business" |

**🔨 Capstone Project:** Each student scopes, builds, tests, and deploys one full agent for a real or simulated client — including a one-page proposal document (use case, cost estimate, timeline) as if pitching a service. This becomes their portfolio piece.

---

## Why This Order Works
Each tier only introduces complexity the students are *ready* for, and every tier ends in something tangible — by the end they'll have **4 working agents** and a pitch document, not just 14 topics of theory. The service-readiness tier at the end is what turns "I learned AI agents" into "I can freelance or consult on this," which sounds like the real outcome you're aiming for.

Want me to convert this into a full 8-week session-by-session syllabus with time blocks, specific tool recommendations per session, and grading/assessment criteria? I can put that together as a document.
