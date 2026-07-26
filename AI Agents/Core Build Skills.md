# TIER 2 — CORE BUILD SKILLS (n8n-Focused)
## Weeks 3–5 | Goal: The agent can act, remember, and know things

---

# TOPIC 4: The n8n AI Agent Node

## The One Concept to Land First

A normal n8n workflow runs the same steps in the same order every time. The **AI Agent node** is different: you give it a goal and a set of tools, and it decides *for itself*, in real time, which tool to use, in what order, and when to stop.

**Explain it like this for non-technical students:**
> "A regular n8n workflow is like a recipe — step 1, step 2, step 3, always the same. The AI Agent node is like handing a task to a smart assistant and a box of tools, and letting them figure out which tool to grab and when. You don't script every step — you describe the job and give them what they need to do it."

## What the Node Actually Needs (3 Connections)

Every AI Agent node has three optional inputs. Draw this on the whiteboard as a simple diagram — it removes 90% of the confusion:

```
   [Chat Model]  →  
   [Memory]      →   [AI AGENT NODE]  →  produces a response
   [Tools]       →  
```

| Connection | Plain-Language Meaning | Required? |
|---|---|---|
| **Chat Model** | Which AI "brain" is doing the thinking (OpenAI, Anthropic, Google Gemini, or a local model via Ollama) | Yes, always |
| **Memory** | Does it remember earlier messages in this conversation? | Optional |
| **Tools** | What actions can it take (Sheets, Slack, Email, custom APIs)? | Optional |

**Teaching point:** An AI Agent node with *zero* tools and *no* memory is still valid — it's just a smart chatbot. You add Memory and Tools only when the job actually needs them. Don't over-build by default.

## The System Message Field

Inside the AI Agent node is a field called **System Message**. This is where students paste the 5-block system prompt they wrote in Tier 1 (Role, Tone, Boundaries, Behavior Rules, Examples). Nothing new to teach here conceptually — just show them exactly where that work lives inside the actual node.

## Starting vs. Production Triggers

| Node | When to Use |
|---|---|
| **Chat Trigger** | For building and testing — opens a simple chat window inside n8n |
| **Webhook Trigger** | For real deployment — lets an external system (a website, WhatsApp, Slack) start the agent |

**Rule to teach directly:** Build and test with Chat Trigger first. Only switch to Webhook Trigger once the agent is behaving reliably. Don't deploy something you haven't tested in the safe sandbox first.

## The Golden Rule of Tool Count

**Teach this as a hard rule, not a suggestion:** Start with 2–3 tools maximum. Every tool added increases the chance the model picks the wrong one or gets confused about which to use. Add more only after testing proves the agent handles the first few reliably.

---

# TOPIC 5: Connecting Tools & APIs

## The Core Idea, Explained Simply

> "Every tool you attach to the AI Agent isn't just 'connected' — it comes with a written description telling the agent *when* to use it. If that description is vague or missing, the agent either ignores the tool completely or uses it at the wrong moment. Writing a good tool description is just like writing a mini system prompt — same skill, smaller scale."

This is the single most important teaching point in this topic. Say it more than once.

## Tools to Demo, in Order of Difficulty

| Node (used as a "Tool") | What It Lets the Agent Do | Good First Use Case |
|---|---|---|
| **Google Sheets** | Read or write rows in a spreadsheet | Log every conversation; check stock levels |
| **Slack** | Post a message to a channel or person | Alert a human when the agent is unsure |
| **Gmail** | Send an email | Follow up with a customer automatically |
| **HTTP Request** | Call any external API, even ones without a pre-built n8n node | Anything not covered by the above |
| **Sub-workflow** | Trigger an entirely separate n8n workflow | Break a big job into smaller reusable pieces |

**Order to teach in:** Google Sheets and Slack first — students already understand both conceptually (a spreadsheet, a chat message), so there's zero new abstraction, just a new context. HTTP Request comes last because it requires understanding that not every app has a ready-made connector.

## The Tool Description Field — Walk Through a Real Example

Show students exactly what a good vs. bad tool description looks like:

**Bad description:** "Google Sheets tool"
→ The agent has no idea when to actually use this.

**Good description:** "Use this tool to log every customer question and whether it was answered, into the tracking sheet. Call this after every conversation ends, not during."

**Teaching point:** The description answers three questions for the agent: *what does this tool do, when should I use it, and when should I NOT use it.* Any tool description missing one of those three is incomplete.

## The "Escalation" Pattern (Tie Back to Tier 1)

This is the most commonly requested real-world pattern, so give it its own callout:

> Behavior Rule in system prompt: "If you are confident in your answer, respond directly. If you are not confident, or the customer seems upset, use the Slack tool to alert the support channel with a one-line summary."

This single pattern — act automatically when confident, alert a human when not — is what most client projects actually want. Make sure every student builds this at least once.

## Common Mistakes to Flag

1. **Too many tools at once** → confusion (see Topic 4's golden rule)
2. **Vague tool descriptions** → the agent either never uses the tool, or uses it at the wrong time
3. **No plan for tool failure** → what happens if the Sheet is unreachable or the Slack channel was deleted? Ask students this question directly, even if they don't build the fix yet — it plants the seed for Tier 3.

---

# TOPIC 6: Memory & Context

## The Two Types, Explained With One Clean Analogy

> "Short-term memory is like talking to a stranger on a train — they remember this one conversation, but forget you completely once it ends. Long-term memory is like your family doctor, who pulls up your full history even if it's been two years since your last visit."

## The n8n Nodes for Each

| Node | Type | When to Use |
|---|---|---|
| **Simple Memory (Window Buffer)** | Short-term | Keeps the last N messages in one conversation. Fine for single-session chats. |
| **Postgres Chat Memory** or **Redis Chat Memory** | Long-term | Persists across sessions, keyed by a Session ID. Use when the agent needs to recognize a *returning* person. |

## The One Concept That Trips Everyone Up: Session ID

**Explain it like this:**
> "Memory in n8n isn't automatic — it's filed under a specific ID, like a filing cabinet drawer labeled with someone's name. If you don't tell the agent whose 'drawer' to use, it either mixes everyone's conversations together or starts fresh every single time. The Session ID is that label — usually a phone number, email address, or chat username."

**Practical rule to teach:** Before building anything with persistent memory, students must answer: *"What unique piece of information will I use to identify this person every time they come back?"* If they can't answer that, the memory setup will break.

## Sizing the Memory (Cost/Context Tradeoff)

Briefly flag this — it's expanded fully in Tier 3, but plant it now:
> "The more memory you keep, the more information gets sent to the AI on every single message, which costs more and can actually make responses worse, not better. Keep only as much memory as the job actually needs."

---

# TOPIC 7: RAG (Retrieval-Augmented Generation)

## The Core Idea in One Line

> "RAG means: before answering, the agent looks up the right information from your documents, and answers from that — instead of guessing from its general training."

**One analogy, used once:**
> "It's the difference between an open-book exam and a closed-book exam. Without RAG, the AI is answering from memory alone. With RAG, it's handed the exact right page before it answers."

## The Full Pipeline in n8n (This Is the Whole Thing — Nothing More)

Walk through this as a straight left-to-right sequence:

```
Upload Document → Text Splitter (chunks it) → Embeddings node (converts to vectors)
→ Vector Store (stores it) → Attach Vector Store as a TOOL on the AI Agent
```

| Node | Job |
|---|---|
| **Default Data Loader** | Reads the uploaded file (PDF, doc, spreadsheet, etc.) |
| **Text Splitter** | Breaks the document into smaller chunks |
| **Embeddings node** (OpenAI or Google) | Converts each chunk into a format the vector database can search by meaning |
| **Vector Store node** (Simple Vector Store for testing; Pinecone or Supabase for production) | Stores the chunks so they can be searched later |
| **Vector Store as Tool** on the AI Agent | Lets the agent query this knowledge base mid-conversation, exactly like any other tool |

**Teaching point to repeat:** Once the Vector Store is attached as a Tool, it behaves exactly like the Google Sheets or Slack tools from Topic 5 — same mechanic, same "tool description" logic, just searching documents instead of writing to a spreadsheet. This isn't new mechanics, it's the same skill applied to a new kind of tool.

## Chunk Size — The One Setting Worth Explaining

> "If a chunk is too big (a whole document at once), the agent gets buried in irrelevant text. If a chunk is too small (one sentence), the agent loses context and doesn't understand what it's reading. Start with the platform's default chunk size, and only adjust it if you notice the agent's answers are consistently off."

## Testing for Honesty, Not Just Accuracy

The most important habit to build here: after setting up RAG, deliberately ask the agent something **not** covered in the uploaded documents.

- **Good outcome:** the agent says "I don't have that information."
- **Bad outcome:** the agent confidently makes something up (a hallucination).

If it hallucinates, the fix is almost always in the **Boundaries block of the system prompt** (Tier 1, Topic 3) — add an explicit instruction like: *"If the answer isn't in the provided documents, say so clearly. Do not guess."* This connects RAG directly back to system prompt design, which is worth calling out explicitly.

---

# 🔨 PROJECT 2 — n8n Build Checklist

Give students this as a literal step-by-step build order:

1. **Chat Trigger** node → connects into →
2. **AI Agent** node
3. Attach a **Chat Model** (pick one: OpenAI, Anthropic, Gemini)
4. Attach **Memory**: Simple Buffer (short conversations) or Postgres/Redis (returning users) — decide based on the Session ID question from Topic 6
5. Attach **Tool #1**: Google Sheets (log conversations) or Slack (alert a human) — write a clear tool description
6. Build the **RAG pipeline**: Load Document → Text Splitter → Embeddings → Vector Store, then attach the Vector Store as **Tool #2** on the Agent
7. Paste the **5-block system prompt** into the AI Agent's **System Message** field, including an explicit "don't guess" instruction
8. Test using Chat Trigger: ask 3 questions the documents can answer, and 1 trick question they can't — confirm the agent responds honestly to the trick question

**Deliverable:** A working n8n workflow, screenshot of the canvas, and a one-paragraph explanation of who this agent is for and what problem it solves.
