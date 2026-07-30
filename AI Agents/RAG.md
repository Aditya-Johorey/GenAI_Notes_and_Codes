# Topic 7: RAG (Retrieval-Augmented Generation)

## The Core Idea, Restated

**RAG means: before answering, the agent looks up the right information from your documents, and answers from that — instead of guessing from its general training.**

> **Analogy (use once, then drop it):** It's the difference between an open-book exam and a closed-book exam. Without RAG, the AI is answering from memory alone. With RAG, it's handed the exact right page before it answers.

Why this matters in practice: an AI agent's training data is frozen at some point in the past, and it was never trained on *your* internal documents — your policies, your product manuals, your client contracts. RAG is how you give a general-purpose agent specific, current, private knowledge without retraining it. Nothing about the model itself changes. You're just changing what it's allowed to *see* before it answers.

---

## Why Not Just Paste the Whole Document Into the Prompt?

Before explaining the pipeline, it's worth answering the question students will already be thinking: "Why not just paste the document into the system prompt?"

- **Size limits.** Models can only read a limited amount of text at once (the "context window"). A 200-page manual won't fit, and even if it did, you'd be paying for and slowing down every single request by re-sending the whole thing.
- **Noise.** Even when it fits, dumping the entire document in means the agent has to wade through mostly irrelevant text to find the one paragraph that matters. That hurts answer quality.
- **Multiple documents.** Real use cases involve dozens or hundreds of files. You can't paste all of them into one prompt.

RAG solves this by pre-processing your documents once, storing them in a searchable form, and then pulling out *only* the relevant few chunks at the moment they're needed.

---

## The Full Pipeline in n8n (Nothing More Than This)

Walk through this as a straight left-to-right sequence:

```
Upload Document
      │
      ▼
Text Splitter  (chunks it)
      │
      ▼
Embeddings Node  (converts chunks to vectors)
      │
      ▼
Vector Store  (stores it)
      │
      ▼
Vector Store attached as a TOOL on the AI Agent
```

| Node | Job | Plain-English explanation |
|---|---|---|
| **Default Data Loader** | Reads the uploaded file (PDF, doc, spreadsheet, etc.) | This is the "intake" step — it opens the file and pulls out raw text, no matter the format. |
| **Text Splitter** | Breaks the document into smaller chunks | Cuts the document into bite-sized pieces so the agent can retrieve just the relevant piece later, not the whole file. |
| **Embeddings Node** (OpenAI or Google) | Converts each chunk into a format the vector database can search by *meaning* | Turns text into a list of numbers ("a vector") that represents its meaning. Chunks about similar topics end up with similar numbers, even if they don't share the same words. |
| **Vector Store Node** (Simple Vector Store for testing; Pinecone or Supabase for production) | Stores the chunks so they can be searched later | This is the "library" where all your chunked, embedded documents live, ready to be searched. |
| **Vector Store as Tool on the AI Agent** | Lets the agent query this knowledge base mid-conversation | The agent can now "ask the library a question" any time it needs facts, exactly like any other tool call. |

### Teaching Point to Repeat

Once the Vector Store is attached as a Tool, it behaves exactly like the Google Sheets or Slack tools from Topic 5 — **same mechanic, same "tool description" logic, just searching documents instead of writing to a spreadsheet.**

This isn't new mechanics. It's the same skill (giving the agent a tool, writing a clear description of when to use it) applied to a new kind of tool. If students understood tool descriptions in Topic 5, they already understand 80% of what makes RAG work well — because the agent still has to *decide* when to call the vector store tool, and that decision quality depends on how clearly the tool is described.

---

## What's Actually Happening When the Agent "Retrieves"

It helps to demystify the moment retrieval happens, step by step:

1. The user asks a question.
2. The agent's system prompt (and the tool description) tells it: "if the question might be answered by the knowledge base, search it first."
3. The agent turns the *question itself* into a vector (using the same embeddings model).
4. The vector store compares that question-vector to every chunk-vector it has stored, and returns the handful of chunks that are mathematically "closest" in meaning.
5. Those chunks get inserted into the conversation as context, right before the agent writes its answer.
6. The agent answers using that retrieved text — ideally citing or grounding its answer in it, not inventing anything extra.

This is why RAG search is described as searching "by meaning" rather than by keyword. A question like "How much do I get refunded if I cancel late?" can retrieve a chunk that says "Cancellations made within 48 hours are non-refundable" even though no words overlap — because the *meaning* is close.

---

## Chunk Size — The One Setting Worth Explaining

> If a chunk is too big (a whole document at once), the agent gets buried in irrelevant text. If a chunk is too small (one sentence), the agent loses context and doesn't understand what it's reading. Start with the platform's default chunk size, and only adjust it if you notice the agent's answers are consistently off.

To make this concrete for students, give them a feel for both failure modes:

**Chunks too large (e.g., 1 chunk = 1 whole 20-page policy manual)**
- The agent retrieves the "closest" chunk, but that chunk is enormous — most of it irrelevant to the question.
- Wastes context space, slows things down, and can dilute the actually-relevant sentence among pages of noise.

**Chunks too small (e.g., 1 chunk = 1 sentence)**
- A single sentence pulled out of context can be misleading. Example: a chunk containing only *"This does not apply to Enterprise customers"* — without the preceding sentence explaining *what* "this" refers to — is nearly useless on its own.
- The agent may retrieve a technically-relevant sentence but misinterpret it because the surrounding context was chopped off.

**The practical rule of thumb:** most platforms default to a chunk size somewhere in the range of a few paragraphs, often with a small "overlap" between chunks (so the end of one chunk repeats a bit at the start of the next, to avoid cutting a thought in half). Students should not touch this setting on day one. Only revisit it after testing shows a specific, repeatable problem — e.g., "every time we ask about refund exceptions, it seems to only get half the policy."

---

## A Quick Vocabulary Grounding

Students don't need to become ML engineers, but these four terms will keep coming up:

| Term | One-sentence definition |
|---|---|
| **Chunk** | A small piece of a document, split so it can be retrieved on its own. |
| **Embedding** | A numeric representation of text's *meaning*, used so a computer can compare "closeness" of ideas, not just matching words. |
| **Vector Store** | A database built specifically to store embeddings and quickly find the closest matches to a new query. |
| **Retrieval** | The act of the agent searching the vector store and pulling back the most relevant chunks before answering. |

---

## Choosing a Vector Store: Testing vs. Production

| | Simple Vector Store | Pinecone / Supabase |
|---|---|---|
| **Best for** | Prototyping, demos, small document sets | Real production use, larger document sets, multiple users |
| **Setup effort** | Minimal — works inside n8n with no extra account | Requires its own account/service, more setup |
| **Persistence** | Often resets or is limited in scale | Designed to store data reliably long-term |
| **When to graduate** | As soon as you're confident the workflow logic is right | Before handing the agent to real users |

Teaching point: students should build and test with the Simple Vector Store first. Don't let infrastructure decisions (which production vector database to use) block learning the *pipeline logic*, which is identical either way.

---

## Testing for Honesty, Not Just Accuracy

This is the most important habit in this entire topic, so it deserves its own emphasis, not just a mention.

**The test:** after setting up RAG, deliberately ask the agent something that is *not* covered in the uploaded documents.

- ✅ **Good outcome:** the agent says something like *"I don't have that information in the documents provided."*
- ❌ **Bad outcome:** the agent confidently makes something up — a hallucination — blending its general training knowledge with the retrieved chunks in a way that sounds authoritative but is fabricated or wrong.

### Why This Test Matters More Than "Does It Get the Right Answer"

An agent that gets right answers 95% of the time but confidently invents an answer the other 5% of the time is *more* dangerous than one that gets things right only 80% of the time but reliably says "I don't know" when it's unsure. Confident wrong answers erode trust and can cause real harm (wrong refund policy told to a customer, wrong compliance detail told to an employee). Students should test for *honesty under uncertainty*, not just correctness under normal conditions.

### The Fix Lives in the System Prompt, Not the RAG Setup

If the agent hallucinates, the fix is almost always in the **Boundaries block of the system prompt** (Tier 1, Topic 3) — not in the Vector Store settings. Add an explicit instruction such as:

> *"If the answer isn't in the provided documents, say so clearly. Do not guess."*

**This connects RAG directly back to system prompt design.** RAG only supplies the agent with better *material* to work from — it does not, by itself, stop the agent from filling gaps with invented content. That behavior is controlled the same way all agent behavior is controlled: through clear instructions in the prompt.

### A Simple Two-Question Test Script for Students

Have students run both of these on every RAG agent they build:

1. **In-scope question:** something clearly answered in the uploaded documents. → Confirms retrieval is working.
2. **Out-of-scope question:** something plausible-sounding but *not* in the documents (e.g., asking about a policy from a country the company doesn't operate in). → Confirms the agent admits the limit instead of guessing.

If test #2 fails, don't touch the Text Splitter or Embeddings node — go straight to the Boundaries block of the system prompt.

---

## Common Mistakes to Watch For

| Mistake | Symptom | Fix |
|---|---|---|
| No Boundaries instruction | Agent hallucinates confidently on out-of-scope questions | Add "if it's not in the documents, say so" to the system prompt |
| Chunk size never tested | Answers are vague or miss obvious details | Run the two-question test; only then consider adjusting chunk size |
| Vector store not attached as a tool, just referenced in text | Agent never actually searches; answers from memory | Confirm the Vector Store node is wired in as a Tool on the AI Agent, not just present in the workflow |
| Documents not re-uploaded after edits | Agent answers based on an outdated version of the document | Re-run the ingestion pipeline (Loader → Splitter → Embeddings → Store) any time the source document changes |
| Vague tool description on the Vector Store tool | Agent doesn't know *when* to search vs. answer directly | Write a clear tool description, same discipline as Topic 5's Sheets/Slack tools |

---

## Quick Recap for Students

1. RAG = open-book exam for the AI. It looks things up instead of guessing.
2. The pipeline is one straight line: **Loader → Splitter → Embeddings → Vector Store → attached as a Tool.**
3. Chunk size is the one setting worth knowing about — not too big, not too small — but start with the default.
4. The real test of a good RAG setup isn't "did it get the right answer," it's "did it admit when it didn't know." Test this on purpose, every time.
5. When it fails that test, the fix isn't in the RAG nodes — it's back in the system prompt's Boundaries block from Topic 3.

---

## Suggested In-Class Exercise

1. Have students upload a short internal document (a fake company policy works well) through the RAG pipeline.
2. Ask an in-scope question and confirm the agent retrieves and answers correctly.
3. Ask an out-of-scope question and watch what happens.
4. If it hallucinates, have them add a Boundaries instruction to the system prompt and re-test — so they *see* the fix land, rather than just hearing about it.
