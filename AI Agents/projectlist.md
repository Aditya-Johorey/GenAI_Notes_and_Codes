Here's the rewritten list — this time each project has a concrete **MCP component built in** (either building an MCP server, consuming one, or both), since MCP is now the de facto standard for how agents connect to tools and to each other in 2026.

---

## 1. Personal Ops Agent with a Custom MCP Server (Easy)
**Industry parallel:** exec assistant / customer-support copilot
**What they build:** An agent that manages email, calendar, and a task tracker — but instead of using pre-built integrations, students **build their own MCP server** exposing 3–4 tools (send_email, create_event, add_task, check_availability), then connect it to a LangGraph agent as the client.
**Stack:** LangGraph (orchestration/state) + a hand-built MCP server (Python SDK for MCP) + Mem0 or Zep for persistent memory.
**New concepts taught:** how MCP actually works under the hood (tool discovery, schemas, transports — stdio vs SSE vs streamable HTTP), memory across sessions, human confirmation before risky actions.
**Why it's in demand:** every company building an internal agent platform right now needs engineers who can wrap *their own* internal APIs as MCP servers, not just consume Gmail's.

Link to project: https://github.com/Aditya-Johorey/personal-ops-agent-using-mcp

## 2. Agentic RAG Assistant Behind an MCP Interface (Easy–Medium)
**Industry parallel:** legal/financial/medical research copilots
**What they build:** An agentic RAG loop (retrieve → critique → reformulate → answer with citations) — then students **expose the whole RAG pipeline itself as an MCP server**, so any other agent or framework (CrewAI, Claude Agent SDK, etc.) can call it as a "research tool."
**Stack:** LangGraph for the loop, Qdrant/Weaviate for vectors, MCP server wrapping the retriever, LangSmith/Langfuse for tracing and retrieval-quality eval.
**New concepts taught:** self-correction loops, citation grounding, agent evaluation, and — new here — **designing a good MCP tool interface** (naming, schemas, error responses) so other agents can use it reliably.

## 3. Multi-Agent Business Workflow over MCP + A2A (Medium)
**Industry parallel:** sales ops, recruiting pipelines, insurance claims triage
**What they build:** A role-based crew (intake → research → drafting → review) where each agent's tools are served over MCP, and the agents themselves talk to each other over **A2A**, so the crew isn't locked into one framework — the "research agent" from Project 2 can literally be reused here.
**Stack:** CrewAI (role-based, native MCP + A2A support) or LangGraph, with each agent's MCP toolset kept separate and swappable.
**New concepts taught:** agent-to-agent handoff protocols, task decomposition, and cross-framework interoperability — plugging an agent built in one framework into a system built in another, which is a real architectural pattern now, not a hypothetical.

## 4. Computer-Use Agent Wrapped as an MCP Tool (Medium–Advanced)
**Industry parallel:** QA automation, RPA replacement, portal data entry
**What they build:** A browser/desktop-controlling agent (fills forms, scrapes dashboards, tests a web app) built with Claude Agent SDK or OpenAI computer-use + Playwright — then **packaged behind an MCP server** so it becomes a reusable "browser" tool that any other agent in the system (including the ones from Projects 1–3) can call instead of needing its own browser logic.
**New concepts taught:** perception-action loops on unstructured UI, error recovery, safety rails around autonomous UI actions, and MCP resource/tool versioning for a tool that's expensive and risky to call.
**Why it's in demand:** this is the highest-value, least-solved automation category right now — "no API exists for this system" is still most of enterprise software.

## 5. Production Multi-Agent Platform: MCP + A2A + Full Observability (Advanced)
**Industry parallel:** an internal agent platform a platform team would actually ship
**What they build:** Students assemble everything above into one platform: multiple MCP servers (their own + third-party) registered in a central **MCP gateway/registry**, agents from different frameworks coordinating over A2A, human-approval gates on high-stakes MCP tool calls, and full tracing/cost/latency monitoring per tool call.
**Stack:** LangGraph as the backbone orchestrator, MCP gateway (e.g., mcp-agent or a custom registry) for centralized auth/discovery across all tool servers, A2A for cross-agent messaging, Langfuse/LangSmith for observability, a guardrails layer for input/output filtering on MCP tool boundaries.
**New concepts taught:** MCP server governance at scale (auth, rate limits, permissioning per agent), checkpointing/time-travel debugging, and the five things that separate a demo from a product — cost, latency, efficacy, assurance, reliability.

---

### Why building MCP servers matters as much as consuming them
Most tutorials only teach students to *call* existing MCP servers (Gmail, Slack, GitHub). The real skill gap in the market is **building** MCP servers that wrap a company's own internal systems and reusing agents as tools for each other — that's what Projects 1, 2, and 4 specifically train, and Project 5 forces students to govern a whole fleet of them at once, which is the exact job description showing up for "AI platform engineer" roles right now.

Want the syllabus version next — weekly milestones, MCP server specs for each project, and a grading rubric?
