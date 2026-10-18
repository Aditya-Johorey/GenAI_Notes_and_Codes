
Product details change often, so I'll check Anthropic's current documentation before writing this guide.

# A Guide to Claude's Features

I combined what I know directly about Anthropic's products with a few web searches. Most of the search results were third-party summaries, not official docs. Plan details, limits, and newer features change often, so check **support.claude.com** and **claude.com/pricing** before relying on specifics.

---

## 1. The models

Claude comes in tiers that trade off speed, cost, and depth.

- **Haiku 5.5** is the fastest and lightest, suited to quick, high-volume tasks.
- **Sonnet 5.5** is the balanced all-rounder, and the one you're talking to now.
- **Opus 5.5** is stronger on harder reasoning, long projects, and complex code.
- **Fable 5.1 and Mythos 5.1** form the top tier above Opus. They share an underlying model, and Fable adds extra safety measures for biology, cybersecurity, and LLM R&D. Access to these was briefly suspended in June 2026 under export controls and restored on July 1. Anthropic's statement is at anthropic.com/news/fable-mythos-access.
- **Claude Mythos Preview** is not public. It is used by a small number of trusted organizations through Project Glasswing.

You can switch models mid-conversation. Which models you can use depends on your plan.

---

## 2. Core conversation abilities

**Writing and editing.** Drafting, rewriting, summarizing, translating, tone adjustment, proofreading, brainstorming, and creative writing.

**Analysis and reasoning.** Breaking down documents, comparing options, working through math and logic, explaining concepts at any level, and reviewing arguments. Extended thinking lets Claude reason longer before answering on hard problems.

**Coding.** Writing, debugging, explaining, refactoring, and reviewing code in most languages. It can also run code in a sandbox to test things or analyze data.

**Vision.** Claude reads screenshots, photos, charts, and diagrams, including OCR, chart analysis, and UI debugging. It understands images but isn't an image generator. It can make visuals through code, SVG, and charts.

**File uploads.** You can upload PDFs, documents, spreadsheets, CSVs, and images for Claude to read, summarize, extract from, or transform.

**Voice mode.** Two-way voice conversations are available on claude.ai and the iOS and Android apps, on all plans including Free. The iOS app also supports voice through Apple CarPlay.

**Long context.** The claude.ai apps run a 200K-token context window, enough for long documents or extended conversations.

**Languages.** Claude works in a very wide range of languages and replies in whichever one you write in.

---

## 3. Finding current information

**Web search.** Claude can search the live web for news, prices, current office-holders, and anything that may have changed recently. It cites sources. Search can be toggled in settings.

**Research mode (deep research).** This is a long-running web investigation. Standard Research takes minutes and returns a cited brief, and you can edit the proposed plan before approving it. It suits comparisons, market scans, and literature-style questions. Connected work sources can be included.

**Image search, maps, weather, and sports.** In the apps, Claude can show inline images, place maps, weather, live scores, and standings where useful.

---

## 4. Organizing your work

**Projects.** Projects are persistent workspaces with their own knowledge files, custom instructions, and chat history. They suit recurring work such as a client, a course, or a codebase. Free plans get up to five Projects, and paid plans are unlimited with more knowledge capacity.

**Memory.** Claude can remember durable facts about you across chats. You can read and edit what it remembers under Settings, Capabilities, Memory. Memory is optional and you control it. You can also search and reference past chats.

**Incognito chats.** An incognito conversation isn't saved to your history or written to Memory.

**Styles and preferences.** Styles control tone, length, and format. There are built-in presets, and you can create a custom style from your own writing samples. User preferences hold standing instructions such as formatting habits or background about you.

---

## 5. Building things: Artifacts and file creation

**Artifacts.** Artifacts are a side-panel workspace where Claude renders code, documents, diagrams, and interactive apps you can preview, edit, version, and share. Common uses:
- Small web apps, calculators, dashboards, and games
- HTML pages and visual explainers
- Diagrams (Mermaid, SVG) and charts
- Published pages with a shareable link

**AI-powered artifacts.** Artifacts can call Claude themselves, so you can build tools like quizzes, tutors, or summarizers. They can also use connected services such as Gmail or Drive.

**Persistent storage.** Artifacts can store data across sessions, which enables trackers, journals, and leaderboards, including shared data across users.

**File creation and code execution.** Claude can create downloadable **Word (.docx), PowerPoint (.pptx), Excel (.xlsx), and PDF** files. It can also produce Markdown, HTML, CSV, and code files. It works in a sandbox computer, so it can clean data, make charts, convert formats, and edit files you upload.

**Docs and design types.** Depending on your account, Claude can make editable documents, slide decks, and visual designs (posters, UI mockups, landing pages) in dedicated editors.

---

## 6. Connecting Claude to your tools

**Connectors.** Connectors are MCP-based integrations that link your apps and data to Claude. You add them under Settings → Connectors. Examples include Gmail, Google Drive, Slack, Notion, Asana, Linear, and Shopify. Claude can then search your email, read files, create tasks, or draft messages. MCP (Model Context Protocol) is Anthropic's open standard for this. Some connectors are read-only and some can act on your behalf.

**Skills.** Skills are folders of instructions that teach Claude how to do a task well, such as your team's report format or a document-building workflow. Claude loads them when relevant, so you don't re-explain every time. Anthropic provides built-in ones (Word, PDF, PowerPoint, Excel), and you or your organization can add your own.

**Plugins.** Plugins bundle skills, commands, and connectors for a role or workflow. Organizations can curate a catalog of them.

---

## 7. Agentic products

These are separate surfaces built around Claude doing multi-step work.

| Product | What it does |
|---|---|
| **Claude Code** | An agentic coding tool for the terminal, desktop app, IDEs (VS Code, JetBrains), and mobile. It reads your codebase, edits files, runs commands, and handles whole tasks. |
| **Claude Cowork** | A desktop app for non-developers. It handles multi-step knowledge work such as research, documents, and file organization, and can use Chrome, Excel, and other tools. |
| **Claude in Chrome** | A browsing agent that navigates and acts on websites for you. |
| **Claude in Excel** | A spreadsheet agent for formulas, cleanup, and models. |
| **Claude in PowerPoint** | A slides agent. |
| **Claude for Google Workspace** | A sidebar in Docs, Sheets, and Slides. |
| **Claude Tag** | A Slack-based interface where anyone can tag @Claude to delegate tasks. |
| **Claude Design / Claude Science** | Newer, plan-dependent offerings for visual design and scientific work. I don't have detailed documentation on these, so check the official site. |

The Claude mobile app can also access Claude Code and Cowork remotely.

---

## 8. For developers

- **Claude API and Claude Platform**: build your own applications with the models (model strings like `claude-sonnet-5-5`).
- **Tool use / function calling, streaming, batch processing, and SDKs.**
- **Web search and MCP support in the API.**
- **Prompt engineering guidance** at docs.claude.com.

Docs live at **docs.claude.com**, and the Claude.ai help center is at **support.claude.com**.

---

## 9. Plans (as reported by third-party sources)

- **Free**: chat on every device, memory, web search, file uploads, code execution, connectors, and extended thinking.
- **Pro**: $20/month, or $17/month billed annually, with more usage, Claude Code, Cowork, unlimited Projects, and more models.
- **Max**: $100/month (5× Pro usage) or $200/month (20× Pro), for heavy users and early access to new features.
- **Team**: $25 per seat for standard or $125 for premium seats, requiring a minimum number of seats.
- **Enterprise**: SSO, audit logs, admin controls, enterprise search, and dedicated support.

---

## 10. Limits and good habits

- Claude can be wrong. Verify important facts, especially on anything legal, medical, or financial. It isn't a lawyer, doctor, or financial advisor.
- It has a knowledge cutoff and uses search for current information.
- It won't help with weapons, malware, or content that harms children, and it won't reproduce copyrighted works like song lyrics.
- Ads: Claude products don't display ads or sell placement in conversations.
- **Getting better results**: give context and goals, show examples of what you want, say the format and length you need, and use Projects, styles, and Skills for anything you repeat.

---

I can go deeper on any section, such as setting up Projects, building a Skill, choosing between Claude Code and Cowork, or building a specific Artifact. I can also put this guide into a Word, PDF, or doc file if you'd like to keep it.
