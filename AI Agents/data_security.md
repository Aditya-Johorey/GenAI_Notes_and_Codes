# Keeping People's Information Safe

## Why are we even talking about this?

Up until now, you've only tested your projects with fake information — a made-up name, a made-up phone number.

Soon, you'll build these tools for real businesses. And real businesses deal with real people's information — their customers' names, phone numbers, home addresses, even health details.

Here's the simple way to think about it:

> **If you were building something that touched a stranger's personal information, wouldn't you want the person building it to be careful?**

That's the whole topic in one sentence. You don't need to be a computer expert or a lawyer to be careful. You just need to build the habit of asking one question, every time:

> **"If this got seen by the wrong person, would it hurt or embarrass someone?"**

If yes — treat it carefully. That one question will get you through 90% of this topic.

---

## The golden rule

> **The safest information is the information you never collected in the first place.**

Before you save anything — a name, a message, a phone number — ask yourself:

> **"Do I actually need to keep this, or am I just keeping it because it was easy to?"**

**Think of it like this:** imagine a receptionist at a clinic. A good receptionist writes down your name and why you're visiting. A nosy receptionist also writes down what you were wearing, what car you drove, and a comment about your mood — "just in case it's useful later."

One of these is doing their job. The other is creating a pile of information that could embarrass someone if the wrong person ever read it.

Be the first kind of receptionist.

---

## What counts as "personal information"?

You don't need a legal definition. You just need to recognize it when you see it — the same way you recognize a stop sign without reading the rulebook.

Personal information includes things like:

- A person's **name**
- Their **email address**
- Their **phone number**
- Their **home address**
- **Payment details** — card numbers, bank details
- **Health information** — an illness, a medicine they take
- Anything that could point to *one specific person*, even if it's a little indirect

That last one is worth slowing down on. Sometimes information looks harmless but isn't:

- An order number by itself means nothing. But the moment it's sitting next to someone's name in the same spreadsheet, it's now personal information.
- A support chat might not have a name typed anywhere — until someone writes "Hi, this is Priya, calling about my son's medicine." Now it's personal, and it's sensitive too.

**The one-line test to remember:**

> *"If this information leaking out would embarrass or harm the person it belongs to, treat it as sensitive. Full stop."*

---

## Things you should never save

Here are the clear "don't touch this" rules. Think of each one as a locked door you shouldn't try to open.

### 1. Payment information
Never save someone's card number in a spreadsheet, a form, or anywhere your automation stores things.

**Why:** A spreadsheet has no lock on it. If it's ever accidentally shared, every card number inside is now exposed. Instead, always let a proper payment system (like the checkout screen from a payment company) handle this — that's exactly what it's built for.

### 2. Passwords
Never ask your automation to collect or hold onto someone's password, not even "just for a second."

**Why:** There's no such thing as "temporarily" storing a password safely outside a real password system. Even a few seconds in the wrong place is too long.

### 3. Sensitive personal topics
If someone talks to your automation about their health, a legal issue, or something private — don't dump the entire conversation into a general spreadsheet that lots of people on the team can open.

**Why:** Picture someone on the team who has nothing to do with that conversation scrolling past it while looking for something else. That's an unnecessary, avoidable exposure.

### 4. Entire conversations, just because you can
Don't save every single word of every conversation by default. Only save what you actually need — like "was this solved, yes or no" — instead of everything.

**Why:** Information you save "just in case" never expires and nobody's really watching it. It just sits there as risk, with no real benefit to anyone.

---

## The four things every privacy rule in the world agrees on

Every country has its own privacy laws, and they're written in complicated legal language. You don't need to read those laws. You just need these four ideas, explained simply:

**1. Only collect what you truly need.**
Don't add a "date of birth" box to a form just because it might be useful one day. If you can't explain in one sentence *why* you need a piece of information, don't ask for it.

**2. Tell people what you're doing with their information.**
A simple, honest sentence is enough. For example:
> "Your message will be used to answer your question and may be reviewed by our team."

**3. Let people ask you to delete their information.**
It doesn't need to be fancy. Even this is a fine start:
> "Email us and we'll remove your information."

**4. Don't keep information forever.**
Get in the habit of clearing out old records now and then, instead of letting a spreadsheet grow bigger and bigger, forever, with no plan.

### What to tell a client, in your own words:

> *"I build things with privacy in mind, but I'm not a lawyer. If your business deals with health information, financial information, or anything in a strictly regulated industry, please have that part checked by someone qualified."*

Saying this **before** a client asks makes you look more trustworthy, not less. It shows you know your limits.

---

## Simple habits to build into every project

These are small, practical things you can actually do, every time, without needing deep tech knowledge:

- **Only share the spreadsheet or file with people who truly need it.** Not the whole company — just the few people whose job actually requires it.

- **Don't put private details in labels, titles, or notes inside your tool.** For example, don't name something "Text message to Priya Sharma, 9876543210" — anyone who opens that project later can see it, even people who have no reason to.

- **Keep secret codes (passwords, access keys) in the tool's proper "secret storage" area — never typed out in plain view.** Most tools have a safe, hidden place built exactly for this. Using it takes the same effort as typing it in plain sight, so there's no excuse not to.

- **When testing, use fake or copied information — never the real, live customer list.** Testing on real information is how accidents happen, like sending a test message to hundreds of real customers by mistake.

---

## Quick stories to think through

**Story A:** A student saves every full conversation from a support bot into a spreadsheet the whole company can open — "just in case marketing wants it later."
> What's wrong? There's no real reason to keep it, and far too many people can see it.

**Story B:** A freelancer tests a client's order system using the client's real list of 3,000 customers — and accidentally sends all of them a test message.
> What's wrong? Testing should always use fake or copied information, never the real list.

**Story C:** Someone names part of their project "Reply to Rahul — his divorce case — urgent," and everyone on the team can see that title.
> What's wrong? A private, sensitive detail is now visible to people who have no reason to see it.

---

## Class activity (15 minutes)

Take a finished project from earlier in the course (yours or a sample one) and go through it like a detective. Answer these four questions:

1. What personal information does this project currently collect?
2. Can more people see it than actually need to?
3. Is anything being saved "just in case," even though nothing actually uses it?
4. Pick **at least one thing** you found, and fix it — remove it or restrict who can see it.

**A simple worksheet to fill in:**

| What information is being saved? | Do we actually need it? (Yes/No) | Who can currently see it? | Should fewer people see it? | What did you change? |
|---|---|---|---|---|
| | | | | |
| | | | | |

**End-of-class question to discuss out loud:**
*"What's one habit from today you'll now do automatically, without even thinking about it, every time you build something new?"*

---

## A few words explained simply

- **Personal information:** Anything that could point to one specific real person.
- **Sensitive information:** Personal information that could hurt or embarrass someone if it got out — health, legal, financial, and similar topics.
- **Secret storage / credentials:** A hidden, locked area in a tool made specifically for passwords and access codes, so they're never left out in the open.
- **Testing with fake data:** Using made-up or copied information to try things out, instead of risking real people's real information.

---

## One sentence to remember this whole lesson by

> **Before you save anything, ask: does someone need this, who could see it, and how long should it really stick around? If you can't answer clearly, don't save it yet.**
