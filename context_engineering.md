# Context Engineering (the bigger picture around RAG)

Everything in this repo teaches **retrieval** — turn data into vectors, store them,
find the nearest ones by meaning. But retrieval is only *part* of a bigger job:
deciding **what to put in front of the LLM** when you ask it something. That job is
**context engineering**.

> **One-line idea:** the model is a brilliant contractor with amnesia. Context
> engineering is preparing the exact **briefing packet** it reads before doing the
> job — the right facts, in the right order, at the right size.

---

## Why it exists

An LLM only "knows" two things:
1. What it was **trained on** (fixed, and possibly out of date).
2. Whatever text you hand it **in the prompt** — its *context window*.

You can't change (1) without retraining. So all your control is in (2). Context
engineering is the discipline of assembling that window well.

It's the grown-up version of "prompt engineering." Prompt engineering was mostly
about **wording the question**. Context engineering is about **everything that goes
in the window**.

---

## What goes into the context window

| Ingredient | What it is | Example |
|---|---|---|
| **Instructions** | system prompt, rules, tone, output format | "Answer only from the context. Reply in JSON." |
| **Retrieved knowledge** | relevant docs/chunks — **this is where RAG lives** | the top-k chunks from your vector search |
| **Examples** | few-shot demonstrations of the task | 2–3 solved question/answer pairs |
| **Memory / history** | prior conversation, earlier decisions | the last few chat turns |
| **Tool outputs** | results from function/API calls | a database row, a web-search result |
| **The user's question** | the actual request | "What's the refund window?" |

The window is **finite**, so context engineering is also about what to **leave out,
compress, or summarize** — and in **what order** (models pay extra attention to the
start and end of a long context).

---

## How it relates to this repo

Retrieval (what you're learning) is **one input** to context engineering.

```
embeddings → vector search → RAG  →  CONTEXT ENGINEERING  →  the LLM's answer
└────────── this repo teaches this ──────────┘   └── assembling the whole prompt ──┘
```

| | Vector DB / RAG (this repo) | Context engineering |
|---|---|---|
| **Question it answers** | "Which of my docs are relevant?" | "What should go in the prompt right now?" |
| **Scope** | one technique: search by meaning | the whole prompt: instructions + docs + history + tools + examples |
| **Role** | a *component* | the *system* that uses that component |
| **Output** | a ranked list of relevant chunks | a finished prompt for the LLM |

So they're **not competing ideas.** When a RAG system drops its top chunks into the
prompt, that placement *is* context engineering. But context engineering also covers
things retrieval doesn't: how much history to keep, how to order information, when to
summarize vs. include raw text, how to fit tool results in, and how to avoid stuffing
the window with junk that dilutes the answer.

---

## The core skills (what to actually practice)

1. **Select** — pick only what's relevant (good retrieval helps here) instead of
   dumping everything.
2. **Order** — put the most important material where the model looks hardest
   (often near the end of the prompt).
3. **Compress** — summarize long history or documents so they fit the budget.
4. **Instruct** — a clear system prompt. The single most important RAG instruction:
   *"Answer only from the provided context; if it's not there, say you don't know."*
   That one line is what stops hallucination.
5. **Budget** — track tokens; know what to drop when the window fills up.

---

## Common failure modes it fixes

- **Hallucination** — model invents an answer → give it the facts + tell it to stick
  to them.
- **Lost in the middle** — key info buried in a huge context gets ignored → retrieve
  less, but more relevant; order it well.
- **Context pollution** — irrelevant chunks crowd out the good ones → better
  retrieval + re-ranking ([Phase 5](learning_notes.md#deep-dive-phase-5--better-retrieval-quality)).
- **Runaway cost/latency** — oversized prompts → compress and trim.

---

## Where it fits your learning path

You meet context engineering the moment you wire a **real LLM call** into your RAG
pipeline. The "answer only from this context" system prompt is your first piece of
it. From there: manage history, order the chunks, and watch your token budget.

**Related in this repo:**
- [learning_notes.md](learning_notes.md) — the retrieval half (embeddings → RAG).
- [rag_architecture_types.md](rag_architecture_types.md) — the RAG designs that
  *produce* the retrieved context.

> **Takeaway:** retrieval finds the right information; context engineering decides
> how to hand it to the model. You need both — this repo gives you the first, and
> this note points at the second.
