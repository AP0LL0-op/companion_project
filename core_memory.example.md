# Core Facts — <your name>

Copy this to `core_memory.md` and rewrite it as yourself. That file is
gitignored and will never be committed.

This is **substrate, not memory**. It's injected into every single
conversation rather than retrieved when relevant, so it should be things that
are true regardless of what's being discussed. Keep it to a page or so — it
sits in the cached prefix, so length costs startup time once rather than
per-turn, but it also competes with the conversation for context.

Write it in the third person, as notes she holds about you. Plain markdown on
purpose: when she gets something wrong, open this file and fix the line. That
is far easier than arguing with a vector store, and it's why nothing in the
code ever rewrites this file.

---

## Household

- **<name>** — partner/spouse. Anything ongoing that shapes daily life.
- **<name>** — child, age, what they're into.
- Pets, by name. She will use them.

## Place

- Where you live, in the terms you'd actually use. "Five acres with a pond"
  tells her more than a city name.

## People outside the house

- Friends, family nearby, anyone who recurs in conversation. A one-line note
  on who they are to you is enough — she'll fill in the rest over time.

## Work

- What you do, where, and what it costs you. If a job drains you by evening,
  say so; it changes how she reads a tired answer.

## Health

- Only what you want present in every conversation. Chronic things, ongoing
  investigations, medication changes. This is the section to be most
  deliberate about — see the note on sensitive material below.

## Craft and hobbies

- The specific version, not the category. "Fly tying, mostly soft hackles for
  small streams" gives her something to actually talk about; "likes fishing"
  does not.

## Projects

- What you're building and why. Include this system if you want her to
  understand her own situation.

## Rhythm

- When you talk to her, and what you're usually doing. Driving home, late at
  night, while working. Context she can't infer.

## How you think

- How you want to be talked to. Whether you want pushback. Whether you'd
  rather hear "I don't know" than a guess.

## Standing context

- Anything that should never drift. What this relationship is and isn't.
  Boundaries you want held.

---

**On sensitive material.** Everything here is present in every exchange,
including ones where someone else is in the room. If something would be
damaging to surface at the wrong moment, it does not belong in this file.
The reference implementation keeps a separate `restricted.md` describing a
tier behind its own retrieval gate — and deliberately leaves that tier
*unpopulated*. That is a reasonable default: material a system can't be
trusted to withhold at the right moment is safer not stored at all.
