# GRECKO landing page — design notes

The page has one job: move a single counterintuitive idea from our heads into a
stranger's head, fast.

> **Counter-swarm defense is won on economics, and the winning move is choosing
> what to let through.**

Every decision below serves comprehension of that one idea. Decoration that did
not serve it was cut.

## The 10-second / 30-second test

- **10 seconds (sound off):** the hero headline — *"The breakthrough isn't a
  better missile. It's knowing which drones to ignore."* — plus a looping
  side-by-side sim with two live **net-cost** counters. Legacy's balloons; GRECKO's
  stays flat. A viewer who watches the counters can state the thesis without
  reading body copy.
- **30 seconds:** the Beat 2 console makes the *decision* legible — value ranks on
  every track, `HOLD FIRE` markers on the two sacrificed drones, a streaming
  decision log, and a verdict card with the dollar swing.

## Why money, not percentages

Loss is felt, not computed. Percentages ("18% lower cost-per-kill") are accurate
but inert. Two ledgers filling in real time — **spent + damage taken = net cost**
— turn the same fact into something you watch happen to someone. The hero demotes
percentages entirely; the bottom line is a dollar figure.

## Why a visible decision log

The novelty *is* the choice, and people believe what they watch happen. A
"trust our allocator" claim is weak; a log that streams

```
t+2.1  [HOLD] G  Track 9 · rank 11/11, $1.2k → HOLD FIRE, preserve magazine
t+3.4  [SOFT] G  Track 3 · $1.4k drone → soft-kill $800, kinetic reserved
t+4.0  [DRY]  L  Magazine empty · 2 tracks still inbound, no rounds left
t+4.6  [LEAK] L  Track 8 · $11.8k HVT leaked — undefended
```

turns the allocator from a black box into a visible reasoner — and makes legacy's
**failure** as legible as GRECKO's success.

## The three-beat arc

Understanding is built, not dumped, so the page is a narrative, not a feature list:

1. **You think it's an aiming problem.** Hero poses the 9-vs-11 setup.
2. **It's an economics problem.** The console + ledgers + verdict deliver the turn.
3. **And the swarm adapts.** Intent prediction and adversarial self-play, labeled
   **roadmap — not yet shipped**, so it reads as next-gen without overclaiming.

## Honesty constraints (non-negotiable for this audience)

- The browser sim is a faithful JS re-implementation of the GRECKO allocation
  logic, not the Python engine running live. The CLI section shows the real engine.
- The scenario is **tuned, not rigged**: high-value targets fly slightly slower, so
  engage-nearest genuinely deprioritizes them by distance and leaks one. The
  mechanism is legitimate; we just chose a raid that exhibits both wins clearly.
- Both advantages are demonstrable independently. Toggle **effector substitution
  off** and GRECKO still beats legacy on net — purely by ranking, because it never
  leaks the expensive target. Toggle it on and the spend saving stacks on top.
- Single-raid figures are labeled illustrative. Headline improvements come from the
  randomized Monte-Carlo `grecko eval`.
- The scope statement and PROVENANCE link stay intact. Nothing implies fielded or
  weaponized capability.

## Craft

- **Palette is money-coded, not generic.** Tactical-night navy (`#05080f`) with a
  phosphor-green GRECKO brand (`#00f5a0`), but the working colors carry meaning:
  **cyan = dollars spent**, **red = damage taken**, **gold = favorable net swing**,
  **amber = hold-fire**. This avoids the default "near-black + one acid accent" look
  by giving each accent a job.
- **Type as ledger.** Space Grotesk (display) for headlines, **JetBrains Mono for
  every dollar figure, the decision log, and the CLI** so numbers read like an
  accounting tape, Inter for body.
- **Meaning never rests on color alone.** Shapes carry it too: ▲ high-value target,
  ● low-value drone, dashed ◇ + `HOLD` text for sacrificed, ◆ asset, and `X`
  markers on leaks — so it survives grayscale and color blindness. The decision log
  pairs every color with a text tag (`[HOLD] [SOFT] [KINETIC] [LEAK] [DRY]`).

## Interaction that deepens understanding (not ornament)

- **Harder raid** — more drones than magazine; the net gap widens.
- **Effector substitution toggle** — watch the kinetic-vs-soft-kill trade move the
  ledger live, isolating that pillar from value ranking.
- **Timeline scrubber** with annotated decision markers — stop on any hold-fire or
  leak and read why. Implemented by deterministic re-simulation, so any time `t`
  reconstructs exactly.

## Accessibility

- Full keyboard support (native buttons + range input, visible focus rings).
- `prefers-reduced-motion`: hero and console settle to an annotated final frame with
  the verdict filled in — the same understanding without animation. The feint
  illustration renders static.
- Canvas work avoids layout thrash; sims pause until scrolled into view via
  `IntersectionObserver`.
