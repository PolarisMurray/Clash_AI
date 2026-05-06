# Project Expo — Talking Script

Designed for the final-event station: 8–10 minute walkthrough, one figure
per "slide" if I'm using a laptop, or one figure per panel if it becomes
a poster.

> Bracketed cues `(emphasis)` mark spots to slow down and speak deliberately.
> `(switch slide)` marks the cue to advance to the next figure.

---

## Opening — 30 seconds

Hi, I built a Clash Royale agent that **only looks at the screen** to decide
what to do. No game APIs. No internal state. Just pixels in, action out.

The interesting part isn't the game — it's that I had to fuse a bunch of
different kinds of information (image patches, detected units, the player's
hand, elixir count, past actions and rewards) into one neural network that
makes decisions over time.

I'll walk through six figures. Each one corresponds to a stage of the system.

`(switch to figure 1)`

---

## Figure 1 — The architecture (≈ 90 s)

This is the whole system on one slide. Six stages, left to right.

**Stage 1.** Input is a raw screenshot. That's it.

**Stage 2.** A YOLOv8 detector scans the frame and produces a structured
record per object: class ID, bounding box, confidence, ally-vs-enemy
inferred from the box's vertical position, and an optional tracker ID.

**Stage 3** — this is the part of the system that's mine.
Image patches, YOLO detections, hand cards, elixir, the previous action, the
previous reward, the return-to-go target, and the timestep — every kind of
information becomes a token. Each color in the diagram is a different token
type. They all live in the same embedding space.

**Stage 4.** A **Perceiver IO** encoder takes the variable-length token cloud
and compresses it into a fixed number of latent vectors using cross-attention.
`(emphasis)` The key idea: the input length doesn't matter — the output is
always the same size.

**Stage 5.** A **Decision Transformer** takes those latents over multiple
timesteps and runs causal attention. `(emphasis)` This is where space, time,
modalities, and rewards all interact in *one* attention operation, instead of
the older "spatial-then-temporal" two-stage design.

**Stage 6.** Three small heads predict: which card to play, where to play it
on a 36×64 grid, and how many frames from now to play it.

The grey strip at the bottom lists everything I deleted from the original
codebase — OCR pipelines, ResNet classifiers, episode cutting. All gone.

`(switch to figure 2)`

---

## Figure 2 — YOLO on 12 real screenshots (≈ 60 s)

This is stage 2 actually running. I sampled 12 frames evenly from a gameplay
recording and pushed each one through my YOLO wrapper.

`(point at boxes)` You can see it correctly picks up:
- The four corner towers (red boxes labeled `queen-tower`)
- Mid-field troops — knights, archers, hog-rider
- Even individual hand-card icons in some frames

Two things to take away here:

1. **The detection count varies.** Some frames have 1 detection, some have 6.
2. **All vision happens with one model.** No second OCR pass, no second ResNet
   for cards. One YOLO. That's a deliberate simplification.

`(switch to figure 3)`

---

## Figure 3 — Detection statistics (≈ 60 s)

Same 12 frames, aggregated.

`(point left)` `queen-tower` dominates the class distribution because it
appears in almost every frame.

`(point middle)` Confidence is concentrated between 0.3 and 0.9 — the model
is confident when it sees something.

`(point right)` This curve is **the most important thing on this slide**.
Detections per frame swing from 1 to 6, sometimes within the same battle.
`(emphasis)` This is *exactly* why I picked Perceiver instead of a vanilla
Vision Transformer. A vanilla ViT needs a fixed-length input; Perceiver
doesn't care.

`(switch to figure 4)`

---

## Figure 4 — Training curves (≈ 80 s)

Proof that the network is wired correctly.

I scaled the model down to 5.4 million parameters so it would fit on my
MacBook's MPS device. Then I trained it for 5 epochs — about 640 gradient
steps — on synthetic random trajectories.

`(point at the loss panel)` Total loss falls smoothly from 11 to 4.8.

`(point at component losses)` The position loss is the largest of the three —
makes sense, because the position head has 2,304 output classes (the 36×64
grid). The select head only has 5 (4 cards + no-op), so its loss drops
quickly.

`(emphasis)` **Important caveat:** the data here is synthetic — random
trajectories from a generator. What this experiment proves is that the
**network architecture, the loss masking, the backprop, and the LR schedule
all work together correctly**. It does *not* claim the model has learned to
play Clash Royale. Real-replay collection is the next step in this project.

`(switch to figure 5)`

---

## Figure 5 — End-to-end on a real screenshot (≈ 90 s)

This is my favorite figure, because it grounds the abstract architecture
diagram in real numbers.

I take one real screenshot — top-left — push it through the entire pipeline,
and label every intermediate tensor's shape.

`(point top-right)` Image goes in as `(1, 4, 3, 64, 64)`. After tokenization
it becomes `(4, 36, 256)` — 4 frames, 36 tokens per frame, 256-dimensional
embeddings. Perceiver compresses 36 tokens into 16 latents per frame. The
Decision Transformer then outputs a 256-dim hidden state per timestep.

`(point at bottom panels)` And the three head outputs are real:
- Select logits over the 5 classes (left).
- A 36×64 position logit grid (middle, as a heatmap).
- 21 delay bins (right).

These outputs are random because the model is at initialization — but `(emphasis)`
**the shapes match end-to-end**. Any shape mismatch would have crashed.

`(switch to figure 6)`

---

## Figure 6 — Offline-RL data layout (≈ 60 s)

Last slide is about the *data*. Decision Transformers train on entire
trajectories, not just single frames.

`(point top-left)` The grey thin line is per-frame reward. The thick red line
is the **return-to-go** — sum of future rewards. Computed backwards from the
end of the episode. The orange vertical lines are frames where the human
actually played a card.

`(point top-right)` The action histogram. Notice **no-op dominates**. That's
honest to the game: you spend most of your time waiting for elixir.

`(point bottom-left)` This is my **weighted sampler**. Action frames and the
frames just before them get higher weight, so during training the model
isn't drowned in inactive frames.

`(switch back to figure 1 if showing)`

---

## Closing — 30 seconds

Five things this project does:

1. **Pixels only** — no game APIs.
2. **Unified token schema** — image, objects, cards, rewards, all in one
   sequence.
3. **Perceiver IO** for variable-length input.
4. **Decision Transformer** for causal sequence modeling, no manual
   spatial/temporal split.
5. **Massive cleanup** — the original codebase had OCR, three classifiers,
   episode cutting, a JAX StARformer. All replaced by 13 PyTorch files.

The whole thing is end-to-end differentiable. Figure 4 shows it learns;
figure 5 shows the shapes line up on a real frame.

What's missing: a real-replay collector. That's the next two weeks of work.

Thanks — happy to take questions.

---

## Likely audience questions (be ready)

**Q: Why Perceiver instead of a vanilla ViT?**
A: ViT's self-attention is O(N²), and our input is multimodal and
variable-length — image patches *plus* a different number of YOLO
detections every frame. Padding to a max length wastes compute and
the cost still grows quadratically. Perceiver IO uses a fixed K latents
that cross-attend the input, dropping the cost to O(M·K).

**Q: Why a 36 × 64 grid instead of regressing continuous (x, y)?**
A: Classification is more stable when supervision is sparse, and human
deployment positions are pretty discretely "in this lane / this side
of the bridge". The model also supports continuous regression if you
flip `pos_mode = "xy"`. I just didn't ablate it.

**Q: Where's the real training data?**
A: Not collected yet — this is the honest limitation. The .npz schema is
defined and the loader is tested with synthetic data. Real replays need
a screen-recorded touch-event log, which is the next sprint.

**Q: What did Claude/AI tools help with?**
A: I used Claude Code to draft and review code. Every line was reviewed
and edited by me, the architecture is mine, and the integration / testing
is mine. I treat it like StackOverflow + a textbook — useful, but the
understanding is mine.

**Q: How is this different from KataCR's StARformer?**
A: StARformer hard-codes "spatial attention then temporal attention" and
has a 32×18 unit-feature grid as its state. My version uses Perceiver
to compress *any* set of multimodal tokens — image patches, objects,
cards, scalars — and lets the Decision Transformer attend across both
time and tokens jointly in one operation. Adding a new modality
(audio, chat) doesn't change the architecture.
