# HT-PCWM: Hierarchical Temporal Predictive Coding World Model

A video prediction model that learns to **imagine what happens next** by building an internal mental model of how the world moves.

---

## Architecture Overview

```
                    ┌─────────────────────────────────────────┐
  Frame t+1 ──────→ │           INFERENCE LOOP                 │
  (target)          │  ┌─────────────────────────────────┐    │
                    │  │  Decode → Error → Fix → Repeat   │    │
                    │  │     ↑         ↓                  │    │
                    │  │     └─────────┘                  │    │
                    │  └─────────────────────────────────┘    │
                    │              ↑                           │
  Frame t ────────→ ENCODER ──→ z1 (8×8) ───→ DECODER ──→ Prediction
                    (compress)    └── z2 (4×4) ──┘
                                  └── z3 (2×2) ──┘
                                  (three-level mind)
```

**Analogy:** Your brain predicts where a bouncing ball will land. HT-PCWM does the same: watch frame → build mental model → predict next frame.

---

## Modules

### Encoder — "The Compressor"
```
  64×64 image (4,096 values) ──→ 8×8×128 latent (8,192 values)
```
**What:** Compresses pixels into a compact "thought" that captures edges, textures, patterns.
**How:** 3 convolutional layers that shrink space but expand features [32→64→128].
**Analogy:** Like reading a novel and writing a one-sentence summary — only the gist survives.

### Latent Hierarchy — "The Three-Layer Mind"
```
  z1 (8×8×128) = FINE DETAILS          │  z2 (4×4×128) = COARSE DYNAMICS     │  z3 (2×2×128) = WORKING MEMORY
  ────────────────────────────────────  │  ──────────────────────────────────  │  ────────────────────────────────────
  • Exact edge positions                │  • Motion direction                  │  • Scene-level context
  • Textures, shapes                    │  • Speed of movement                 │  • What's been happening
  • Where things are                    │  • Where things are going            │  • Patterns over time
  Like reading individual letters       │  Like understanding the plot         │  Like remembering the last few seconds

  Role: The renderer                    │  Role: The planner                   │  Role: The memory
  Values: 8,192                         │  Values: 2,048                       │  Values: 512
  Only z1 → Decoder → pixels            │  Predicts what z1 should be          │  Predicts what z2 should be
```
**Why three layers:** z1 sees WHAT (details), z2 sees HOW (dynamics), z3 remembers CONTEXT (working memory). Together they capture what happened, what's happening, and what's been happening.
**Communication:** z2 predicts what z1 should look like. z3 predicts what z2 should look like. z1 feeds details back up. They talk to refine each other.
**Why z3 is 2×2:** Very abstract, biologically grounded (PFC has coarse spatial but rich temporal). Forces specialization in temporal patterns rather than spatial redundancy.

### Decoder — "The Artist"
```
  8×8×128 latent ──→ 64×64 image
```
**What:** Takes the compressed thought and paints back a full frame.
**How:** 3 transposed convolutional layers [128→64→32→1].
**Analogy:** Like reading a summary and writing a new novel — a best-effort reconstruction.

### Learned Predictor — "The Smart Advisor"
```
  ┌─────────────────────────────────────┐
  │ INPUT:                              │
  │ • Current latents z1, z2, z3        │
  │ • Errors (what went wrong)          │
  │ • Frames (context)                  │
  │                              ↓       │
  │ NEURAL NETWORK (2 layers, CNN)      │
  │                              ↓       │
  │ OUTPUT: How to fix z1, z2, and z3   │
  └─────────────────────────────────────┘
```
**What:** Replaces fixed math (`error × 0.5`) with a neural network that learns optimal fixes.
**Why better:** Fixed math can't handle occlusions, rotations, or collisions. The learned predictor adapts to context — learned from millions of examples.
**Analogy:** Like a GPS that learns road conditions (smart) vs. always saying "turn left" (dumb).

### Inference Loop — "Iterative Refinement"
```
  Step 1: Decode z1 → frame prediction
  Step 2: Compare with target → error
  Step 3: Ask predictor how to fix z1, z2, z3
  Step 4: Update z1, z2, z3
  Step 5: Check if done (changes tiny?) → STOP if yes
  ──────────── repeat ────────────
```
**What:** Cycles of predict→check→fix until convergence.
**Typical steps:** 1-5 iterations (adaptive stopping decides).
**Analogy:** Like sculpting — rough cut → look at target → chisel → repeat until perfect.

### Adaptive Stopping — "Knows When Good Enough"
```
  Step 1: ████████████████ (big change)    [compute used]
  Step 2: ████████ (medium)
  Step 3: ██ (tiny) → STOP!               [saves 40-60% compute]
  Step 4: █ (skipped)                     [fixed methods waste this]
  Step 5: █ (skipped)                     [fixed methods waste this]
```
**What:** Stops early when updates drop below threshold (1e-4).
**Savings:** 40-60% less compute vs fixed 5-step loop.
**Analogy:** Like knowing when a painting is done — adding more brushstrokes won't help.

### SIGReg — "Traffic Controller for Latents"
```
  WITHOUT SIGReg:               WITH SIGReg:
  z1: -1000 to +10000  ← CHAOS │ z1: -1 to +1 (Gaussian) ← STABLE
  z2: -500 to +5000    ← CRASH │ z2: -1 to +1 (Gaussian) ← OK
  z3: -200 to +2000    ← CRASH │ z3: -1 to +1 (Gaussian) ← OK
  Training collapses!           Training runs 100+ epochs ✓
```
**What:** Penalty term that keeps latents bounded around Gaussian (mean=0, std=1).
**Why needed:** Without it, latents explode at epoch 12 (we saw this!).
**Analogy:** Air traffic controller — keeps planes in their lanes, prevents crashes.

### Energy Function — "The Error Score"
```
  Energy = 1.0 × Frame Error    (pixel accuracy — most important)
         + 0.5 × Latent1 Error  (do fine details match coarse prediction?)
         + 0.5 × Latent2 Error  (do dynamics follow learned rules?)
         + 0.5 × Latent3 Error  (does context match scene-level prediction?)
         + 0.1 × SIGReg         (are latents well-behaved?)
```
**What:** Measures how wrong our prediction is. Lower = better.
**Frame Error:** ||target - prediction||²
**Latent1 Error:** ||z1 - predictor(z2)||²
**Latent2 Error:** ||z2 - predictor(z3)||²
**Latent3 Error:** ||z3 - transition(z3)||²

---

## Downstream Tasks

| With z1,z2 | With z1,z2,z3 |
|------------|--------------|
| Frame prediction | Scene understanding |
| Object tracking | Long-range prediction |
| Motion recognition | Temporal reasoning |
| Collision detection | Event detection |
| Action recognition | Working memory tasks |

**z1,z2 = perception + short-term prediction** (what's happening now, what happens next)
**z1,z2,z3 = perception + prediction + understanding** (what's happening, what's next, why it's happening)

---

## Training

```bash
# Train from scratch
python train.py --epochs 50

# Resume training
python train.py --resume --epochs 100

# Resume from specific checkpoint
python train.py --checkpoint checkpoints/model_epoch_3.pt --epochs 100
```

| Flag | Description |
|------|-------------|
| `--resume` | Resume from latest checkpoint |
| `--checkpoint PATH` | Resume from specific checkpoint |
| `--epochs N` | Set total epochs (default: 50) |

**Model:** ~5.14M parameters | **Input:** 64×64 grayscale | **Latent:** z1(8×8×128), z2(4×4×128), z3(2×2×128)

---

## Data Flow Example

```
INPUT: Frame t (digit "0" at top-right) + Frame t+1 (digit "0" moved down-right)

  1. ENCODE: Frame t → z1 → z2 → z3 (compresses scene + dynamics + context)

  2. INFERENCE (3 iterations typical):
     Iter 1: Decode z1 → prediction → big error → predictor says "move z1 right + down"
     Iter 2: Decode z1 → prediction → small error → predictor says "slight adjust"
     Iter 3: Decode z1 → prediction → tiny error → STOP (adaptive)

  3. DECODE: Final z1_pred → reconstructed Frame t+1 (digit "0" at new position)

  4. ENERGY: Compare prediction vs target → score = 0.12 (low = good)
```

---

## Checkpoints

Saved to `checkpoints/` as `model_epoch_N.pt`. Metrics in `checkpoints/metrics.csv`.
