# Emotion Recognition with CLIP

Zero‑shot *and* fine‑tuned facial‑emotion recognition built on top of **OpenAI CLIP**.  
The goal is to map images of faces to one of seven basic emotions — *angry, disgust, fear, happy, sad, surprise,* and *neutral* — with strong performance even when labelled data are scarce.

| Section | Quick Link |
|---------|-----------|
| [Overview](#overview) | [Project Structure](#project-structure) |
| [Quick Start](#quick-start) | [Dataset](#dataset) |
| [Training](#training) | [Evaluation](#evaluation) |
| [Fine‑tuning](#fine-tuning) | [Results](#results) |
| [Acknowledgements](#acknowledgements) | [License](#license) |

---

## Overview
CLIP (Contrastive Language–Image Pre‑training) learns to align vision and language in a shared embedding space.  
We leverage this property to **classify emotions** in two ways:

1. **Zero‑shot**: prompt‑engineer emotion labels (e.g. *“a portrait of a **happy** person”*) and rank them by cosine similarity to the image embedding.
2. **Fine‑tune**: add a lightweight linear classifier on top of CLIP’s frozen image encoder **or** unfreeze the last *N* transformer layers for full fine‑tuning.

All major training & evaluation utilities are implemented in pure PyTorch — no lightning abstractions, so every line is transparent.

---
