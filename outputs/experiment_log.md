# Experiment Log - StemGen Training

## Config
- Model: 3M params (256 dim, 4 layers, 8 heads, dropout 0.1)
- Epochs: 400, Patience: 50, LR: 1e-3
- Baseline (1cb, 4s, 315 tracks): train loss 1.93, train acc 43.6%, val acc 28.0%

---

## E1: 2 codebooks, 10s, 200 tracks
- Train: 1018, Val: 179 | Loss: 4.39, Acc: 41.1% | Val Loss: 6.07, Val Acc: 29.4%
- Early stop: epoch 226, best val at 176
- **Val Acc improved vs 1cb/315t (28.0% → 29.4%) — expanding data further**

## E2: 2 codebooks, 10s, 550 tracks, 400 epochs
- Train: 2773, Val: 489 | Loss: 3.79, Acc: 50.0% | Val Loss: 3.65, **Val Acc: 51.8%**
- Ran FULL 400 epochs (no early stop, best at epoch 353)
- **Val acc jumped from 29.4% → 51.8% with more data (2.5x clips)**
- Model still improving at epoch 400 → increase epochs

## E3: 4 codebooks, 10s, 150 tracks, 400 epochs
- Train: 757, Val: 133 | Loss: 12.15, Acc: 25.6% | Val Loss: 16.03, **Val Acc: 16.8%**
- Early stop: epoch 162 (best val loss at 112, best val acc at 161)
- **4cb/150t FAILED: val acc 16.8% << 1cb baseline 28% — 数据量不足**
- 4 codebook 需要更多数据才能泛化

## E4: 3 codebooks, 10s, 550 tracks, 400 epochs
- **+ activity head (BCE, lambda=0.2) + causal-biased decoding (weight=0.1)**
- Debug guide Priority 1-4 implemented

---

## E5: Audio-token faithful small-scale cleanup (planned / code prepared)
- Goal: return focus to the StemGen audio-token route and make the small reproduction internally aligned.
- Default config updated to: 2 codebooks, 10s clips, 512 dim, 8 layers, 8 heads, variable mask ratio 0.50-1.00.
- Code changes prepared:
  - codec/model/trainer/generator now share the same `num_codebooks`
  - decoder no longer pads missing RVQ levels with token 0
  - trainer computes multi-codebook masked CE instead of only codebook 0
  - generation supports per-codebook decoding steps, top-k, and greedy decoding
  - added `scripts/diagnose_audio_token.py` for codec reconstruction + partial-mask reconstruction
- First validation gate:
  - run codec reconstruction
  - run partial reconstruction at mask 15/30/50/75/100%
  - only then judge full-mask generation quality
