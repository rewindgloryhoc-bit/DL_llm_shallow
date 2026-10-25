# Experiment: adam_revisi

## Timing

- **Training start time:** 2026-10-13 19:56:48
- **Training end time:** 2026-10-13 23:06:57
- **Total training time:** 3h 10m 8.7s (11408.7 s)
- **Steps completed:** 5000 (step 0 to 4999)

### Per-step time

| statistic | seconds |
| --- | --- |
| mean (excluding first step) | 1.3534 |
| median | 1.3040 |
| min | 1.2600 |
| max | 6.8772 |
| first step (includes compile) | 2.5258 |

## Hyperparameters

| name | value |
| --- | --- |
| optimizer | Muon (2D hidden) + AdamW (rest) |
| lr_min | 0.0008 |
| lr_max | 0.008 |
| Tw (warmup steps) | 500 |
| Tc (cosine cycle steps) | 5000 |
| weight_decay | 0.1 |
| total steps | 5000 |
| batch_size | 64 |
| max_seq | 256 |
| d_model | 512 |
| num_heads | 16 |
| num_layers | 4 |
| d_ff | 1344 |
| vocab_size | 10000 |
| rope theta | 10000 |
| device | mps |
| dtype | torch.float32 |

## Loss

- **Initial loss:** 76.9108
- **Final loss:** 1.7596
- **Mean of last 100 steps:** 1.7270
- **Min loss:** 1.6201

![loss curve](loss_curve.png)
