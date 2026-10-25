# Experiment: adam_w_revisit

## Timing

- **Training start time:** 2026-10-14 20:37:54
- **Training end time:** 2026-10-14 22:08:41
- **Total training time:** 1h 30m 46.8s (5446.8 s)
- **Steps completed:** 3455 (step 1545 to 4999)
- **Status:** interrupted (Ctrl-C) before reaching the final step

### Per-step time

| statistic | seconds |
| --- | --- |
| mean (excluding first step) | 1.3274 |
| median | 1.3205 |
| min | 1.2469 |
| max | 2.1200 |
| first step (includes compile) | 2.4664 |

## Hyperparameters

| name | value |
| --- | --- |
| optimizer | AdamW only |
| lr_min | 0.0008 |
| lr_max | 0.008 |
| Tw (warmup steps) | 500 |
| Tc (cosine cycle steps) | 4000 |
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

- **Initial loss:** 3.1500
- **Final loss:** 1.7176
- **Mean of last 100 steps:** 1.6865
- **Min loss:** 1.5718

![loss curve](loss_curve.png)
