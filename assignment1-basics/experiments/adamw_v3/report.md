# Experiment: adamw_v3

## Brief summary

this experiment used the correct cos decay, started decay earlier, hold lr_max until 3800 steps, TW remain unchanged. hopefully loss can settle avg 1.6 around tail region

## Timing

- **Training start time:** 2026-10-14 22:17:45
- **Training end time:** 2026-10-15 00:05:20
- **Total training time:** 1h 47m 34.9s (6454.9 s)
- **Steps completed:** 5000 (step 0 to 4999)

### Per-step time

| statistic | seconds |
| --- | --- |
| mean (excluding first step) | 1.2778 |
| median | 1.2800 |
| min | 1.2516 |
| max | 2.1979 |
| first step (includes compile) | 2.5340 |

## Hyperparameters

| name | value |
| --- | --- |
| optimizer | AdamW only |
| lr_min | 0.0008 |
| lr_max | 0.008 |
| Tw (warmup steps) | 500 |
| Tc (cosine cycle steps) | 5000 |
| decay_steps | 1200 |
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

- **Initial loss:** 76.5436
- **Final loss:** 1.7371
- **Mean of last 100 steps:** 1.8041
- **Min loss:** 1.6923

![loss curve](loss_curve.png)
