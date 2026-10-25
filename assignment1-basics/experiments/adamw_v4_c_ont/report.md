# Experiment: adamw_v4_c_ont

## Brief summary

none, incorrect lr schedule, ceased early.

## Timing

- **Training start time:** 2026-10-15 09:47:27
- **Training end time:** 2026-10-15 10:53:32
- **Total training time:** 1h 6m 5.5s (3965.5 s)
- **Steps completed:** 3095 (step 1413 to 4507)
- **Status:** interrupted (Ctrl-C) before reaching the final step

### Per-step time

| statistic | seconds |
| --- | --- |
| mean (excluding first step) | 1.2676 |
| median | 1.2638 |
| min | 1.2381 |
| max | 2.1361 |
| first step (includes compile) | 2.4978 |

## Hyperparameters

| name | value |
| --- | --- |
| optimizer | AdamW only |
| lr_min | 0.0008 |
| lr_max | 0.008 |
| Tw (warmup steps) | 500 |
| Tc (cosine cycle steps) | 5000 |
| decay_steps | 800 |
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

- **Initial loss:** 3.3147
- **Final loss:** 1.7480
- **Mean of last 100 steps:** 1.8283
- **Min loss:** 1.7016

![loss curve](loss_curve.png)
