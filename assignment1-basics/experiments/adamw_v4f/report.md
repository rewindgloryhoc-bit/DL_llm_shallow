# Experiment: adamw_v4f

## Brief summary

final v4 training

## Timing

- **Training start time:** 2026-10-15 13:03:08
- **Training end time:** 2026-10-15 13:15:17
- **Total training time:** 0h 12m 9.1s (729.1 s)
- **Steps completed:** 492 (step 4508 to 4999)
- **Status:** interrupted (Ctrl-C) before reaching the final step

### Per-step time

| statistic | seconds |
| --- | --- |
| mean (excluding first step) | 1.2716 |
| median | 1.2680 |
| min | 1.2503 |
| max | 2.2846 |
| first step (includes compile) | 2.6366 |

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

- **Initial loss:** 1.8221
- **Final loss:** 1.5670
- **Mean of last 100 steps:** 1.6015
- **Min loss:** 1.4943

![loss curve](loss_curve.png)
