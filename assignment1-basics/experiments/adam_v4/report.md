# Experiment: adam_v4

## Brief summary

decay steps revised, shorter steps

## Timing

- **Training start time:** 2026-10-15 08:01:14
- **Training end time:** 2026-10-15 08:31:37
- **Total training time:** 0h 30m 23.7s (1823.7 s)
- **Steps completed:** 1413 (step 0 to 1412)
- **Status:** interrupted (Ctrl-C) before reaching the final step

### Per-step time

| statistic | seconds |
| --- | --- |
| mean (excluding first step) | 1.2764 |
| median | 1.2726 |
| min | 1.2546 |
| max | 2.2370 |
| first step (includes compile) | 2.4921 |

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

- **Initial loss:** 76.3900
- **Final loss:** 3.1607
- **Mean of last 100 steps:** 3.2635
- **Min loss:** 3.1268

![loss curve](loss_curve.png)
