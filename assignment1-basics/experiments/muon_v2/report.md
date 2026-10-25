# Experiment: muon_v2

## Timing

- **Training start time:** 2026-10-12 11:20:15
- **Training end time:** 2026-10-12 12:16:49
- **Total training time:** 0h 56m 34.0s (3394.0 s)
- **Steps completed:** 2562 (step 0 to 2561)
- **Status:** interrupted (Ctrl-C) before reaching the final step

### Per-step time

| statistic | seconds |
| --- | --- |
| mean (excluding first step) | 1.2895 |
| median | 1.2830 |
| min | 1.2632 |
| max | 2.2538 |
| first step (includes compile) | 2.6126 |

## Hyperparameters

| name | value |
| --- | --- |
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

- **Initial loss:** 76.9989
- **Final loss:** 2.4407
- **Mean of last 100 steps:** 2.4708
- **Min loss:** 2.3232

![loss curve](loss_curve.png)
