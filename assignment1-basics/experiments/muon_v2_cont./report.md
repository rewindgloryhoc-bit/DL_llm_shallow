# Experiment: muon_v2_cont.

## Timing

- **Training start time:** 2026-10-12 12:59:50
- **Training end time:** 2026-10-12 13:53:49
- **Total training time:** 0h 53m 58.9s (3238.9 s)
- **Steps completed:** 2438 (step 2562 to 4999)

### Per-step time

| statistic | seconds |
| --- | --- |
| mean (excluding first step) | 1.2931 |
| median | 1.2877 |
| min | 1.2645 |
| max | 2.2537 |
| first step (includes compile) | 2.6076 |

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

- **Initial loss:** 2.4611
- **Final loss:** 1.9307
- **Mean of last 100 steps:** 1.9259
- **Min loss:** 1.8037

![loss curve](loss_curve.png)
