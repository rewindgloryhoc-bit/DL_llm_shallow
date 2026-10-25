# Experiment: muon_v5

## Timing

- **Training start time:** 2026-10-13 17:48:27
- **Training end time:** 2026-10-13 19:54:49
- **Total training time:** 2h 6m 21.4s (7581.4 s)
- **Steps completed:** 5000 (step 0 to 4999)
- **Status:** interrupted (Ctrl-C) before reaching the final step

### Per-step time

| statistic | seconds |
| --- | --- |
| mean (excluding first step) | 1.3089 |
| median | 1.3003 |
| min | 1.2603 |
| max | 2.3872 |
| first step (includes compile) | 2.5416 |

## Hyperparameters

| name | value |
| --- | --- |
| lr_min | 0.0002 |
| lr_max | 0.02 |
| Tw (warmup steps) | 300 |
| Tc (cosine cycle steps) | 4000 |
| weight_decay | 0.005 |
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

- **Initial loss:** 77.3706
- **Final loss:** 2.6985
- **Mean of last 100 steps:** 2.7903
- **Min loss:** 2.6202

![loss curve](loss_curve.png)
