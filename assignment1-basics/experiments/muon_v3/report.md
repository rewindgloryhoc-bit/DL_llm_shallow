# Experiment: muon_v3

## Timing

- **Training start time:** 2026-10-12 20:36:51
- **Training end time:** 2026-10-12 23:06:06
- **Total training time:** 2h 29m 15.9s (8955.9 s)
- **Steps completed:** 5000 (step 0 to 4999)

### Per-step time

| statistic | seconds |
| --- | --- |
| mean (excluding first step) | 1.3263 |
| median | 1.3146 |
| min | 1.2656 |
| max | 3.8957 |
| first step (includes compile) | 2.5455 |

## Hyperparameters

| name | value |
| --- | --- |
| lr_min | 0.002 |
| lr_max | 0.02 |
| Tw (warmup steps) | 250 |
| Tc (cosine cycle steps) | 5000 |
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

- **Initial loss:** 76.4795
- **Final loss:** 2.8863
- **Mean of last 100 steps:** 2.8247
- **Min loss:** 2.6619

![loss curve](loss_curve.png)
