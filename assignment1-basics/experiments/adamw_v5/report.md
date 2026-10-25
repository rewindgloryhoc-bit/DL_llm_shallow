# Experiment: adamw_v5

## Brief summary

fine tune adamw tail performance

## Timing

- **Training start time:** 2026-10-15 13:16:46
- **Training end time:** 2026-10-15 15:03:41
- **Total training time:** 1h 46m 55.4s (6415.4 s)
- **Steps completed:** 5000 (step 0 to 4999)

### Per-step time

| statistic | seconds |
| --- | --- |
| mean (excluding first step) | 1.2697 |
| median | 1.2697 |
| min | 1.2515 |
| max | 2.2119 |
| first step (includes compile) | 2.5185 |

## Hyperparameters

| name | value |
| --- | --- |
| optimizer | AdamW only |
| lr_min | 0.0008 |
| lr_max | 0.008 |
| Tw (warmup steps) | 500 |
| Tc (cosine cycle steps) | 5000 |
| decay_steps | 1000 |
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

- **Initial loss:** 76.7740
- **Final loss:** 1.5430
- **Mean of last 100 steps:** 1.5992
- **Min loss:** 1.5004

![loss curve](loss_curve.png)
