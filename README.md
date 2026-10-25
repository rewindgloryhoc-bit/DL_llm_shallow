~200 Million Parameter model on 45minute time pre-training
- naive baseline, without tiling and other triton kernels

some specs:
d_model: 1024
num_heads: 16
layers: 12
context_len: 512
batch: 256
batch_accum: None
optimizer: AdamW
FFN: Swiglu
vocab_size: 32000
dataset: OpenWebtext


validation and loss curves are included in experiments/runpod_owt10 as well as reports.
