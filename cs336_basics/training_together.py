from cs336_basics.optimizer import get_lr_cosine_schedule
import numpy as np
import torch
from cs336_basics import training
from cs336_basics import optimizer
from cs336_basics import module

train_data_fname = "tinystories_train_tokenized.npy"
valid_data_fname = "tinystories_valid_tokenized.npy"
checkpoint_fname = "tinystories_checkpoint.pkl"
batch_size = 10
context_length = 256
device = "mps"
vocab_size = 10_000
iters = 100
d_model = 512
num_layers = 20
num_heads = 16
d_ff = 1344
rope_theta = 10_000
max_learning_rate = 1e-3
min_learning_rate = 1e-6
warmup_iters = 5
cosine_cycle_iters = 80
weight_decay = 0.01
betas = (0.90, 0.99)
eps = 1e-6
validation_cycle = 10
checkpoint_cycle = 100
max_l2_norm = 50

train_data = np.load(train_data_fname, mmap_mode="r")
valid_data = np.load(valid_data_fname, mmap_mode="r")
model = module.TransformerLM(vocab_size, context_length, d_model, num_layers, num_heads, d_ff, rope_theta).to(device)
optim = optimizer.AdamW(model.parameters(), max_learning_rate, weight_decay, betas, eps)

for it in range(iters):
    model.zero_grad()
    inputs, targets = training.get_batch(train_data, batch_size, context_length, device)
    logits = model.forward(inputs)
    loss = module.cross_entropy(logits, targets)
    print(f"training loss: {loss}")
    loss.backward()
    optimizer.gradient_clipping(model.parameters(), max_l2_norm)
    lr = get_lr_cosine_schedule(it, max_learning_rate, min_learning_rate, warmup_iters, cosine_cycle_iters)
    for group in optim.param_groups:
        group["lr"] = lr
    optim.step()
    
    if it > 0 and it % validation_cycle == 0:
        with torch.no_grad():
            valid_inputs, valid_targets = training.get_batch(valid_data, batch_size, context_length, device)
            valid_logits = model.forward(valid_inputs)
            valid_loss = module.cross_entropy(valid_logits, valid_targets)
            print(f"validation loss: {valid_loss}")
        
    if it > 0 and it % checkpoint_cycle == 0:
        training.save_checkpoint(model, optim, it, checkpoint_fname)
    