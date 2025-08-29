import torch as th
from einops import einsum
from jaxtyping import Float
import matplotlib.pyplot as plt

# Define different theta and d_k values
thetas = [100, 1000]
d_ks = [128, 64]

max_seq_len = 1000
# all the positions within max sequence len
i: Float[th.Tensor, 'seq_len'] = th.arange(start=0, end=max_seq_len, dtype=th.float32)

# Create a plot of theta_k values
plt.figure(figsize=(12, 6))

for theta in thetas:
    for d_k in d_ks:
        # all the odd values smaller than d_k
        theta_k: Float[th.Tensor, 'half_d'] = th.pow(theta, -(th.arange(1, d_k, 2) / d_k))
        plt.plot(theta_k.numpy(), label=f'θ={theta}, d_k={d_k}')

plt.title('Theta_k values for different θ and d_k')
plt.xlabel('Index')
plt.yscale('log')
plt.ylabel('Value')
plt.grid(True)
plt.legend()
plt.show()


# create angle for each seq_pos, vector_pos combination
theta_ik = einsum(theta_k, i, 'half_d, seq_len -> half_d seq_len')
