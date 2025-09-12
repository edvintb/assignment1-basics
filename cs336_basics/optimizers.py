import math
from typing import Optional, Callable

import torch as th


class SGD(th.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate {lr}")
        defaults = {"lr" : lr}
        super().__init__(params, defaults)
    
    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                p.data -= lr * p.grad

        return loss

class AdamW(th.optim.Optimizer):
    def __init__(
        self,
        params,
        lr=1e-3,
        weight_decay=0.01,
        betas=(0.9, 0.999),
        eps=1e-8,
    ):
        if lr < 0:
            raise ValueError(f"Invalid learning rate {lr}")
        defaults = {
            "lr": lr,
            "beta1": betas[0],
            "beta2": betas[1],
            "epsilon": eps,
            "weight_decay": weight_decay,
        }
        super().__init__(params, defaults)
    
    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            beta1 = group["beta1"]
            beta2 = group["beta2"]
            eps = group["epsilon"]
            weight_decay = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]

                # Initialize state if needed
                if len(state) == 0:
                    state['t'] = 0
                    state['m'] = th.zeros_like(p.data)
                    state['v'] = th.zeros_like(p.data)

                # adapt the learning rate to the timestep
                state['t'] += 1
                t = state['t']
                alpha = lr * math.sqrt(1 - math.pow(beta2, t)) / (1 - math.pow(beta1, t))

                # update moments
                grad = p.grad.data

                m = state['m']
                v = state['v']
                m = beta1 * m + (1 - beta1) * grad
                v = beta2 * v + (1 - beta2) * th.pow(grad, 2)

                # update the parameters
                p.data -= alpha * (m / (th.sqrt(v) + eps))

                # Apply weight decay directly to parameters (AdamW style)
                if weight_decay != 0:
                    p.data -= lr * weight_decay * p.data

                # update the state
                state["m"] = m
                state["v"] = v

        return loss


if __name__ == "__main__":
    weights = th.nn.Parameter(5 * th.randn((10, 10)))
    for lr in [1, 10, 100, 1000]:
        opt = SGD([weights], lr=lr)
        print(f"learning rate: {lr}")
        for t in range(100):
            opt.zero_grad() # Reset the gradients for all learnable parameters.
            loss = (weights**2).mean() # Compute a scalar loss value.
            print(f"iter: {t:03d}, loss: {loss}")
            loss.backward() # Run backward pass, which computes gradients.
            opt.step() # Run optimizer step.