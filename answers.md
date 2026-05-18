A few sentences discussing of your findings on batch sizes and their impacts on
training.

Going to low in learning rate seems to have a large negative impact than going
too high. At least for a fixed number of steps.

I'm able to get lower loss with batch size 64 than with 32. 

The optimal learning rate is the same for 16, 32, 64. For smaller batch size,
the punishmen is smaller on a smaller learning rate though. 

For the smaller batches, 8 and 4, convergence is poor. I guess I take a fixed
number of steps, not a fixed number of tokens. But the grad norms are way
noisier too.

With bs=128 we get all the way down to validation loss 0.3 (instead of 0.9 for
batch size 64). Intersting -- 2x compute -> 3x loss reduction. 


