# ML Commentary
Notes and summaries of machine learning papers.

## Exploring the Limits of Large Scale Pre-training (https://arxiv.org/abs/2110.02095)

This is an amazing paper which trains more than 4800 models to investigate vision pre-training and transfer learning, so I will comment on all sections of the paper.

The paper starts off by describing a recent, but common view that scaling up the model size and the amount of pre-training data improves performance on the pre-training/upstream (US) task, and downstream (DS) task. Crucially, this view holds that these improvements don't saturate as one scales up, i.e. that better performance can always be achieved with more data and larger models. If this were true, it would imply that investing large amounts of resources in pre-training large models would be worth it because we can simply finetune this model on any DS task later on - as opposed to training models from scratch for every individual task.

My take-away from this paper is that this claim is partially true.

