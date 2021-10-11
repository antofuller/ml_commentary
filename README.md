# ML Commentary
Notes and summaries of machine learning papers.

## Exploring the Limits of Large Scale Pre-training (https://arxiv.org/abs/2110.02095)

This is an amazing paper which trains/analyses more than 4800 models to investigate vision pre-training and transfer learning, so I will comment on all sections of the paper.

The paper starts off by describing a recent, but common view that scaling up the model size and the amount of pre-training data improves performance on the pre-training/upstream (US) task, and downstream (DS) tasks. Crucially, this view holds that these improvements don't saturate as one scales up, i.e. that better performance can always be achieved with more data and larger models. If this were true, it would imply that investing large amounts of resources in pre-training large models would be worth it because we can simply finetune this model on any DS task later on - as opposed to training models from scratch for every individual task.

My take-away from this paper is that this view is partially true.

The first observation (Fig 1) is that most few-shot DS tasks see substantial accuracy saturation, even as US accuracy continuously improves with more training resources. Put differently, for most few-shot DS tasks, you cannot simply invest more pre-training resources in order to increase your DS accuracy. By "few-shot" I mean both 1-shot and 25-shot, which freezes the model and trains a DS head. 

The main contribution of this paper is that, the relationship between the DS task and US task largely determines DS saturation characteristics. For instance, the closer the DS and US tasks are to each other, the more DS accuracy will improve with increasing US accuracy. In fact, some DS tasks, like CIFAR100 do not seem to saturate (with US tasks JFT and ImageNet21k). As far as I can tell, Fig 2 also shows that for a given US accuracy, the more diverse the US task is, the better the DS accuracy will be - assuming the JFT is more diverse than ImageNet21k. For some DS tasks, this difference is negligible, in others it is enormous, like cars (top right in Fig 2). But I cannot tell if more compute was used to reach a given US accuracy for JFT (compared with ImageNet21k), which would muddy this claim.


