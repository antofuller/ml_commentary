# ML Commentary
Notes and summaries of machine learning papers.

## Exploring the Limits of Large Scale Pre-training (https://arxiv.org/abs/2110.02095)

This is an amazing paper which trains/analyses more than 4800 models to investigate vision pre-training and transfer learning, so I will comment on all sections of the paper.

The paper starts off by describing a recent, but common view that scaling up the model size and the amount of pre-training data improves performance on the pre-training/upstream (US) task, and downstream (DS) tasks. Crucially, this view holds that these improvements don't saturate as one scales up, i.e. that better performance can always be achieved with more data and larger models. If this were true, it would imply that investing large amounts of resources in pre-training large models would be worth it because we can simply finetune this model on any DS task later on - as opposed to training models from scratch for every individual task.

My take-away from this paper is that this view is partially true.

The first observation (Fig 1) is that most few-shot DS tasks see substantial accuracy saturation, even as US accuracy continuously improves with more training resources. Put differently, for most few-shot DS tasks, you cannot simply invest more pre-training resources in order to increase your DS accuracy. By "few-shot" I mean both 1-shot and 25-shot, which freezes the model and trains a linear classifier head. Further, when I refer to US tasks this means supervised pre-training on JFT (Google's private 300M image dataset), and ImageNet21k (the publicly available 14M image dataset).

The main contribution of this paper is that, the relationship between the DS task and US task largely determines DS saturation characteristics. For instance, the closer the DS and US tasks are to each other, the more DS accuracy will improve with increasing US accuracy. In fact, some DS tasks, like CIFAR100 do not seem to saturate (with US tasks JFT and ImageNet21k). As far as I can tell, Fig 2 also shows that for a given US accuracy, the more diverse the US task is, the better the DS accuracy will be - assuming the JFT is more diverse than ImageNet21k. For some DS tasks, this difference is negligible, in others it is enormous, like cars (top right in Fig 2). But I cannot tell if more compute was used to reach a given US accuracy for JFT (compared with ImageNet21k), which would muddy this claim. The two US tasks also have a different number of classes, which would effect accuracy (more classes would presumably be more difficult) - although this is such an obvious point that I'm sure the authors took it into account. 

Next the paper makes two more claims. Between data size, number of epochs, and model size, the latter contributes relatively more on both US and DS accuracy. Secondly, "conditioned on US accuracy, none of these three parameters provides extra information on DS accuracy." From this, it seems like "all you need" to reasonably predict DS accuracy is US accuracy and the similarity of US and DS tasks.

One key point about this paper is that it analyses the best performing models (with varying hyper-parameters), rather than the average model. So as far as I understand, these conclusions would only hold if you can afford extensive hyper-parameter searches. Take a look at Fig 1, here, its easy to see that taking the average model would result in less steep saturation curves - although they would still saturate.

The next finding, Fig 7, is very interesting. The authors first probe DS accuracy at every layer of the models and find that for some tasks, a linear probe on lower representations outperforms probing the final representations. This by itself is not a big surprise, but they also find a striking similarity of US vs DS accuracy saturation and the layer-wise saturation of DS tasks. That is, DS tasks that saturate quickly with respect to US accuracy, also don't benefit as much when deeper layers are used for classification. The authors write "performance saturation on DS happens when the pre-trained network lacks the fine-grained features required to perform well on DS. Therefore, one can get similar performance on such DS task when cutting the top layers of the pre-trained model, as seen in Figure 7."

To me, this essentially confirms the rule-of-thumb that earlier layers learn more general features and final layers learn more task-specific features. 

Now, let's investigate head weight decay (WD) and learning rate (LR) during finetuning. Looking at Fig 8, it seems for similar US and DS tasks, large head WD values (during pre-training) seems to benefit the DS task at the expense of US. But for DS and US tasks that are far apart (like uc_merced or col_hist), cranking up WD reduces both US and DS performance. In the conclusions of this section they state: "In other words, there are cases where increasing or decreasing US head WD results in improved performance for a DS task and degraded performance for another DS task. Therefore, one cannot simply save a checkpoint of a model pre-trained on an upstream task and use it for all downstream tasks." However, this conclusion is strongest for fewer DS shots. So if you expect to have more finetuning data (say more than 20 shots), this result shouldn't be as meaningful for you. Lowering head LR (during pre-training) is also said to have a similar effect as increasing head WD.

By comparing the norms of layers vs the head's norm (Fig 9) the authors claim that higher pre-training head WD forces the upper layers to essentially take on more and more of the US task. So a higher head WD pushes the US task further into the model's layers. If I am understanding this correctly, it would make sense of the previous finding, since the US task would be pushed into the backbone model, at higher head WD values. Thus, if the DS task is similar to the US task, the model could make use of those upper layers. 

Fig 11 plots the best WD for each task, and the US/DS correlations. It shows that closer US and DS tasks have larger optimal WD (for a given DS task). There are 2 exceptions, the resisc45 and caltech datasets, which both have very high US/DS correlations but an optimal WD of 0. Resisc45 is a remote sensing dataset, which at least to me would be closer to the eurosat dataset than JFT. Eurosat also has an optimal WD around 0, but a low US/DS correlation. So I'm not sure why resisc45 has such a high correlation with JFT. The caltech dataset seems like it would be more similar to JFT since its task is to classify fairly common objects - so caltech's near-zero optimal WD is strange. The reasons for these 2 outliers is not mentioned in the paper. Eurosat's images are small (64x64) and have 10 classes, UC Merced (another remote sensing dataset) has images of size 256x256 with 21 classes. Resisc45 is also 256x256 but with 45 classes. This remains a mystery...

In section 5 the authors comment about how generalizable these conclusions are. For a full-blown transfer learning scenario, with 1000 samples per class, they claim similar general trends as the few-shot setup. More finetuning results are available in the appendix, section E. There we can see massive gains in CIFAR100 accuracy when pre-training with a high head WD, and massive losses in accuracy for clevr (visual QnA) when increasing head WD. My takeaway is that if you know your DS task is very similar to your US task, then using a higher head WD during pre-training may be worth it. If not, it may be wiser to train more general final layers via a very low head WD (or a larger head, not discussed in this paper) - or a higher head LR, which is cheaper than a larger head. 

The authors note "that the effect of (model) architecture is only observed through the US performance." Basically, if your model architecture can improve US performance, then it may help with DS. But for a given US accuracy, model archicture likely doesn't play much of a role in DS accuracy. 

The paper concludes with: "We demonstrate the role of hyper-parameters and emphasize that one cannot hope to find one pre-trained checkpoint that performs well on all possible downstream tasks. We assert that we should refrain from focusing on the performance of only one downstream task, which usually ends up being close to the upstream task. Instead, we should make design choices that improve performance on a breadth of downstream tasks. Moreover, scaling has both monetary and environmental costs [Patterson et al., 2021]. We argue that, when investing in terms of scaling in terms of data, model parameters and compute, we should think of an additional axis which is data diversity."

## Factors of Influence for Transfer Learning across Diverse Appearance Domains and Task Types (https://arxiv.org/abs/2103.13318)

This is another super interesting paper on transfer learning in vision, and is referenced by the previous paper (Exploring the Limits...). The authors perform over 1200 transfer learning experiments on many vision domains, dataset sizes, and tasks. Their experimental setup is ImageNet pre-training -> source task -> target task.

They create 7 domain groups, from their datasets: consumer photos, driving, indoor, aerial, underwater, close-ups, and synthetic. Along with 4 tasks: semantic segmentation, object detection, keypoint detection, and depth estimation - all of which involve spatial localization. With 40 total datasets, each dataset is used as both a source and a target.

Based off their experiments, the authors make 10 claims: (direct quotes)
1. Classic ILSVRC’12 transfer learning always outperforms training a model from scratch.
2. For most target tasks there exists a source task which brings further benefits on top of ILSVCR’12 pre-training.
3. The image domain strongly affects transfer gains.
4. For positive transfer, the source image domain should include the target domain.
5. Multi-source models yield good transfer, but are outperformed by the largest within-domain source.
6. Transfer across task types can bring positive transfer gains.
7. Transfer within-task-type and within-domain yields very positive effects.
8. Transfer naturally flows from larger to smaller datasets.
9. Transfer learning effects are larger for small target training sets.
10. The source domain including the target is more important than the number of source samples.


