# ML Commentary
Notes and summaries of machine learning papers.

## Exploring the Limits of Large Scale Pre-training (https://arxiv.org/abs/2110.02095)

This is an amazing paper which trains/analyses more than 4800 models to investigate vision pre-training and transfer learning, so I will comment on all sections of the paper.

The paper starts off by describing a recent, but common view that scaling up the model size and the amount of pre-training data improves performance on the pre-training/upstream (US) task, and downstream (DS) tasks. Crucially, this view holds that these improvements don't saturate as one scales up, i.e. that better performance can always be achieved with more data and larger models. If this were true, it would imply that investing large amounts of resources in pre-training large models would be worth it because we can simply finetune this model on any DS task later on - as opposed to training models from scratch for every individual task.

My take-away from this paper is that this view is partially true.

The first observation (Fig 1) is that most few-shot DS tasks see substantial accuracy saturation, even as US accuracy continuously improves with more training resources. Put differently, for most few-shot DS tasks, you cannot simply invest more pre-training resources in order to increase your DS accuracy. By "few-shot" I mean both 1-shot and 25-shot, which freezes the model and trains a linear classifier head. Further, when I refer to US tasks this means supervised pre-training on JFT (Google's private 300M image dataset), and ImageNet21k (the publicly available 14M image dataset).

The main contribution of this paper is that the relationship between the DS task and US task largely determines DS saturation characteristics. For instance, the closer the DS and US tasks are to each other, the more DS accuracy will improve with increasing US accuracy. In fact, some DS tasks, like CIFAR100 do not seem to saturate (with US tasks JFT and ImageNet21k). As far as I can tell, Fig 2 also shows that for a given US accuracy, the more diverse the US task is, the better the DS accuracy will be - assuming the JFT is more diverse than ImageNet21k. For some DS tasks this difference is negligible, in others it is enormous, like cars (top right in Fig 2). But I cannot tell if more compute was used to reach a given US accuracy for JFT (compared with ImageNet21k), which would muddy this claim. The two US tasks also have a different number of classes, which would effect accuracy (more classes would presumably be more difficult) - although this is such an obvious point that I'm sure the authors took it into account. 

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
1. Classic ImageNet-1k transfer learning always outperforms training a model from scratch.
2. For most target tasks there exists a source task which brings further benefits on top of ImageNet-1k pre-training.
3. The image domain strongly affects transfer gains.
4. For positive transfer, the source image domain should include the target domain.
5. Multi-source models yield good transfer, but are outperformed by the largest within-domain source.
6. Transfer across task types can bring positive transfer gains.
7. Transfer within-task-type and within-domain yields very positive effects.
8. Transfer naturally flows from larger to smaller datasets.
9. Transfer learning effects are larger for small target training sets.
10. The source domain including the target is more important than the number of source samples.

## SustainBench: Benchmarks for Monitoring the Sustainable Development Goals with Machine Learning (https://arxiv.org/abs/2111.04724)

This paper introduces benchmarks for remotely monitoring Sustainable Development Goals (SDGs). Most (11 of 15) of the datasets are released for the first time, and the majority use satellite imagery. Below is a table outlining key features of their data.

https://github.com/sustainlab-group/sustainbench/

| Name  | Inputs |
| ------------- | ------------- |
| Task 1A: Predicting poverty over space | 255x255x8 (7 from Landsat and 1 from either DMSP or VIIRS satellites) plus street-views |
| Task 1B: Predicting change in poverty over time | 255x255x8 (7 from Landsat, and 1 from either DMSP or VIIRS satellites) |
| Task 2A: Cropland mapping  | 50x50x7 (from Landsat) |
| Task 2B1: Crop type mapping, in Ghana in South Sudan | 64x64x17 (3 from Sentinel-1, 10 from Sentinel-2, and 4 from PlanetScope) |
| Task 2B2: Crop type mapping, in Kenya | 64x64x10 (from Sentinel-2) to be confirmed... |
| Task 2C: Crop yield prediction | 32x32x9 (from MODIS) |
| Task 2D: Field delineation | 224x224x3 (from Sentinel-2) |
| Task 3A: Child mortality rate | 255x255x8 (7 from Landsat and 1 from either DMSP or VIIRS satellites) plus street-views |
| Task 3B: Women BMI | 255x255x8 (7 from Landsat and 1 from either DMSP or VIIRS satellites) plus street-views |
| Task 4A: Women educational attainment | 255x255x8 (7 from Landsat and 1 from either DMSP or VIIRS satellites) plus street-views |
| Task 6A: Clean water | 255x255x8 (7 from Landsat and 1 from either DMSP or VIIRS satellites) plus street-views |
| Task 6B: Sanitation | 255x255x8 (7 from Landsat and 1 from either DMSP or VIIRS satellites) plus street-views |
| Task 13A: Brick kiln classification | 64x64x13 (from Sentinel-2) |
| Task 15A: Feature learning for land cover classification | 100x100x4 (from aerial imagery) |
| Task 15B: Out-of-domain land cover classification | 46x8(?) (7 from MODIS and 1 from NDVI) |

## How to train your ViT? Data, Augmentation, and Regularization in Vision Transformers (https://arxiv.org/abs/2106.10270)

This is another massive study which investigates the roles of regularization, data augmentation, training data size, and compute budget for vision transformers (ViT). The two main pre-training datasets used are ImageNet-1k (IN1K) and ImageNet-21k (IN21K). IN1K contains 1.3M images with 1k classes, and IN21K contains 14M images with 21k classes. Further, they stick with 4 ViT architectures ViT-Ti (6M), ViT-S (22M), ViT-B (86M) and ViT-L (307M parameters); typically with patch sizes of 16x16 pixels, but add on a 32x32 variant for ViT-S and ViT-B. They also do not include a hidden layer in the head.

For regularization they experiment with dropout and stochastic depth. For data augmentation they try RandAugment and Mixup. Section 3.3 contains their full hyper-parameter sweep. They additionally set the batch size to 4096, use gradient clipping, the Adam optimizer, and use a cosine LR schedule with a linear warmup for the first 10k steps. For fine-tuning they use SGD with momentum 0.9 and batch size 512.

Now to the findings...

1. A proper AugReg strategy can yield performance gains roughly equal to increasing dataset size by 10x.
2. Always start with a pre-trained model.
3. For a fixed compute budget use more data! For instance, IN1K for 300 epochs is signifcantly worse than IN21K for 30 epochs, specially using the ViT-L model.
4. AugReg is tricky since many settings perform worse than no AugReg at all. Generally, augmentation helps more consistently than regularization. If we focus on ViT-L, it appears that regularizing helps only when no augmentation is used - this regime should be easier to setup via dropout(0.1). Looking at Figure 4, it also seems that augmentation is even more crucial when training under the 300 epoch setting rather than the 30 epoch setting. 
5. When fine-tuning, selecting the pre-trained model with the best upstream accuracy is the most cost-effective approach. But with lots of resources it could help to fine-tune on many pre-trained models and select the best via its downstream accuracy. 
6. Looking at Figure 6, it appears that larger models (i.e. ViT-L) benefit more from dataset size than smaller models (with a fixed budget).

## An Empirical Study of Training Self-Supervised Vision Transformers (https://arxiv.org/abs/2104.02057)

The main contribution of this paper is to show the instability of SSL of ViT models. The authors identify a gradient spike that occurs in earlier layers first, and they propose a simple fix; freezing the patch encoder (i.e. using a random patch encoder). After this main result they perform a bunch of ablations to investigate other aspects of SSL of ViT models.

The following observations were found under a MoCo v3 SSL setup, unless stated otherwise. Sine-cosine absolute position embeddings performed similarly to learned, while no position encoding only dropped linear accuracy from 76.1 to 74.9. Without a CLS token, you'll need to pool the final representations (for image classification), but using a LayerNorm (LN) on these final representations will significantly hurt the pooled representation. ViT models do not use BatchNorm (BN), but including BN in the MLP head (used in many SSL setups) helps linear probing (74.4 -> 76.5). An extra MLP head boosts accuracy (75.5 -> 76.5). Model size matters! ViT-B/16 @ 300-epochs (76.5) is better than ViT-S/16 @ 600-epochs (73.4). Swap out LNs for BNs in the transformer architecture (excluding the attention layers) for a consistent %1 boost in accuracy. Figure 8 shows that smaller patches are better than larger patches, i.e. ViT-BN/7 is better than ViT-BN/16 by a few percenatge points given the same input image size (so the smaller patches would translate to a longer input sequence).

The authors end by discussing SSL more broadly. Supervised pre-training saturates as model size scales (can even get worse with size), whereas SSL pre-training saturates less. They hint at designing more difficult SSL tasks, foreshadowing Kaiming He's (last author on this paper) masked auto-encoder paper, that tries to limit saturation at scale. 

## Discrete Vision Transformers (https://arxiv.org/abs/2111.10493) and (https://arxiv.org/abs/2111.12710)

Two papers exploring discrete ViTs were released simultaneously, they will be refered to as Dr ViT and PeCo. 

