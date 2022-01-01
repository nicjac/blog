---
title: "Good practices for neural network training: identify, save, and document best models"
date: 2021-12-30T10:00:08Z
draft: false
tags: ["Deep Learning", "AI", "Weights&Biases", "fastai", "good practices"]
categories: ["Applied AI"]
---

An introduction to the good practice of *best model* saving when training a neural network, and a practical implementation using fastai and Weights & Biases

<!--more-->

## Preamble

In this post, we are going to:
- Introduce the concept of *best model*
- Discuss how the *best model* can be identified, saved, and documented during training
- Explore how to leverage fastai and Weights & Biases to do all of this (almost) automatically and effortlessly

## What is a _best model_ and why should I care?

Model training can be seen as the generation of subsequent versions of a model  --  after each batch, the model weights are adjusted, and as a result, a new version of the model is created. Each new version will have varying levels of performance (as evaluated against a validation set).

If everything goes well, training and validation loss will decrease with the number of training epochs. However, the best performing version of a model (here abbreviated as *best model*) is rarely the one obtained at the end of the training process.

{{< figure align=center src="/images/training/overfitting-best-model.png" caption="A somewhat typical example of overfitting training curves." >}}

Take a typical overfitting case  --  at first, both training and validation losses decrease as training progresses. At some point, the validation loss might start increasing, even though the training loss continues to decrease; from this point on, subsequent model versions produced during the training process are overfitting the training data. These model versions are less likely to generalize well to unseen data. In this case, the *best model* would be the one obtained at the point where the validation loss started to diverge.

Overfitting is a convenient example, but similar observations would also apply to other dynamics of model training, such as the presence of local maxima or minima.

## Identifying and saving the *best model* during training
The naive approach to the problem would be to save our model after every epoch during training and to retrospectively select the optimal version based on the training curves. This approach is associated with a couple of notable drawbacks:
- **Storage space**: large models tend to occupy a significant amount of storage when saved, with file sizes reaching 100s of MBs to GBs. Multiply this by the number of epochs, and you end up with a pretty sizeable amount of storage dedicated to saving all versions of a model. This can quickly become problematic, especially when training models remotely. 
- **Computational impact**: saving a model after every epoch will impact overall training time  --  serialization / export of some models can be computationally intensive and slow.

{{< figure align=center src="/images/training/best-model-naive-and-best.png" caption="Naive approach where all model versions are persisted versus the best model only approach, where only the most recent best performing model is persisted" >}}

As an alternative to this brute force approach where all model versions are saved during training, one can be more selective. We know from the above that our *best model* is likely one associated with a low validation loss. We can thus formulate a criterion for the selection of our *best model*: it must have a lower validation loss than the previous candidate. As pseudo-code:

```none
if current validation loss lower than candidate validation loss:
    save model to disk overwriting previous candidate
    set candidate validation loss to current validation loss
```
The key advantages of this approach is that a) a new model is exported only when the validation loss is improved over the previous best candidate, and b) we only ever have one model version persisted to storage at any given time. As thus, we successfully addressed the two drawbacks of the naive approach. 

Maybe more importantly, only saving the *best model* also encourages good practices by requiring the performance evaluation methodology to be decided before training starts, and it removes the temptation to retroactively evaluate other versions of the model on a separate test dataset.

## A note on validation loss, alternative metrics, and model documentation
Up to this point, we used validation loss as our target metric to identify the *best model* during training. Why validation loss you might ask? The fact that it is almost always computed during training made it a convenient example to illustrate the concepts discussed in this article. 

However, validation loss might not be that relevant to your particular use-case or domain, and any other metric can be used instead. For classification tasks, accuracy could be a good choice. Similarly, you can choose the target metric to ensure that the *best model* is also one that will generalize well to unseen data, for example by using [Matthews Correlation Coefficient](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.matthews_corrcoef.html) when dealing with severely imbalanced datasets.

Whatever target metric you decide to use, it is also important to document other aspects of this particular version of the model. Typically, this would include all performance metrics tracked during training. Persisting this information together with the actual model artifact can be very useful later on, for example to rank models obtained from a hyperparameter search or to carry out integration testing when deploying in production (more on this in a future article!).

## Effortlessly save the *best model* during training with fastai

Implementation of best model saving requires alteration of the training loop in order to monitor the target metric and to trigger model saving when an improvement is detected. Many modern frameworks come with this capability built-in. Here, we will be focusing on a fastai implementation, but similar capabilities are likely available for your library of choice. You can still follow along to get an idea how this can be implemented in practice.

If you are unaware of what fastai is, its [official description is](https://github.com/fastai/fastai):

> fastai simplifies training fast and accurate neural nets using modern best practices

The fastai training loop can be modified and extended using [callback methods](https://docs.fast.ai/callback.core.html) that are called at specific stages of training, for example after an epoch is completed, or at the end of training. Conveniently, the [SaveModelCallback](https://docs.fast.ai/callback.tracker.html#SaveModelCallback) happens to do (almost) exactly what we need.

Using the callback couldn't be easier:

```python
learner.fit_one_cycle(... ,cbs=[..., SaveModelCallback(monitor='valid_loss')])
```

where `learner` is a standard fastai [Learner object](https://docs.fast.ai/learner.html#Learner). By default, the callback will track the validation loss to determine when to save a new *best model*. Use the `monitor` argument to set it to any other metric tracked by your `learner` object. Following each epoch during training, the current value for the target metric is compared to the previous best value - if it is an improvement, the model is persisted in the `models` directory (and overwriting the previous best candidate, if present). 

Behind the scene, the callback tries to figure whether an improvement is a smaller value (if the target metric contains `loss` or `error`) or a larger value (everything else). This behavior can be overridden using the `comp` argument. The model is persisted using fastai's [`save_model`](https://docs.fast.ai/learner.html#Learner.save) function, which is a wrapper for Pytorch's native [`torch.save`](https://pytorch.org/docs/stable/generated/torch.save.html).

The reason why the built-in callback is not *exactly* what we need is that it will only log the target metric used to identify the *best model*, and nothing else. It will not log other metrics (for example accuracy, if the *best model* is determined based on validation loss). This might be fine, but given that our *best model* might end up being used as part of a product somewhere, it would be a good idea to characterize it as much as possible. I put together a custom version of the `SaveModelCallback` that will log all metrics tracked by fastai during training. The code for can be found [here](https://gist.github.com/nicjac/b363d2454ea253570a54e5e178e7666a).

This custom version of the callback can be used as a drop-in replacement. All it really does is to internally keep track of a dictionary of metrics (`last_saved_metadata`) associated with the *best model*. How to make use of this? All is to be revealed in the next section!

## Automatically document the *best model* with Weights & Biases

Saving the *best model* locally is a good start, but it can quickly become unwieldy if you work remotely, or carry out large number of experiments. So how to keep track of the models created, and of their associated metrics? This is where [Weights & Biases](https://wandb.ai/site) comes in. W&B is one of those tools that make you wonder how you have ever been able to properly function without them. While officially described as "The developer-first MLOps platform", I prefer to refer to it as the swiss army knife of MLOps. 

W&B is very useful to track and compare experiments. However, for the purpose of this article, we are mainly interested in its almost universal versioning capabilities. In the W&B ecosystem, [artifacts](https://wandb.ai/site/artifacts) are components that can be versioned, possibly together with their lineage. Models can be versioned as artifacts.

Conveniently, fastai has a built-in callback to integrate with W&B, aptly named [`WandbCallback`](https://docs.fast.ai/callback.wandb.html#WandbCallback). To use it, one need to initialize a W&B run, and to add the callback to the learner object like so:

```python
# Import W&B package
import wandb

# Initialize W&B run (can potentially set project name, run name, etc...)
wandb.init()

# Add Callback to learner to track training metrics and log best models
learn = learner(..., cbs=WandbCallback())
```

The main purpose of the callback is to log useful telemetry regarding the training process to your W&B account, including environment information and metrics. The magic happens when it is used in combination with the `SaveModelCallback` -- at the end of the training process, the best performing model will be automatically logged as an artifact of the W&B run.

There is one major issue with the default `WandbCallback`: the metadata associated with the model is recorded at the end of the run and not at the epoch when the *best model* was saved. In other words, the metadata **does not** correspond to the saved model at all, and can be misleading (for example when the tracked metric diverged towards the end of training due to overfitting).

This is where the custom `SaveModelCallback` that was discussed in the previous section comes in. It will save all the information needed to associate the model with its *actual* metadata. To take advantage of this, it is also necessary to use a custom version of `WandbCallback`, which can be found [here](https://gist.github.com/nicjac/9efb56cccd57f9c84910f02ccabf6fac).

The changes made in the custom callback are highlighted here:

{{< highlight python "linenos=table,hl_lines=6-10 ,linenostart=101" >}}
def after_fit(self):
    if self.log_model:
        if self.save_model.last_saved_path is None:
            print('WandbCallback could not retrieve a model to upload')
        else:
            log_model(self.save_model.last_saved_path, metadata=self.save_model.last_saved_metadata)
            
            for metadata_key in self.save_model.last_saved_metadata:
                wandb.run.summary[f'best_{metadata_key}'] = self.save_model.last_saved_metadata[metadata_key]
{{< / highlight >}}

As a result, the following will automatically happen:
- The model logged to the W&B run is associated with metadata containing the correct metric values
- The values for all metrics for the *best model* are added to the run summary, with the prefix `best_`. This allows runs to be sorted and compared based on the performance of their respective *best model*

{{< figure align=center src="/images/training/wandb-model-combined.png" caption="Best model logged in Weights and Biases. (Left) Model metadata including key metrics; (Right) Models in a W&B project sorted by the `best_matthews_corrcoef` metadata associated with their respective *best models*" >}}

## Wrapping up

So, what have we learned in this article?
- Only saving the *best model* during training is efficient and encourages good practices
- The metadata, including key metrics associated with the *best model*, is almost as important as the model artifact itself
- Using fastai and Weights & Biases, saving and documenting the *best model* can be done automatically for you. Two custom callback functions were described to make this process even better ([SaveModelCallback](https://gist.github.com/nicjac/b363d2454ea253570a54e5e178e7666a) and [WandbCallback](https://gist.github.com/nicjac/9efb56cccd57f9c84910f02ccabf6fac)).