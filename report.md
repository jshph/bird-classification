

Briefly remind why classifying birds is challenging, due to fine-grained recognition challenges. Describe how this paper will talk about tradeoffs interspersed with description of approach. Geometry is discussed in terms of preference for logistic regression over SVM due to dataset size, nature, and similarity to original ImageNet model. Geometry is also briefly discussed in terms of similarity to HOG and activations of CNN.

## Approach description and justification

### Intuition of convolutional neural networks for image classification

Convolutional neural networks provide a trainable, end-to-end pipeline for feature extraction from images. For our dataset, this means identifying bird parts that are salient to discrimination between classes. At the very bottom, we feed an RGB image, and the convolutional layers pick up details about the image that are increasingly large as layer goes up. The output is a higher-level embedding that can be used to classify the bird. For classification, we use fully-connected layers, in order to maintain the end-to-end trainability of the network.

In developing our architecture, we begin with a subset of 50 classes so that we can quickly iterate on it based on performance. Once we have settled, we train on the full dataset of 200 classes and see that its performance is comparable.

### VGG-16 over other architectures

We develop our bird classification pipeline with a variation of the VGG-16 neural network architecture, detailed by Simonyan and Zisserman for the ILSVRC-2014 competition [cite]. We choose this architecture over Inception V3, even though that has been shown to classify our dataset, CUB-200-2011 (as in Caltech-UCSD-Birds, 200 classes, released in 2011) with 84.4% accuracy – close to the state-of-the-art (Krause et al., 2016) ==cite== –  and even though there is a tutorial about this pipeline. We also choose this architecture over the very deep ResNet architecture, with which a guided visual attention approach led to 85.5% accuracy when trained on CUB-200-2011 (Liu et al, 2016) ==cite==. We intend to examine the convolutional filter responses of different network layers, in order to find filters that correspond to bird parts. This is more realistic to achieve with a shallower architecture than ResNet, and one without the complex and more opaque submodules as in Inception.

### Modifications to VGG-16 in terms of tradeoffs

#### Fully-connected layer parameters

We load Simonyan and Zisserman’s pre-trained weights to our convolutional layers, which they had used to achieve 92.7% accuracy on ImageNet. However, we replace the fully-connected layers (two with 4096 neurons, one with 1000 neurons) with the following: one with 4096, one with 1024, and one with 50. We train our classifier to identify a subset of 50 among the 200 classes. We choose 1024 neurons for the middle fully-connected layer for our intuition that 4096 neurons dropping suddenly to 50 neurons would have a harsh effect on classification accuracy. As in the original architecture, however, we maintain the ReLU activations following fully-connected layers (all except for the last), in order to better cope with nonlinearity in the data.

#### Applying batch normalization

We also experiment with other intermediary layers between the fully-connected layers. In backpropagation, the variance of weight (parameter) distributions in one layer impact the variance of weights in the next layer. In deep networks with many layers, this becomes a significant issue that contributes to time needed to converge. To mitigate this problem, we try applying batch normalization after the first two fully-connected layers. This involves normalizing each feature activation $x_i^{(k)}$ by the variance and mean of that feature activation $k$, across all samples $i$ in the input batch. This gives the normalized value $\hat{x_i^k} = (x_i^k - \mu_B) / \sqrt{\sigma^2_B + \epsilon}$ for the input batch $B$ and some $\epsilon$. Among the samples of that batch, this normalizes the distribution of a feature activation to values in a unit Gaussian, which is desirable to prevent, in the words of Szegedy, "internal covariate shift" between the output of this layer and the input of the next, that would make training take longer to converge. "Batch," here, is used to refer to the number of training examples in a batch, whereas "feature activations" can be thought of as neuron outputs (Ioffe and Szegedy, 2015) ==cite==. Because it converges faster, the practice of batch normalization generally allows the use of a higher learning rate, requiring fewer iterations to reach the same accuracy.

#### Applying dropout

To further improve the performance, we experiment with applying a dropout layer after batch normalization. The tradeoff is that dropout requires more iterations to converge, but promotes better generalization among weights. Thus, it balances out the speed improvements in batch normalization. This allows us to maintain hyperparameters, such as learning rate, whether or not we utilize batch normalization and dropout.

Dropout works by randomly selecting nodes, for some layer, to ignore with some probability $p$ (specified as a hyperparameter). This disables the effect of the node on the ultimate decision, including all its incoming and outgoing activations. When a neural network is being trained without dropout, nodes develop co-dependencies and thus represent less robust decisions on their own. Training with dropout reduces these co-dependencies. Testing using a dropout-trained network involves reducing the entire network's activations by the same factor $p$ (Srivastava et al, 2014) ==cite==

#### Xavier initialization of weights

Our fully-connected layer implementation, in Tensorflow's `contrib.layers` library, makes use of Xavier weight initialization. It has benefits that will be elaborated upon here. In short, it ensures that weights in a layer will begin at a reasonable scale: not too small to cause any incoming signal to disappear, and not so large that it will distort the true significance of an incoming signal. Here, signal is interpreted as the incoming activations from a previous layer, and weights "begin" means that weights are trained and change (grow or shrink) with iteration.

Consider an output $Y$ that can be expressed in terms of $n$ components of input $X$ and some weights, $W$, such that:
$$
Y = W_1X_1 + W_2X_2 + \dots + W_nX_n
$$
We intend for $Y$ to have the same variance as that of input $X$, and so we choose variance of weights $W$ accordingly. Considering that $X, W$ both have mean 0:
$$
\text{Var}(W_iX_i) = \text{Var}(W_i)\text{Var}(X_i)
$$
And assuming that $X_i, W_i$ are i.i.d., combining with the first equation:
$$
\text{Var}(Y) = n\text{Var}(W_i)\text{Var}(X_i)
$$
To satisfy our goal, we only have to choose a value such that $n\text{Var}(W_i) = 1$. Hence, $\text{Var}(W_i) = 1/n$. In practice, $n_{\text{in}}$ differs from $n_{\text{out}}$, and so the implementation in Tensorflow uses an average of the two, such that:
$$
\text{Var}(W_i) = \frac{2}{n_\text{in} + n_\text{out}}
$$
Weights are sampled such that they satisfy this variance. (Glorot and Bengio, 2010)

### Training process

#### Mini-batch gradient descent

We use mini-batch gradient descent to train our model, a method that randomly samples a batch of input with which to update network weights. The batch size stays constant as a hyperparameter. Batches are randomly sampled under the assumption that each batch is a representative distribution of all of the input data. Any noise introduced due to using different batches may actually help the model to avoid local minima in loss [cite Brownlee].

Since using smaller batch sizes translates to more frequent updates, we can justify our choice of mini-batch gradient descent with our need to quickly see results and iterate upon our choice of architecture and hyperparameters. However, we also choose mini-batch gradient descent due to constraints in our memory. Since each input image is relatively large, we must limit our batch size to fit in memory; we settle on batch size of 64.

#### Transfer learning: two-stage training process

As mentioned previously ==cite section==, we load our model with pre-trained weights for the convolutional layers, and randomly initialize parameters for the fully-connected layers. As described before, the convolutional layers act as feature extractors and in this case they have already been trained on images. Thus, we can expect them to perform reasonably with other image-based feature extraction tasks, such as extracting parts from bird images. With this intuition, we can employ transfer learning, which is for us a two-stage training process:

1. Fix the weights of the convolutional layers to update only the weights of the fully-connected layers. This results in fewer weight updates and thus faster training.
2. Un-fix the weights of the convolutional layers and train the entire network at once. Set a much smaller learning rate for this step. This fine-tunes the feature-extracting convolutional layers by adapting them to birds.

#### Sigmoid cross entropy loss

We choose to optimize our network's weights to minimize a sigmoid cross entropy loss. Beyond being widely used in deep learning architectures, it is advantageous over mere MSE or classification loss in that it takes advantage of the underlying probability distribution rather than merely the accuracy of the prediction. Intuitively, it measures the difference between some one-hot encoding of the label, such as (for 3 classes) $z = (0, 1, 0, 0)^T$ and the prediction $x = (0.1, 0.4, 0.3, 0.05)^T$.

Tensorflow provides a sigmoid cross entropy loss function [^1] that is defined by the following:
$$
C = z\cdot - \log(\sigma(x)) + (1 - z) \cdot -\log(1 - \sigma(x))
$$
where $ \sigma(x) = 1 / (1 + e^{-W_j x + b})$ for some weight vector $W_j$ and bias $b$. For all training examples $x$ among $1 \leq i \leq n$, this is an average:
$$
C = -\frac{1}{n} \sum_i z \log(\sigma(x_i)) + (1 - z)\cdot \log(1 - \sigma(x_i))
$$
Some properties of this function allow it to be intrepreted as cost: it is always nonnegative, given that $x$ ranges between $[0, 1]$, and it minimizes $C$ if $\sigma(x)$ aligns with $z$, whether they are each close to $0$ or $1$ (for a 1-length feature vector $x$, although $x$ can be multidimensional), as binary classification outputs.

The sigmoid function is chosen because it makes for a high cost gradient if the error is large, so that the network learns faster. This is shown by taking the partial derivative of cost with respect to some dimension's weight $W^{(i)}$ [^2]:
$$
\sigma'(x) = \sigma(x) \cdot (1 - \sigma(x) \\
\begin{align}
\frac{\partial{C}}{\partial{W^{(i)}}} &= - \frac{1}{n} \sum_i (\frac{z}{\sigma(x_i)} - \frac{1 - z}{1 - \sigma(x_i)}) \cdot \sigma'(x_i)x_i^{(j)}\\
&= \frac{1}{n} \sum_i \frac{\sigma'(x_i)x_i^{(j)}}{\sigma(x_i)(1 - \sigma(x_i)} \cdot (\sigma(x_i) - z) \\
&= \frac{1}{n} \sum_i x_i^{(j)} \cdot (\sigma(x_i) - z)
\end{align}
$$
Maximizing the error $\sigma(x_i) - z$ maximizes the learning rate.

#### AdamOptimizer

AdamOptimizer keeps a distinct learning rate per parameter, rather than the same learning rate for all parameters, and these rates change during training. It borrows this from AdaGrad, which performs well when there are sparse gradients, such as in image classification tasks like our own [^3]. In practice we find that this causes our model to converge within just a few epochs, and it is easier for us to find hyperparameters that produce good results. However, using AdamOptimizer incurs significant space overhead by requiring a learning rate per parameter. The alternative is spending much more time tuning hyperparameters with GradientDescentOptimizer, which maintains a constant learning rate for all parameters, so this is acceptable for our case.

## Geometry in image feature extraction

### Shortcomings of hand-crafted feature extraction

In the past, the problem of image classification was solved with more hand-crafted features, taken across multiple scales of the image. One such feature is the Histogram of Oriented Gradient (HoG) descriptor, which is a grid where each grid element is a bin containing a histogram of gradient orientations in that part of the original image. Originally, HoG descriptors were used to solve detection problems, such as the PASCAL VOC challenge, where a well-known approach [^6] learned HoG filters that summarized the whole image, as well as HoG filters that summarized its parts, where each bin stood for finer-grained details. Scoring the detection involved creating HoG features of the image at multiple scales, sweeping the learned HoG detectors over their respective scales, and computing an overall score indicating how well the HoG detectors activated on the image's HoG features with respect to how deformed some detected parts were from their original position and appearance.

Such HoG approaches have also been used for classification. Krause et al described an approach toward fine-grained classification in their 2014 paper [^7], where:

1. Images were clustered as nearest neighbors in HoG space, with a HoG descriptor taken over the whole image. Here they classified cars, and the result of this action is groups of car images depicting cars in similar poses. 
2. From each cluster, part detectors were learned to identify car parts as they were commonly depicted in that pose.
3. Linear SVM's are used to classify car images using these part detectors.

It could be argued that these earlier hand-crafted approaches exploited geometry of images more explicitly, in defining the image scales, the bin sizes and orientations, choosing whether or not to ignore color, and choosing evenly-spaced bins at each scale level to summarize the image gradients. This accompanies other heuristics that are difficult to manage in hand-crafted approaches. One example of another heuristic is pose normalization: images must either be transformed to eliminate diverse pose from the input feature space, or part representations must account for all poses. Despite their explicit use of image geometry, choosing the scale discretization and pose normalization are two examples of heuristics that are difficult to tune in order to create classifiers that extend to other domains.

### Feature extraction using convolutional neural networks

Works in recent years have explored convolutional neural networks for image classification tasks, and they have found CNN's to perform well even for fine-grained classification tasks such as CUB-200-2011. These methods make use of image geometry in various ways, such as transforming images to normalize pose and feeding the resulting features into the CNN [^8].

With that said, using CNN's for feature extraction shares some similarities with HoG approaches. CNN's convolve their filters over all positions in an image, meaning that for each position they compute the dot product over each pixel in the filter and the corresponding pixel in the image. Doing this for all pixels obtains a activation map for that filter. A similar HoG response map involves computing correlation over all bins in the HoG representation of the input image and some HoG filter. The difference is that whereas HoG approaches scale the image to produce levels of responses, CNN approaches' scale is inherent in feeding lower layer's activation map as input into higher CNN layer.  Thus, as layer level increases, spatial receptivity increases (each pixel in that layer corresponds to a larger region of the original input). As a result, lower layers extract general image features such as edges and blobs, whereas higher layers extract parts of birds.

To address Dr. Vouga's point that a CNN may emulate a HoG pipeline for some choice of weights, work has been done by Girshick et al [^9] to map the concept of max-pooling in a CNN – a layer that reduces input size – to distance transforms in a Deformable Part Model, which measures a part detector match's deviation in position. Their work mainly addresses detection rather than classification, but it does perform detection with an order of magnitude more efficiency than using an R-CNN architecture.

Below are images illustrating the activation maps at increasing levels of our VGG-16 CNN architecture:

==TODO==

### Exploiting geometry of CNN filters

Whereas many approaches have attempted to improve fine-grained classification accuracy by either modifying the input (image) or output (classification layers) of a CNN pipeline, even more recent work has dwelled on the convolutional layers themselves. In particular, these works explore second order statistics of convolutional filters, and one such method called Bilinear Pooling improves accuracy of fine-grained classification tasks such as CUB-200-2011. Effectively, this family of work brings image classification and other tasks full-circle, being inspired by the explicit geometry of hand-crafted pipelines but generalizing to improve general architecture of CNN's [^10]. Because of time constraint and implementation difficulty, we will not delve into these improvements. However, we noticed VGG-16 arise as a baseline architecture that many of these approaches improve, so this led us to settle upon understanding how to employ VGG-16 architecture on our dataset as a basis for future work.

### Classification layer geometry tradeoffs

We have considered using an SVM, in place of the fully-connected layers at the end, for classifying the features extracted by convolutional layers. However, we have come across some guidelines for fine-tuning pre-trained models [^11] that reinforce our decision to use fully-connected layers and minimize a logistic loss. Relative to the size of ImageNet (upwards of 450k training samples), which our VGG-16 was pre-trained on, the CUB-200-2011 dataset (our 50-class subset, about 1.5k training samples) is relatively small, but contains similar training examples. SVM's are suggested to adapt neural networks to classify relatively small but very different datasets.

We can also justify using logistic regression rather than SVM's for our case based on some intuition about the rest of the pipeline. Taken as a whole, the neural network is a probabilistic interpretation of the data. Maximizing a margin as with an SVM classifier is less intuitive for the probabilistic output embedding of the convolutional layers. This makes minimizing a logistic loss more consistent with the rest of the model than a hinge loss.

## Results and Analysis

### Accuracy improvements since presentation

At the time of our presentation, our results featured test accuracies of 6.2%, 16.5%, and 26.8% (for top-1, top-3, and top-5 respectively) on 20 classes. Our training accuracy was shown to increase, and this led us to believe that we were training correctly – that our architecture was merely overfitting on the training data.

First, we decided to partition our dataset into a more conventional ratio, and this time include a validation set. We chose to improve upon our original training and testing set generation method, which was to randomly choose whether an image would be in the training or testing set with 90% or 10% probability. In this random choice, we ignored the class of the image, so this method had the potential to skew the training set towards a class that was not as well-represented in the testing set, or vice versa. Instead, for each class, we distributed 50% of images to the training set, 25% to the validation set, and 25% to the testing set. We found that these were more widely accepted ratios for partitioning the data. Including a validation set also allowed us to monitor our pipeline's performance through epochs, in order to identify potential issues.

We originally trained and tested among the data for 20 classes to reduce training time and iterate more quickly. However, this new partitioning greatly reduced our training set size and we feared overfitting to the training data. Especially since our pipeline learns a feature extractor, we wanted to avoid limiting the bird images that the extractor would learn parts from. So we increased the number of classes to 50.

After fixing other bugs, we arrived at much better increasing training accuracy, but our validation accuracy remained level, at mere chance probability. We tried many different things to try and resolve the issue, but eventually we realized that the library that we were using to read in images was misinterpreted by the neural network architecture. We found a library that was more compatible with the architecture, and immediately saw our validation accuracy rise in accordance with training accuracy. We also observed our testing accuracy improve overall.

### Visualizing training results

![vis](../../../../../../Downloads/vis.png)

To explain the labels in the legend:

* Prefix  `BN_DO` refers to runs with batch normalization and dropout in the architecture, using AdamOptimizer
* Prefix `REG` refers to runs with neither batch normalization nor dropout, using AdamOptimizer
* Prefix `GD` refers to runs with neither batch normalization nor dropout, using GradientDescentOptimizer

Other notes:

* All but `200REG` were trained and validated on as subset of 50 classes. `200REG` represents training and validation on the full dataset of 200 classes.
* Warm-up training phase is 15 epochs, and full-network phase is 10 epochs. These values were chosen through experimentation.

==Testing accuracy==

#### Generalizability of pre-trained weights

These results demonstrate that training accuracy quickly converges. In fact, it crosses the 90% accuracy threshold at epoch 3, which is well before the second stage (full-network) of our transfer learning (epoch 16). This shows that the convolutional layers, even though they were trained on ImageNet, generalize reasonably well as a feature extractor for bird images. _To examine some specifics, we turn to Dr. Vouga's concern that mean-centering the colors of bird images might cause us to lose important color information to help us classify the bird. To take the blue bird as an example, we would expect…_

==TODO confusion matrix for test accuracy of blue bird: it should be more confident about the blue bird than the others, if input whitening has no effect==

#### Effect of batch normalization and dropout

The training accuracy curves of runs with batch normalization and dropout (`BN_DO`) are consistently lower than runs without them (`REG`). Including them also prevents loss from converging to 0, which would indicate to the model that a minimum has been reached and that there is no room left for training improvement. Thus, `BN_DO` runs prevent overfitting on the training data. Still, however, their validation and testing accuracy leaves a little to be desired, since `BN_DO` runs under-perform VGG-16 without batch normalization and dropout (`REG`).

#### AdamOptimizer vs. GradientDescentOptimizer

AdamOptimizer was used in all of the displayed runs except for `GD1`. `GD1` is included for comparison, to show how much more slowly it takes to converge a network with uniform learning rates across all parameters (weights), rather than keeping per-parameter learning rates in AdamOptimizer.

### Extending to 200 classes

_more classes = 50, test on 200 classes_

### Future work: achieving better bounds

#### Factors in dropout/batch normalization convergence

From the visualizations, it is obvious that batch normalization and dropout take more epochs overall to converge in accuracy. They introduce additional hyperparameters, such as the probability with which dropout keeps a node in the network. Variations of these compound on other parameters, such as the ratio between warm-up and full-network epochs and batch size. Future work could involve experimentation on these hyperparameters to have batch normalization and dropout achieve their intended effect.

---

Citations:

- Ioffe and Szegedy, batch normalization: https://arxiv.org/pdf/1502.03167.pdf
- Srivastava, dropout: https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf
- Glorot and Bengio, Xavier initialization: http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf
- Jason Brownlee on Mini-batch gradient descent: https://machinelearningmastery.com/gentle-introduction-mini-batch-gradient-descent-configure-batch-size/
- when and how to fine tune: http://cs231n.github.io/transfer-learning/

[^1]: https://www.tensorflow.org/api_docs/python/tf/nn/sigmoid_cross_entropy_with_logits
[^2]: http://neuralnetworksanddeeplearning.com/chap3.html#introducing_the_cross-entropy_cost_function
[^3]: https://machinelearningmastery.com/adam-optimization-algorithm-for-deep-learning/
[^6]: https://cs.brown.edu/~pff/papers/lsvm-pami.pdf
[^7]: http://ai.stanford.edu/~tgebru/papers/icpr14.pdf
[^8]: https://arxiv.org/pdf/1406.2952.pdf
[^9]: https://arxiv.org/pdf/1409.5403.pdf
[^10]: http://vis-www.cs.umass.edu/bcnn/docs/improved_bcnn.pdf
[^11]: http://cs231n.github.io/transfer-learning/