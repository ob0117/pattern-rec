## [`emotions-challenge`](./emotions-challenge)

### Emotion Classification with Classical ML

This project tackles a sentiment analysis task **without** using any deep learning!

**Goal**: Classify tweets into one of six emotions: *joy, sadness, anger, fear, love, surprise*  
**Dataset:** [nelgiriyewithana/emotions](https://www.kaggle.com/datasets/nelgiriyewithana/emotions) (400K+ labeled tweets)

For this challenge, I tuned and stacked only **Random Forest** and **Linear Support Vector Machine (SVM)** classifiers. 

This was a neat demonstration of the power of classical ML models. Despite how informal and nuanced tweet language can be, with careful data preprocessing and a good understanding of each model’s strengths, it's possible to achieve results that rival deep learning approaches.

#### Results:
- **Validation Accuracy:** 94%  
- **Test Accuracy:** 93%  
- **Training Time:** ~6.5 minutes (Apple M2 chip)

#### Notebooks:
- [`01_data_processing.ipynb`](./emotions-challenge/01_data_processing.ipynb)  
  Data pre-processing, lemmatization
- [`02_model_training_eval.ipynb`](./emotions-challenge/02_model_training_eval.ipynb)  
  Vectorization, model training, tuning, validation, stacking


## [`style-transfer`](./style-transfer)

### Image Style Transfer with Gram Matrix & Sliced Wasserstein Distance

In this project, I implement two popular image style transfer methods from scratch, using feature activations from convolutional layers of a pre-trained VGG19 network on ImageNet, to extract content and style representations.

- **Gram Matrix (GM)** — Based on [*A Neural Algorithm of Artistic Style*](https://arxiv.org/pdf/1508.06576) by Gatys et al.  
  Represents correlations between VGG feature maps to represent the overall style of an image.

- **Sliced Wasserstein Distance (SWD)** — Based on [*Sliced Wasserstein Distance for Neural Style Transfer*](https://www.sciencedirect.com/science/article/pii/S0097849321002600) by Kolkin et al.  
  Projects flattened style feature maps onto random vectors and compares their distributions in order to transfer texture.

Both methods are built using TensorFlow (2.16+), with custom loss functions and gradient descent optimization loops.

#### Example Results:

<table>
  <tr>
    <td align="center"><img src="./style-transfer/content/cityscape.png" width="250px"><br><em>Content: Cityscape</em></td>
    <td align="center"><img src="./style-transfer/styles/starrynight.jpg" width="250px"><br><em>Style: Starry Night (Van Gogh)</em></td>
  </tr>
  <tr>
    <td align="center"><img src="./style-transfer/results/GM/cityscape%20and%20starry%20night/alpha%201000%20beta%2010%20gamma%200.01/epoch_10_step_100.png" width="300px"><br><em>Gram Matrix Output</em></td>
    <td align="center"><img src="./style-transfer/results/SWD/city%20scape%20and%20starry%20night/alpha%20100000%20beta%200.05%20gamma%200.1/epoch_10_step_100.png" width="300px"><br><em>SWD Output</em></td>
  </tr>
</table>


#### Notebook:
- [`style_transfer_methods.ipynb`](./style-transfer/style_transfer_methods.ipynb)  
  Full implementation, training, tuning, and results for both methods on 4 different content/style examples!

