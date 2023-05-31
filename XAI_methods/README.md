## XAI Metrics
In the paper, we have incorporated metrics defined in [Quantus by Hedström et. al. 2022](https://www.jmlr.org/papers/volume24/22-0142/22-0142.pdf) to quantify results obtained from the XAI Methods

1. Faithfulness

	* FaithfulnessCorrelation - </b><a href="https://www.ijcai.org/Proceedings/2020/0417.pdf">(Bhatt et al., 2020)</a>: Iteratively substituting a baseline value with a randomly chosen subset of the provided attributions. The correlation between the difference in the function's output and the total of these subset of attributions is then calculated. 
2. Robustness 

	* MaxSensitivity - </b><a href="https://arxiv.org/pdf/1901.09392.pdf">(Yeh et al., 2019)</a>: Utilizes Monte-Carlo sampling to assess the maximum change in the explanation with the introduction of slight perturbation to the input variable 'x'.
3. Localisation

	* RelevanceRankAccuracy - </b><a href="https://arxiv.org/abs/2003.07258">(Arras et al., 2021)</a>: measures the proportion of pixels with high attribution within a ground-truth mask in relation to the mask's size (Region of Interest).
4. Complexity 

	* Sparseness - </b><a href="https://arxiv.org/abs/1810.06583">(Chalasani et al., 2020)</a>: the attributions of irrelevant or weakly relevant features should be negligible,thus resulting in concise explanations in terms of the significant features
5. Randomisation

	* ModelParameterRandomisation - </b><a href="https://arxiv.org/abs/1810.03292">(Adebayo et. al., 2018)</a>: compares the output of a saliency method on a trained model with the output of the saliency method on a randomly initialized untrained network of the same architecture
6. Axiomatic metrics

	* Completeness - </b><a href="https://arxiv.org/abs/1703.01365">(Sundararajan et al., 2017)</a>: that the attributions add up to the difference between the output of F at the input x and the baseline x'.