---
title: "Technical Project"
excerpt: "This folder contains descriptions of technical projects that demonstrate my AI-related aptitude.<br/>" 
collection: portfolio
---

## Project Overview

I served as the lead data scientist for a six-month collaborative research project between 4Paradigm and China Petro. Working closely with geological experts, I translated domain-specific challenges into well-defined AI modeling tasks, designed and implemented the full solution in Python, and took end-to-end responsibility for model performance. 

Through this project, I gained extensive hands-on experience in time-series data processing—skills that transfer directly to audio signal analysis in music. I also conducted in-depth research on time-series shapelet similarity algorithms, reviewing approximately 50–60 academic papers. This included running numerous modeling experiments using TensorFlow and scikit-learn, as well as adapting and extending open-source implementations from GitHub to refine model architectures — work very similar to what I expect to do in my graduate research.

## Project Problem
<img src="https://github.com/shermanhung/shermanhung.github.io/blob/845fdede152fd43c09893780542d617e87fb8d02/images/well%20logging%20curves%20of%20a%20well.png" align="left" width="200" height="250" title="figure 1"/>

The data used in this project consists of well-logging curves. Figure 1 illustrates this type of data, which can be viewed as a form of time-series signal where the time axis is replaced by depth (y-axis), and geophysical measurements are recorded at regular spatial intervals (blue curve along the x-axis). Conceptually, these signals are similar to audio waveforms used in music information retrieval, allowing many of the same signal-processing and pattern-analysis techniques to be applied.

Given a sequence of geophysical measurements collected at different depths within a well, the goal is to segment the well into distinct geological strata, each defined by a top and bottom boundary (shown as red layer boundaries in Figure 1). This project aims to automate this process using an AI-based solution, reducing the need for manual interpretation by geological experts.

## Methodlogy

<img src="https://github.com/shermanhung/shermanhung.github.io/blob/2f7cdbd110b85ffe9b74ac11d98ad7f26c584ad0/images/Standard%20well%20%26%20Comparison%20well.png" align="left" width="200" height="330" title="Figure 2"/>

<img src="https://github.com/shermanhung/shermanhung.github.io/blob/d66095c4f2b2b2fe0ea25056e77cf4141bf3bb91/images/comparision%20well%20candidates.png" align="left" width="200" height="330" title="Figure 3"/>

To begin the project, I worked closely with geologists to understand their manual workflow and identify which steps could be automated with AI. As illustrated in Figure 2, their process starts by designating one well as the standard well, where geological intervals have been manually annotated. To interpret other wells (the comparison wells), geologists visually match their curves against the standard well to identify corresponding strata.

Following the same reasoning, I reformulated the problem as a template-matching task. As shown in Figure 3, the annotated intervals from the standard well (numbered in the figure) act as templates. The goal is to find, for each template, the most similar interval among all candidate intervals in a comparison well. For example, to identify Layer 1 in a comparison well, we take Layer 1 from the standard well as the template and compare it against all candidate intervals in the comparison well. The most similar candidate is then assigned as Layer 1 for that comparison well.

## Technical Details

The aboved approach involves two key technical challenges:

### 1.	How can we generate candidate intervals on the comparison wells?

<img src="https://github.com/shermanhung/shermanhung.github.io/blob/ef753ac8bba0efda6f6e7100b2733f507dd09842/images/derivative%20function.png" align="left" width="200" height="350" title="Figure 4"/>

To address the first challenge of generating candidate intervals on comparison wells, we applied an activation-function heuristic grounded in geological principles. The idea is straightforward: stratigraphic boundaries typically occur where one rock type transitions to another, causing abrupt changes in the subsurface’s physical properties. These changes are reflected in well-logging curves, where the largest shifts correspond to points with the highest derivatives. In other words, the stronger the change in the logging signal, the more likely it marks a true geological boundary.

The activation-function procedure is as follows: we first select a comparison well, compute the derivative of its well-logging curves, and identify the points with the largest derivative values (the peak points). These peaks form the basis for constructing candidate intervals. Each peak is associated with a specific depth, and every peak is paired with all subsequent peaks located at greater depths.

For example, as illustrated in Figure 4, if Well NP203 has its first peak at 2110 meters and its second peak at 2120 meters, the interval from 2110 m to 2120 m becomes a candidate interval. Reverse pairings do not apply—the second peak is not paired with the first.

### 2.	Once the candidate intervals are generated, how can we identify the ones most similar to the template intervals on the standard well?

<img src="https://github.com/shermanhung/shermanhung.github.io/blob/f4c511c6a28da6636ba1247587f144408eca4abb/images/full%20curve%20features.png" align="left" width="300" height="220" title="Figure 5"/>

<img src="https://github.com/shermanhung/shermanhung.github.io/blob/f4c511c6a28da6636ba1247587f144408eca4abb/images/local%20curve%20features.png" align="left" width="300" height="220" title="Figure 6"/>

To address the second challenge — identifying the most similar candidate interval on the comparison wells — we developed an XGBoost binary classifier. The model takes two well-logging curve intervals as input: one from the standard well’s template and one from a candidate interval on a comparison well. It outputs a probability score indicating their similarity.

The effectiveness of this classifier depends on careful feature engineering. We designed two types of features to capture similarity at different levels:
-	Global similarity (Figure 5): The full curves from the standard and comparison wells are processed using various shape-extraction algorithms, which produce continuous values representing the overall similarity between the curves. These values serve as features for the XGBoost classifier.
-	Local similarity (Figure 6): Each curve is divided into six equal-length segments. Every segment from the standard well is paired with each segment from the comparison well, creating multiple segment pairs. For example, segment 1 of the standard well is paired sequentially with segments 1 through 6 of the comparison well. Each pair is processed through the shape-extraction algorithms, producing continuous values that reflect local similarity. These values are also used as features for the XGBoost classifier.

The shape-extraction algorithms employed include Dynamic Time Warping, Minimum Jump Cost, ROCKET, InceptionTime, Shapelets, and various statistical features. Detailed descriptions of each algorithm are provided in the appendix and referenced therein.

## Appendix

### Dynamic Time Warping

Dynamic Time Warping (DTW) measures the similarity between two time series, even when they differ in length or speed. It aligns the sequences by stretching or compressing segments so that similar shapes match, then computes the optimal alignment. The result is a continuous score indicating how similar the two overall patterns are, even if they are not aligned in time.

### Minimum Jump Cost

Minimum Jump Cost (MJC) measures how different two time series are by repeatedly “jumping” forward between their data points. Starting at the beginning of one curve, the algorithm alternates between the two sequences, each time choosing the smallest forward jump until it reaches the end of either series. The total distance of these jumps becomes the dissimilarity score: similar curves produce short, low-cost jumps, while dissimilar curves result in much larger cumulative cost.

### ROCKET (RandOm Convolutional KErnel Transform)

ROCKET (RandOm Convolutional KErnel Transform) transforms time series using a large number of randomly generated convolutional kernels—each with random length, weights, bias, dilation, and padding—without learning any kernel parameters. For each kernel, two features are extracted from the resulting feature map: the maximum value, indicating the strongest pattern match, and the proportion of positive values (PPV), reflecting how often the pattern appears in the series. These features are then used to train our XGBoost binary classifier.

### InceptionTime

InceptionTime is a state-of-the-art time series classification model built as an ensemble of five Inception networks. Each network stacks Inception modules that include:
-	1×1 bottleneck convolutions to reduce dimensionality and prevent overfitting
-	Multi-scale convolutions with filters of varying lengths (e.g., 10, 20, 40) to capture patterns at different time scales
-	Residual connections to stabilize training and avoid vanishing gradients

An ensemble of five networks is used because a single Inception model can have high variance due to random initialization and stochastic training. Averaging their predictions yields more stable, state-of-the-art performance. The feature vector from the final hidden layer of the InceptionTime ensemble is then used as input to our XGBoost binary classifier.

### Shapelet

Learning Shapelets is a time series classification method that learns the most discriminative subsequences—called shapelets—directly from the data. Since classes are often distinguished by short, characteristic patterns rather than the entire series, the distances from a time series to these learned shapelets serve as informative features for our XGBoost binary classifier.

The method starts with initial shapelet candidates (often cluster centroids) and converts each time series into a feature vector by computing its minimum distance to each shapelet. These distance features are then passed to a simple classifier such as logistic regression. Both the shapelets and classifier weights are learned jointly using gradient descent; because the minimum-distance function is not differentiable, a soft-min approximation is used to allow backpropagation.

### Statistical Characteristics

This method computes the absolute differences between various statistical features of the two curves, including Average Rectified Value, Mean, Mean Square, Variance, Standard Deviation, Root Mean Square, Crest Factor, Impulse Factor, Margin Factor, Skewness, Kurtosis, and Form Factor.

## References
