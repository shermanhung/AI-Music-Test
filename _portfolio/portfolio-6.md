---
title: "Technical Project"
excerpt: "This folder contains descriptions of technical projects that demonstrate my AI-related aptitude.<br/>" 
collection: portfolio
---

## Project Overview

I served as the lead data scientist for a six-month collaborative research project between 4Paradigm and China Petro. Working closely with geological experts, I translated domain-specific challenges into well-defined AI modeling tasks, designed and implemented the full solution in Python, and took end-to-end responsibility for model performance. 

Through this project, I gained extensive hands-on experience in time-series data processing—skills that transfer directly to audio signal analysis in music. I also conducted in-depth research on time-series shapelet similarity algorithms, reviewing approximately 50–60 academic papers. This included running numerous modeling experiments using TensorFlow and scikit-learn, as well as adapting and extending open-source implementations from GitHub to refine model architectures — work very similar to what I expect to do in my graduate research.



## Project Problem
<img src="https://github.com/shermanhung/shermanhung.github.io/blob/845fdede152fd43c09893780542d617e87fb8d02/images/Well%20logging%20curves%20of%20a%20well.png" align="left" width="200" height="250" title="Figure 1"/>

The data used in this project consists of well-logging curves. Figure 1 illustrates this type of data, which can be viewed as a form of time-series signal where the time axis is replaced by depth (y-axis), and geophysical measurements are recorded at regular spatial intervals (blue curve along the x-axis). Conceptually, these signals are similar to audio waveforms used in music information retrieval, allowing many of the same signal-processing and pattern-analysis techniques to be applied.

Given a sequence of geophysical measurements collected at different depths within a well, the goal is to segment the well into distinct geological strata, each defined by a top and bottom boundary (shown as red layer boundaries in Figure 1). This project aims to automate this process using an AI-based solution, reducing the need for manual interpretation by geological experts.

## Methodlogy

<img src="https://github.com/shermanhung/shermanhung.github.io/blob/2f7cdbd110b85ffe9b74ac11d98ad7f26c584ad0/images/Standard%20well%20%26%20Comparison%20well.png" align="left" width="200" height="400" title="Figure 2"/>

<img src="https://github.com/shermanhung/shermanhung.github.io/blob/cfb643d320f26ff758e8f236daeed1787986c82a/images/Comparision%20well%20candidates.png" align="left" width="200" height="400" title="Figure 3"/>

To begin the project, I worked closely with geologists to understand their manual workflow and identify which steps could be automated with AI. As illustrated in Figure 2, their process starts by designating one well as the standard well, where geological intervals have been manually annotated. To interpret other wells (the comparison wells), geologists visually match their curves against the standard well to identify corresponding strata.

Following the same reasoning, I reformulated the problem as a template-matching task. As shown in Figure 3, the annotated intervals from the standard well (numbered in the figure) act as templates. The goal is to find, for each template, the most similar interval among all candidate intervals in a comparison well. For example, to identify Layer 1 in a comparison well, we take Layer 1 from the standard well as the template and compare it against all candidate intervals in the comparison well. The most similar candidate is then assigned as Layer 1 for that comparison well.

This approach involves two key technical challenges:
1.	How can we generate candidate intervals on the comparison wells?

To address the first challenge of generating candidate intervals on comparison wells, we applied an activation-function heuristic grounded in geological principles. The idea is straightforward: stratigraphic boundaries typically occur where one rock type transitions to another, causing abrupt changes in the subsurface’s physical properties. These changes are reflected in well-logging curves, where the largest shifts correspond to points with the highest derivatives. In other words, the stronger the change in the logging signal, the more likely it marks a true geological boundary.

The activation-function procedure is as follows: we first select a comparison well, compute the derivative of its well-logging curves, and identify the points with the largest derivative values (the peak points). These peaks form the basis for constructing candidate intervals. Each peak is associated with a specific depth, and every peak is paired with all subsequent peaks located at greater depths.

For example, as illustrated in Figure 4, if Well NP203 has its first peak at 2110 meters and its second peak at 2120 meters, the interval from 2110 m to 2120 m becomes a candidate interval. Reverse pairings do not apply—the second peak is not paired with the first.

2.	Once the candidate intervals are generated, how can we identify the ones most similar to the template intervals on the standard well?

To address the second challenge — identifying the most similar candidate interval on the comparison wells — we developed an XGBoost binary classifier. The model takes two well-logging curve intervals as input: one from the standard well’s template and one from a candidate interval on a comparison well. It outputs a probability score indicating their similarity.

The effectiveness of this classifier depends on careful feature engineering. We designed two types of features to capture similarity at different levels:
-	Global similarity (Figure X): The full curves from the standard and comparison wells are processed using various shape-extraction algorithms, which produce continuous values representing the overall similarity between the curves. These values serve as features for the XGBoost classifier.
-	Local similarity (Figure Y): Each curve is divided into six equal-length segments. Every segment from the standard well is paired with each segment from the comparison well, creating multiple segment pairs. For example, segment 1 of the standard well is paired sequentially with segments 1 through 6 of the comparison well. Each pair is processed through the shape-extraction algorithms, producing continuous values that reflect local similarity. These values are also used as features for the XGBoost classifier.

The shape-extraction algorithms employed include Dynamic Time Warping, Minimum Jump Cost, ROCKET, InceptionTime, Shapelets, and various statistical features. Detailed descriptions of each algorithm are provided in the appendix and referenced therein.
