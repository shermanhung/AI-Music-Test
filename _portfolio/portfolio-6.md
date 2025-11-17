---
title: "Technical Project"
excerpt: "This folder contains descriptions of technical projects that demonstrate my AI-related aptitude.<br/>" 
collection: portfolio
---

## Project Overview

I served as the lead data scientist for a six-month collaborative research project between 4Paradigm and China Petro. Working closely with geological experts, I translated domain-specific challenges into well-defined AI modeling tasks, designed and implemented the full solution in Python, and took end-to-end responsibility for model performance. 

Through this project, I gained extensive hands-on experience in time-series data processing—skills that transfer directly to audio signal analysis in music. I also conducted in-depth research on time-series shapelet similarity algorithms, reviewing approximately 50–60 academic papers. This included running numerous modeling experiments using TensorFlow and scikit-learn, as well as adapting and extending open-source implementations from GitHub to refine model architectures — work very similar to what I expect to do in my graduate research.

## Project Problem
<img src="https://github.com/shermanhung/shermanhung.github.io/blob/e67c5438bba20c8c972a944ee7e57cbf6b9bfaec/images/ShermanHung_Picture4.png" align="left" width="200" height="250" title="Figure 1"/>

The data used in this project consists of well-logging curves. Figure 1 illustrates this type of data, which can be viewed as a form of time-series signal where the time axis is replaced by depth (y-axis), and geophysical measurements are recorded at regular spatial intervals (blue curve along the x-axis). Conceptually, these signals are similar to audio waveforms used in music information retrieval, allowing many of the same signal-processing and pattern-analysis techniques to be applied.

Given a sequence of geophysical measurements collected at different depths within a well, the goal is to segment the well into distinct geological strata, each defined by a top and bottom boundary (shown as red layer boundaries in Figure 1). This project aims to automate this process using an AI-based solution, reducing the need for manual interpretation by geological experts.

## Methodlogy

<img src="https://github.com/shermanhung/shermanhung.github.io/blob/2f7cdbd110b85ffe9b74ac11d98ad7f26c584ad0/images/Standard%20well%20%26%20Comparison%20well.png" align="left" width="200" height="350" title="Figure 2"/>

<img src="https://github.com/shermanhung/shermanhung.github.io/blob/cfb643d320f26ff758e8f236daeed1787986c82a/images/Comparision%20well%20candidates.png" align="left" width="400" height="100" title="Figure 3"/>

To begin the project, I worked closely with geologists to understand their manual workflow and identify which steps could be automated with AI. As illustrated in Figure 2, their process starts by designating one well as the standard well, where geological intervals have been manually annotated. To interpret other wells (the comparison wells), geologists visually match their curves against the standard well to identify corresponding strata.



Following the same reasoning, I reformulated the problem as a template-matching task. As shown in Figure 3, the annotated intervals from the standard well (numbered in the figure) act as templates. The goal is to find, for each template, the most similar interval among all candidate intervals in a comparison well. For example, to identify Layer 1 in a comparison well, we take Layer 1 from the standard well as the template and compare it against all candidate intervals in the comparison well. The most similar candidate is then assigned as Layer 1 for that comparison well.

