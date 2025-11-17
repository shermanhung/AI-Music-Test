---
title: "Technical Project"
excerpt: "This folder contains descriptions of technical projects that demonstrate my AI-related aptitude.<br/>" 
collection: portfolio
---

## Project Overview

I served as the lead data scientist for a six-month collaborative research project between 4Paradigm and China Petro. Working closely with geological experts, I translated domain-specific challenges into well-defined AI modeling tasks, designed and implemented the full solution in Python, and took end-to-end responsibility for model performance. Through this project, I gained extensive hands-on experience in time-series data processing—skills that transfer directly to audio signal analysis in music. I also conducted in-depth research on time-series shapelet similarity algorithms, reviewing approximately 50–60 academic papers. This included running numerous modeling experiments using TensorFlow and scikit-learn, as well as adapting and extending open-source implementations from GitHub to refine model architectures — work very similar to what I expect to do in my graduate research.

## Project Problem
<img src="https://github.com/shermanhung/shermanhung.github.io/blob/c315227edc3258023ee0c4e0396dcd18c161c6de/images/Well%20logging%20curves%20of%20a%20well.png" align="left" width="200" title="Figure 1 - Well-logging Curves"/>

The data used in this project consists of well-logging curves. Figure 1 illustrates this type of data, which can be viewed as a form of time-series signal where the time axis is replaced by depth (y-axis), and geophysical measurements are recorded at regular spatial intervals (blue curve along the x-axis). Conceptually, these signals are similar to audio waveforms used in music information retrieval, allowing many of the same signal-processing and pattern-analysis techniques to be applied.

Given a sequence of geophysical measurements collected at different depths within a well, the goal is to segment the well into distinct geological strata, each defined by a top and bottom boundary (shown as red layer boundaries in Figure 1). This project aims to automate this process using an AI-based solution, reducing the need for manual interpretation by geological experts.

