Performance-Interpretability Trade-offs and Generalization in Deep Learning for Pneumonia Detection: A Benchmarking Study of CNN Architectures
==============================================================================================================================================

**Deployment & Documentation & Stats**

.. image:: https://img.shields.io/github/stars/SynergIA-Lab/pneumoniacnn.svg
   :target: https://github.com/SynergIA-Lab/pneumoniacnn/stargazers
   :alt: GitHub stars


.. image:: https://img.shields.io/github/forks/SynergIA-Lab/pneumoniacnn.svg?color=blue
   :target: https://github.com/SynergIA-Lab/pneumoniacnn/network
   :alt: GitHub forks


.. image:: https://img.shields.io/badge/License-BSD_3--Clause-blue.svg
   :target: https://github.com/SynergIA-Lab/pneumoniacnn/blob/main/LICENSE
   :alt: License

----


Chest X-ray imaging continues to be the primary radiological tool for the early detection of pneumonia, which is one of the leading causes of morbidity and mortality worldwide. Recent advances in deep learning have demonstrated strong potential for automating diagnosis; however, the clinical applicability of such systems depends not only on predictive performance but also on model transparency and robustness. 

In this study, **we evaluate the diagnostic capabilities and interpretability of multiple convolutional neural network architectures using a widely adopted public dataset of pediatric chest radiographs**. Five CNN architectures were trained and evaluated under a standardized cross-validated protocol. Performance was assessed using accuracy, precision, recall, F1-score, and area under the ROC curve (AUC). To ensure clinical reliability, the benchmarking was complemented with an extensive eXplainable AI (XAI) analysis using three different methods applied to both pneumonia and normal cases in a qualitative and quantitative study. In addition, external validation was performed using a dataset from a different cohort (adults).

The results obtained show that two of these architectures consistently and significantly achieved the highest diagnostic performance across all folds, with higher F1-score and AUC values, while producing the most focused and clinically plausible XAI explanations. Moreover, the external dataset validation reveals a consistent architectural ranking across independent datasets, suggesting that performance-interpretability coupling is robust to population variation and imaging protocol differences supporting clinical generalization of our framework.

These findings highlight that robust benchmarking combined with explainability and external validation is essential for developing trustworthy deep learning systems for medical imaging. Both the dataset and code used in this study are publicly available in this repository.

This study is featured for:

* Unified benchmarking of five CNN architectures for pneumonia detection from chest X-ray images.
* Combined qualitative and quantitative evaluation of model explainability using Saliency Maps, SmoothGrad, and Grad-CAM.
* Quantitative XAI metrics reveal a strong link between explanation stability and predictive performance.
* External validation on an independent adult cohort confirms generalization trends across populations.
* Fully reproducible framework applicable to other medical imaging classification tasks.

----

Execution
=========

1) **Training models:** First, the models must be trained. This file includes CNN training and predictive performance evaluation using 5-fold stratified cross-validation with class-balanced undersampling.

.. code-block:: bash

   python3 train_kfold.py

2) External validation actions and XAI evaluations can be carried out in parallel.

2.1) **XAI execution:** This study performs a qualitative analysis (xai_qualitative.py) and a quantitative analysis (xai_quantitative.py).

.. code-block:: bash

   python3 xai_qualitative.py
   python3 xai_quantitative.py

2.2) **External validation:** Generalization experiment on an external pneumonia dataset. The model is NOT retrained. The best model from the paper (fold 1) is evaluated on a balanced external dataset (500 NORMAL + 500 PNEUMONIA).

.. code-block:: bash

   python3 external_validation.py

**Required Dependencies**\ :

* **Python**>=3.11
* **numpy**>=2.0.1
* **pandas**>=2.2.2
* **tensorflow**>=2.17.1
* **matplotlib**>=3.9.0
* **scikit-learn**>=1.6.0

----

**Cite us**\ :

This paper is under review.
We would appreciate citations to the following paper::

   Under review
