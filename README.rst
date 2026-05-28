Performance-Interpretability Trade-offs and Generalization in Deep Learning for Pneumonia Detection: A Benchmarking Study of CNN Architectures
==============================================================================================================================================

**Deployment & Documentation & Stats**

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

Environment Setup
=================

To make setting up the environment as easy as possible across Windows, macOS, and Linux, we have included an automated setup script. 

Simply run the following command in your terminal from the project root:

.. code-block:: bash

   python setup.py

This script will automatically:
1) Create a local virtual environment (``.venv``) in the project root.
2) Upgrade ``pip`` inside the virtual environment.
3) Install all required dependencies from ``requirements.txt``.

*Note for IDE users (VS Code, Cursor, PyCharm):* The IDE will automatically detect the new ``.venv`` environment in the project folder. Once detected, clicking the "Play" button will execute the scripts using this environment automatically, ensuring all libraries (including ``keras-tuner``) are resolved.

Execution
=========

1) **Hyperparameter Tuning:** Find the optimal hyperparameter values using Bayesian Optimization:

.. code-block:: bash

   python code/1_tune_hyperparameters.py

Update the ``"hyperparameters"`` block in ``code/config.json`` with the best parameters output by the script.

2) **Training models:** Train the CNN architectures using 5-fold stratified cross-validation and multiple seeds for robustness:

.. code-block:: bash

   python code/2_train_kfold.py

3) **External validation:** Evaluate generalization performance of the models on an independent external dataset (adult cohort):

.. code-block:: bash

   python code/3_external_validation.py

4) **Explainable AI (XAI) execution:** Run the qualitative and quantitative evaluations:

.. code-block:: bash

   python code/4_xai_qualitative.py
   python code/5_xai_quantitative.py

----

**Cite us**\ :

This paper is under review.
We would appreciate citations to the following paper::

   Under review

----

**Download our models**\ :

You can download the training templates used from [here](https://drive.google.com/drive/folders/1loicfGMJPnHikJ2PMJ7HopHUj82NXAN9)
