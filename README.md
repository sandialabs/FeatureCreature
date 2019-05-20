# FeatureCreature

**FeatureCreature**  is an **open-source** software tool for structural visualization of machine learned **Quantitative Structure Activity Relationships (QSARs)** explanations.

**QSARs** are often *black-box models*, principally built using difficult to interpret *Artificial Neural Networks*, *Random Forests* (and related techniques), or other machine learning techniques. It is common to learn a QSAR that has very high *accuracy*, *precision* and *recall*; but little **interpretability**. The goal of **FeatureCreature** is to *visualize* locally explanations for QSAR predictions or classifications in 2D chemical space.

**FeatureCreature** relies heavily on our [BioCompoundML](http://pubs.acs.org/doi/abs/10.1021/acs.energyfuels.6b01952) work. 

>Whitmore, L. S., Davis, R. W., McCormick, R. L., Gladden, J. M., Simmons, B. A., George, A., & Hudson, C. M. (2016). BioCompoundML: a general biofuel property screening tool for biological molecules using Random Forest Classifiers. *Energy & Fuels, 30*(10), 8410-8418.

Parts of this work was presented at the 2016 NIPS Workshop on Interpretable Machine Learning in Complex Systems

>Whitmore, L.S., George, A. and Hudson, C.M., 2016. Mapping chemical performance on molecular structures using locally interpretable explanations. arXiv preprint arXiv:1611.07443. [arxiv](https://arxiv.org/abs/1611.07443)

## Getting started

Clone this github repo by running the FeatureCreature Jupyter Notebook (http://jupyter.org)

## Dependencies

Visualization functions require the Indigo Toolkit Python bindings (http://lifescience.opensource.epam.com/download/indigo/index.html).

Additionally **FeatureCreature** requires a scientific python environment, including Numpy, Scipy and scikit-learn. The easiest way to do this is to download and use a **conda** environment (https://www.continuum.io/downloads)