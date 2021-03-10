# Language Clustering of Statistical Learning Books
[![Python 3.7.10](https://img.shields.io/badge/python-3.7.10-blue.svg)](https://www.python.org/downloads/release/python-360/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)


[_The Elements of Statistical Learning_](https://web.stanford.edu/~hastie/Papers/ESLII.pdf) is the more technical treatment of similar topics.



This repository contains code and figures related to an statistical investigation of both books using the HDBSCAN clustering algorithm.


Pipeline steps:
1) Make HTTP request to get data
2) Convert PDF to array of PNGs
3) Use Pytesseract OCR to convert image to text
4) Rule based pipeline to extract n-grams of theoretically unlimited length n if rules are met
5) Map tokens to GloVe embeddings, averaging across spans
6) Norm embeddings
7) Cluster using HDBSCAN
8) Reduce dimensionality with PCA from 300dims --> 50dims
9) Reduce dimensionality with t-SNE from 50dims --> 3dims