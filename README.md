<div align="center">
  <h1>📙 NLP and Data Viz Pipeline with GloVe, HDBSCAN, and t-SNE</h1>
<p align="center">
<a href="https://img.shields.io/badge/python-3.7.10-blue.svg">
    <img src="https://img.shields.io/badge/python-3.7.10-blue.svg" alt="Python 3.7.10" />
</a>
<a href="https://img.shields.io/badge/code%20style-black-000000.svg">
    <img src="https://img.shields.io/badge/code%20style-black-000000.svg" alt="Code style: black" >
</a>
</div>

This project uses NLP and unsupervised learning to visualize the text of the canonical machine learning book [_The Elements of Statistical Learning_](https://web.stanford.edu/~hastie/Papers/ESLII.pdf).

<div>
<p align="center"><a href='https://connor-mccarthy.github.io/nlp-visualization-of-statistical-learning-book/figure.html'>Click here</a> to explore the data yourself.</p>

<div>

The pipeline represents the text of the book with GloVe embeddings, clusters it with HDBSCAN, and visualizes it with t-SNE.


## Pipeline steps:
1) Make HTTP request to obtain PDF
2) Convert single PDF file to array of PNG files
3) Use OCR to convert image to text
4) Apply rule-based pipeline to extract n-grams of theoretically unlimited length n if rules are met for all tokens in n-gram
5) Map tokens to GloVe embeddings (averaging where n-gram has n > 1)
6) Normalize vector embeddings
7) Cluster using HDBSCAN
8) Reduce dimensionality with PCA from dimensions (300,) --> (50,) for computational efficiency in subsequent t-SNE step
9) Reduce dimensionality further with t-SNE from dimensions (50,) --> (3,)
10) Plot vectors
