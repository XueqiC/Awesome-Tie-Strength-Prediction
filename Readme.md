<h1 align="center"> BTS: A Comprehensive Benchmark for Tie Strength Prediction </h1>

  <p align="center">
    <a href='https://arxiv.org/abs/2410.19214'>
      <img src='https://img.shields.io/badge/Paper-PDF-green?style=flat&logo=arXiv&logoColor=green' alt='arXiv PDF'> </a>
    <a href='https://github.com/XueqiC/Awesome-Tie-Strength-Prediction'>
      <img src='https://img.shields.io/badge/Project-Page-blue?style=flat&logo=Google%20chrome&logoColor=blue' alt='Project Page'> </a>
    <a href='https://github.com/XueqiC/Awesome-Tie-Strength-Prediction/blob/main/LICENSE'>
      <img src='https://img.shields.io/badge/License-MIT-green.svg' alt='MIT License'> </a>
  </p>

## Introduction
This is the code repository for the paper [BTS: A Comprehensive Benchmark for Tie Strength Prediction].

The rapid growth of online social networks has highlighted the importance of understanding tie strength in user relationships. Despite prior efforts to assess tie strength (TS) in networks, the typically inherent lack of ground-truth labels, varying researcher perspectives, and limited performance of existing models hinder effective prediction for real-world applications.
To address this gap, we introduce $\textbf{BTS}$, a comprehensive \textbf{B}enchmark for \textbf{T}ie \textbf{S}trength prediction, aiming to establish a standardized foundation for evaluating and advancing TS prediction methodologies. Specifically, our contributions are: \textbf{TS Pseudo-Label Techniques} - 
we categorize TS into seven standardized pseduo-labeling techniques based on prior literature; \textbf{TS Dataset Collection} – we present a well-curated collection of three social networks and perform data analysis by investigating the class distributions and correlations across the generated pseudo-labels; \textbf{TS Pseudo-Label Evaluation Framework} - we propose a standardized framework to evaluate the pseudo-label quality from the perspective of tie resilience; \textbf{Benchmarking} - we evaluate existing tie strength prediction model performance using the BTS dataset collection, exploring the effects of different experiment settings, models, and evaluation criteria on the results. Furthermore, we derive key insights to enhance existing methods and shed light on promising directions for future research in this domain.

## Run the code
- To run the heurstic random forest experiments, please check the code in 'rf.sh'.
- To run the MLP experiments, please check the code in 'mlp.sh', where MLP is with edge features and MLP2 is without edge features.
- To run the GCN experiments, please check the code in 'gcn.sh'.
- To run the GTN experiments, please check the code in 'tran.sh'.
- `greedy.sh' is for the STC-greedy experiments, with the prediction results generated from this [STC-Greedy matlab code](https://bitbucket.org/ghentdatascience/stc-code-public/src/master/). 

## License
MIT License

## Contact 
Please feel free to email me (xueqi.cheng [AT] vanderbilt.edu) for any question about this work.

