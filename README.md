# Exact Optimality of Communication-Privacy-Utility Tradeoffs in Distributed Mean Estimation
Implementation of Randomly Rotating Simplex Coding (RRSC) algorithm by the authors of the NeurIPS 2023 paper "Exact Optimality of Communication-Privacy-Utility Tradeoffs in Distributed Mean Estimation".

> [Exact Optimality of Communication-Privacy-Utility Tradeoffs in Distributed Mean Estimation](https://arxiv.org/pdf/2306.04924.pdf) <br/>
>[Berivan Isik](https://sites.google.com/view/berivanisik), [Wei-Ning Chen](https://web.stanford.edu/~wnchen), [Ayfer Ozgur](https://web.stanford.edu/~aozgur/), [Tsachy Weissman](https://web.stanford.edu/~tsachy/), [Albert No](http://albertno.hongik.ac.kr/) <br/>
> Conference on Neural Information Processing Systems (NeurIPS), 2023. <br/>

## Instructions

Set which parameter to sweep through with `vary`, number of runs to repeat the experiment with `num_itr`, list of dimensions $d$ with `d_list`, list of epsilons $\varepsilon$ with `eps_list` when calling `rrsc_comparison()` in `main.py`. 

To reproduce the plots in the paper, run `run.ipynb`. 

## References
If you find this work useful in your research, please consider citing our paper:
```
@inproceedings{
isik2023exact,
title={Exact Optimality of Communication-Privacy-Utility Tradeoffs in Distributed Mean Estimation},
author={Berivan Isik and Wei-Ning Chen and Ayfer Ozgur and Tsachy Weissman and Albert No},
booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
year={2023},
url={https://openreview.net/forum?id=7ETbK9lQd7}
}
```
