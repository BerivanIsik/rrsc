# Exact Optimality of Communication-Privacy-Utility Tradeoffs in Distributed Mean Estimation
Implementation of Randomly Rotating Simplex Coding (RRSC) algorithm by the authors of the NeurIPS 2023 paper "Exact Optimality of Communication-Privacy-Utility Tradeoffs in Distributed Mean Estimation".

> [Exact Optimality of Communication-Privacy-Utility Tradeoffs in Distributed Mean Estimation](https://arxiv.org/pdf/2306.04924.pdf) <br/>
>[Berivan Isik](https://sites.google.com/view/berivanisik), [Wei-Ning Chen](https://web.stanford.edu/~wnchen), [Ayfer Ozgur](https://web.stanford.edu/~aozgur/), [Tsachy Weissman](https://web.stanford.edu/~tsachy/), [Albert No](http://albertno.hongik.ac.kr/) <br/>
> Conference on Neural Information Processing Systems (NeurIPS), 2023. <br/>

## Instructions

Set which parameter to sweep through via `vary`, number of runs to repeat the experiment with `num_itr`, list of dimensions $d$ to repeat with `d_list`, list of epsilon $\varepsilon$ to repeat with `eps_list` when calling `rrsc_comparison()` in `main.py`. 

To reproduce the plots in the paper, run `run.ipynb`. 

## References
If you find this work useful in your research, please consider citing our paper:
```
@article{isik2023exact,
  title={Exact Optimality of Communication-Privacy-Utility Tradeoffs in Distributed Mean Estimation},
  author={Isik, Berivan and Chen, Wei-Ning and Ozgur, Ayfer and Weissman, Tsachy and No, Albert},
  journal={arXiv preprint arXiv:2306.04924},
  year={2023}
}
```
