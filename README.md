# Justicia
This is the implementation of our [AAAI-2021 paper](https://arxiv.org/pdf/2009.06516.pdf) where we have proposed a SSAT-based approach to formally verify fairness in machine learning.

## Install
- Install the python library
`pip install justicia`
- Install other python dependencies
`pip install -r requirements.txt`

### Other dependencies
- [SSAT solver](https://github.com/nianzelee/ssatABC). Checkout to compatible version.
    
    ```
    git clone https://github.com/NTU-ALComLab/ssatABC.git
    cd ssatABC
    git checkout 91a93a57c08812e3fe24aabd71219b744d2355ad
    ```
- [PySAT](https://github.com/pysathq/pysat)
- [Notears](https://github.com/xunzheng/notears)

## Documentation
Python tutorials are available in [doc](doc/).

## Citations
Please cite the following paper.
```
@inproceedings{ghosh2020justicia,
author={Ghosh, Bishwamittra and Basu, Debabrota and Meel, Kuldeep S.},
title={Justicia: A Stochastic {SAT} Approach to Formally Verify Fairness},
booktitle={Proceedings of AAAI},
month={2},
year={2021},
}
```

### Issues, questions, bugs, etc.
Please click on "issues" at the top and [create a new issue](https://github.com/meelgroup/justicia/issues). All issues are responded to promptly.
