## Scripts for benchmarking SSAT instances by Justicia

A sample command for generating benchmarks is the following.
```
python script.py --encoding Learn Learn-dependency Learn-correlation --dataset adult
```


### Details
- Benchmarks are generated inside [benchmark](benchmark/) folder in the current directory.
- For effeciency in benchmarking, we have turned of the SSAT solver using a Boolean variable `do_not_solve` inside `init` method in [justicia\ssat_wrap.py](../justicia/ssat_wrap.py). If you want to collect benchmarks along with solving them, set `do_not_solve=False` in Line 19 in [justicia\ssat_wrap.py](../justicia/ssat_wrap.py).
- [script.py](script.py) contains code for generating benchmarks. Details can be found by running the command.
    ```
    python script.py -h
    ```
    - possible datasets are {titanic, adult, ricci, german, bank, compas, communities}. 
    - Each dataset takes a config (default is 0), which is an integer between 0 and 6 (config may change based on a dataset). More details can be found in [dataset-scripts](../data/objects/). 
    - To generate ER-SSAT instances, possible encodings are {Learn, Learn-dependency, Learn-correlation}. 
    - To generate RE-SSAT instances, possible encodings are {Enum, Enum-dependency, Enum-correlation}

