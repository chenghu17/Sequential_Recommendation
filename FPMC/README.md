# FPMC

FPMC[1] implementation for python3 with Numba.

## Dependencies
- Python3
- Numpy
- Numba >= 25.0

## How to run
Just type

    python3 run.py data/

If Numba is not installed, implementation in generic python will be used. Numba version is 10x faster than generic version.

## Notes
This implemtation is the same as original paper except:
- Number of negative sample: default is 10
- Use one basket to predict one item. That is, size of "next basket - i" is 1.

## Data format
Please refer to data/idxseq.txt.


The format is:

    [user index] [item index] ... [item index]
The last one item is regarded as next item (next basket), and is what our FPMC will predict.


## Reference

-  [1] Rendle, S., Freudenthaler, C., & Schmidt-Thieme, L. (2010). Factorizing personalized Markov chains for next-basket recommendation. Proceedings of the 19th International Conference on World Wide Web - WWW ’10, 811. http://doi.org/10.1145/1772690.1772773
