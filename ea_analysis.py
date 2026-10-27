from pathlib import Path

import pandas as pd
from pandas import DataFrame

from fit import evolutionary_algorithm, pod_lifecycle_osc_fitness


nits = 1000
OUTFILE = "iterations.csv"

if __name__ == '__main__':

    for i in range(nits):
        records:DataFrame = evolutionary_algorithm(
                return_each_gen_best=True)
        if i == 0 and not Path(OUTFILE).exists(): 
            records.T.to_csv(OUTFILE, header=True, mode='a')
        else:
            records.T.to_csv(OUTFILE, header=False, mode='a')

        print(f"iteration {i+1} completed")
        

