import pandas as pd
from pandas import DataFrame

from fit import evolutionary_algorithm, pod_lifecycle_osc_fitness


nits = 10
OUTFILE = "ea_analysis.csv"

if __name__ == '__main__':

    for i in range(nits):
        records:DataFrame = evolutionary_algorithm(
                return_each_gen_best=True)
        records.to_csv(OUTFILE, mode='a')
        print(f"iteration {i} completed")
        
