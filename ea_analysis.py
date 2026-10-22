import pandas as pd

from fit import evolutionary_algorithm, pod_lifecycle_osc_fitness


nits = 10
OUTFILE = "ea_analysis.csv"

if __name__ == '__main__':
    df = pd.DataFrame()

    for i in range(nits):
        best_of_gen_params, best_of_gen_fitness = evolutionary_algorithm(
                return_all_generations=True)

        rows = []
        for p, f in zip(best_of_gen_params, best_of_gen_fitness):
           row = p + [f] 
           rows.append(row)





            

        #write to dataframe
        
    















