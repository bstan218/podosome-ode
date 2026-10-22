import random
from copy import deepcopy
from multiprocessing import Pool
from functools import partial

import numpy as np
import pandas as pd
from pandas import DataFrame
from deap import base, creator, tools

from mass_action import MassAction, MAParameters, MAPFactory


### Set Globals ###
CXPB, MUTPB, NGEN, NIND = 0.5, 0.2, 50, 200
OSC_THRESHOLD = 0.9

FACTORY = MAPFactory(steps=20000, dt=0.001)
POOL = Pool()

####################### Define Fitness Function ##############################

def measure_oscillation(model:MassAction, oscillation_threshold_magnitude:float=0.1):
    osc_scores = [] 
    try:
        trajectories = model.generate_trajectory()
    except OverflowError:
        return -1
    if model.contains_nan:
        return -1

    derivative_trajectories = {key : trajectories[key] for key in model.get_derivative_keys()}
    
    for derivative, t in derivative_trajectories.items():
        oscillations = 0

        is_positive = t[0] >= 0
        for i in range(len(t)):
            step = t[i]
            if is_positive == False and step > oscillation_threshold_magnitude:
                oscillations += 1
                is_positive = True
            if is_positive == True and step < -oscillation_threshold_magnitude:
                oscillations += 1
                is_positive = False

        #Can vary the value k to make score more or less sensitive sensitive to frequent oscillations
        trajectory_fitness = oscillations / (1 + oscillations)
        osc_scores.append(trajectory_fitness)

    mean_osc_score = sum(osc_scores) / len(osc_scores)
    return mean_osc_score
    

def pod_lifecycle_osc_fitness(individual,
                              factory:MAPFactory=FACTORY, 
                              signal_range=1, 
                              signal_step=.05, 
                              ):
    parameters = factory.from_list(individual)
    #print(parameters)
    init_signal = parameters.Ca_constants.Signal
    max_fitness = 0
    optimal_individual = individual

    #Iterate through range of signals around given initial signal
    for signal in np.arange(init_signal - signal_range/2,
                            init_signal + signal_range/2,
                            signal_step):
        if signal < 0:
            continue
        parameters.Ca_constants.Signal = signal
        model = MassAction(parameters)
        fitness = measure_oscillation(model)
        if fitness == -1:
            return 0
        if fitness > max_fitness:
            max_fitness = fitness
            optimal_individual[:] = parameters.aslist()
            
    return max_fitness, optimal_individual




##################### Set Up DEAP Framework #####################

# creator is a meta FACTORY that allows runtime creation of classes
creator.create("FitnessMax", base.Fitness, weights=(1.0,))

#Individual must inherit from Sequence and have methods for MutableSequence except not del to be compatible with tools.mut* methods
creator.create("Individual", list, fitness = creator.FitnessMax)

#TODO: refactor so it's an obj method, doesn't belong here at all
def normalize(param:MAParameters):
    #Cap Signal
    param.Ca_constants.Signal = min(4, abs(param.Ca_constants.Signal))
    #Cap Ct
    param.Ca_constants.Ct = min(4, abs(param.Ca_constants.Signal))
    return param

def create_individual(factory:MAPFactory=FACTORY):
    p = factory.generate()
    p.Ca_constants.Ct += 1
    p.Ga_constants.Gt += 1
    p = normalize(p)
    return creator.Individual(p.aslist())

# a container for operators
toolbox = base.Toolbox()

toolbox.register("individual", partial(create_individual,FACTORY))
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("mate", tools.cxTwoPoint)

def absGaus(individual): 
    c = deepcopy(individual)
    mutant = tools.mutGaussian(c, mu=0, sigma=5, indpb=0.1)[0]
    #normalize
    p = normalize(FACTORY.from_list(mutant))
    mutant[:] = p.aslist()
    for i in range(len(mutant)):
        mutant[i] = abs(mutant[i])
    return mutant

toolbox.register("mutate", absGaus)
#toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.1)
toolbox.register("select", tools.selTournament, tournsize=3)



########################## Evolutionary Algorithm ######################

def evolutionary_algorithm(return_each_gen_best:bool = False):
    fittest_individual = None
    highest_fitness_value = 0

    record_setters = DataFrame(index=['fitness','ngen','parameters'])


    # Evaluate the entire population
    #TODO: Fix this so it uses POOL
    pop = toolbox.population(n=NIND)
    for i in range(len(pop)):
        individual = pop[i]
        fitness, optimal_signal_individual= pod_lifecycle_osc_fitness(individual)
        individual = optimal_signal_individual



    fitnesses = list(POOL.map(toolbox.evaluate, pop))
    print(fitnesses)
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = (fit,)
        if fit > highest_fitness_value:
            highest_fitness_value = fit
            fittest_individual = ind

    for g in range(NGEN):
        generation = g+1
        print('NGEN:', generation)
        # Select the next generation individuals
        offspring = toolbox.select(pop, len(pop))
        # Clone the selected individuals
        offspring = list(map(toolbox.clone, offspring))

        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        #at this point, any individuals with invalid fitness are just children that haven't been evaluated yet
        for i in range(len(offspring)):
            if random.random() < MUTPB:
                mutant = toolbox.mutate(offspring[i])
                del mutant.fitness.values
                offspring[i] = mutant
#TODO: still have some mutants slipping through constraints. probably due to initially generated parameters getting crossed. Need to prevent.
        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = list(POOL.map(toolbox.evaluate, invalid_ind))

        #TODO: change so printing variance instead
        print([round(e, 2) for e in fitnesses])

        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = (fit,)
            if fit > highest_fitness_value:
                highest_fitness_value = fit
                fittest_individual = ind

                best_of_gen_params.append(fittest_individual)
                best_of_gen_fitness.append(highest_fitness_value)

        ###save record
        record = [highest_fitness_value,
                  generation,
                  fittest_individual[:]]
        try:
            i = record_setters[-1] + 1
        except IndexError:
            i = 0

        record_setters[i] = record

        # The population is entirely replaced by the offspring
        pop[:] = offspring

        if highest_fitness_value > OSC_THRESHOLD:
            if return_each_gen_best:
                return record_setters
            else:
                return fittest_individual, highest_fitness_value

        print(highest_fitness_value)

    if return_each_gen_best:
        return record_setters
    else:
        return fittest_individual, highest_fitness_value


if __name__ == '__main__':
    fittest_individual, highest_fitness_value = evolutionary_algorithm()
    highest_fitness_value, fittest_true_signal = pod_lifecycle_osc_fitness(FACTORY, fittest_individual,return_parameters=True)
    print(highest_fitness_value)
    fittest_true_signal.to_json('fittest_individual.json')

    """
    p = FACTORY.from_json('fittest_individual.json')
    highest_fitness_value, fittest_true_signal = pod_lifecycle_osc_fitness(FACTORY, p.aslist(),return_parameters=True)
    print(fittest_true_signal.Ca_constants)
    """


