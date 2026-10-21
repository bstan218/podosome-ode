from typing import Any, Callable, TypeVar
from functools import partial
import random
import sys
from copy import deepcopy
from multiprocessing import Pool

from deap import base, creator, tools
import numpy as np

from mass_action import MassAction, MAParameters, MAPFactory


factory = MAPFactory(steps=20000, dt=0.001)
CXPB, MUTPB, NGEN, NIND = 0.5, 0.2, 50, 200




####################### Define Fitness Function ##############################

def measure_oscillation(model:MassAction):
    fitnesses = [] 
    try:
        trajectories = model.generate_trajectory()
    except OverflowError:
        return -1
    if model.contains_nan:
        return -1

    derivative_trajectories = {key : trajectories[key] for key in model.get_derivative_keys()}
    
    threshold_magnitude = .1
    k= 1

    for derivative, t in derivative_trajectories.items():
        oscillations = 0

        is_positive = t[0] >= 0
        for i in range(len(t)):
            step = t[i]
            if is_positive == False and step > threshold_magnitude:
                oscillations += 1
                is_positive = True
            if is_positive == True and step < -threshold_magnitude:
                oscillations += 1
                is_positive = False

        trajectory_fitness = oscillations / (k + oscillations)
        fitnesses.append(trajectory_fitness)

    result = sum(fitnesses) / len(fitnesses)
    return result
    

def pod_lifecycle_osc_fitness(factory:MAPFactory, individual, signal_range=1, signal_step=.05, return_parameters=False):
    parameters = factory.from_list(individual)
    #print(parameters)
    init_signal = parameters.Ca_constants.Signal
    max_fitness = 0
    max_parameters = None
    for signal in np.arange(init_signal - signal_range/2, init_signal + signal_range/2, signal_step):
        if signal < 0:
            continue
        parameters.Ca_constants.Signal = signal
        model = MassAction(parameters)
        fitness = measure_oscillation(model)
        if fitness == -1:
            return 0
        if fitness > max_fitness:
            max_fitness = fitness
            max_parameters = deepcopy(parameters)
    if return_parameters:
        return (max_fitness, max_parameters)
    else:
        #print(max_fitness)
        return max_fitness




##################### Set Up DEAP Framework #####################

# creator is a meta factory that allows runtime creation of classes
creator.create("FitnessMax", base.Fitness, weights=(1.0,))

#Individual must inherit from Sequence and have methods for MutableSequence except not del to be compatible with tools.mut* methods
creator.create("Individual", list, fitness = creator.FitnessMax)

def normalize(param:MAParameters):
    #Cap Signal
    param.Ca_constants.Signal = min(4, abs(param.Ca_constants.Signal))
    #Cap Ct
    param.Ca_constants.Ct = min(4, abs(param.Ca_constants.Signal))
    return param

def create_individual(factory:MAPFactory):
    p = factory.generate()
    p.Ca_constants.Ct += 1
    p.Ga_constants.Gt += 1
    p = normalize(p)
    return creator.Individual(p.aslist())

# a container for operators
toolbox = base.Toolbox()

toolbox.register("individual", partial(create_individual,factory))
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("mate", tools.cxTwoPoint)

def absGaus(individual): 
    c = deepcopy(individual)
    mutant = tools.mutGaussian(c, mu=0, sigma=5, indpb=0.1)[0]
    #normalize
    p = normalize(factory.from_list(mutant))
    mutant[:] = p.aslist()
    for i in range(len(mutant)):
        mutant[i] = abs(mutant[i])
    return mutant

toolbox.register("mutate", absGaus)
#toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.1)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", pod_lifecycle_osc_fitness, factory)



########################## Evolutionary Algorithm ######################

def evolutionary_algorithm(return_all_generations:bool = False):
    fittest = None
    highest_fitness_value = 0
    pool = Pool()

    best_of_gen_params = []
    best_of_gen_fitness = []

    # Evaluate the entire population
    pop = toolbox.population(n=NIND)
    fitnesses = list(pool.map(toolbox.evaluate, pop))
    print(fitnesses)
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = (fit,)
        if fit > highest_fitness_value:
            highest_fitness_value = fit
            fittest = ind

    for g in range(NGEN):
        print('NGEN:', g)
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
        fitnesses = list(pool.map(toolbox.evaluate, invalid_ind))

        #TODO: change so printing variance instead
        print([round(e, 2) for e in fitnesses])

        new_best = False
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = (fit,)
            if fit > highest_fitness_value:
                new_best = True
                highest_fitness_value = fit
                fittest = ind

        if return_all_generations and new_best:
            best_of_gen_params.append(fittest.aslist())
            best_of_gen_fitness.append(highest_fitness_value)

        # The population is entirely replaced by the offspring
        pop[:] = offspring
        p = factory.from_list(fittest)
        p.to_json('fittest.json')
        if highest_fitness_value > 0.9:
            if return_all_generations:
                return best_of_gen_params, best_of_gen_fitness
            else:
                return fittest, highest_fitness_value

        print(highest_fitness_value)

    if return_all_generations:
        return best_of_gen_params, best_of_gen_fitness
    else:
        return fittest, highest_fitness_value


if __name__ == '__main__':
    fittest, highest_fitness_value = evolutionary_algorithm()
    highest_fitness_value, fittest_true_signal = pod_lifecycle_osc_fitness(factory, fittest,return_parameters=True)
    print(highest_fitness_value)
    fittest_true_signal.to_json('fittest.json')

    """
    p = factory.from_json('fittest.json')
    highest_fitness_value, fittest_true_signal = pod_lifecycle_osc_fitness(factory, p.aslist(),return_parameters=True)
    print(fittest_true_signal.Ca_constants)
    """


