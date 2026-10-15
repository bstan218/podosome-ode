from typing import Any, Callable, TypeVar
from functools import partial
import random
import sys
from copy import deepcopy

from deap import base, creator, tools
import numpy as np

from mass_action import MassAction, MAParameters, MAPFactory

factory = MAPFactory(steps=2000)

# create a type
# creator is a meta factory that allows runtime creation of classes
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness = creator.FitnessMax)

def create_individual(factory:MAPFactory):
    p = factory.generate()
    return creator.Individual(p.aslist())

# a container for operators
toolbox = base.Toolbox()


toolbox.register("individual", partial(create_individual,factory))
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

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
    

def pod_lifecycle_osc_fitness(factory:MAPFactory, individual:creator.Individual, signal_range=1, signal_step=.1, return_parameters=False):
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
        print(max_fitness)
        return max_fitness



toolbox.register("mate", tools.cxTwoPoint)
#TODO: add custom mutation logic to ensure parameters result in viable
def absGaus(individual): 
    c = deepcopy(individual)
    mutant = tools.mutGaussian(c, mu=0, sigma=5, indpb=0.1)[0]
    for i in range(len(mutant)):
        mutant[i] = abs(mutant[i])
    return mutant

toolbox.register("mutate", absGaus)
#toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.1)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", pod_lifecycle_osc_fitness, factory)

def isnegative(l:list[float]):
    return sum([e < 0 for e in l]) > 0


def fit_search(search_fitness=.9, outfile:str='search_fitness.json'):
    factory = MAPFactory()
    while True:
        p = factory.generate(mean=1)
        fitness, max_osc_p = pod_lifecycle_osc_fitness(factory, p.aslist(), return_parameters=True)
        print(fitness)
        if fitness > search_fitness:
            max_osc_p.to_json(outfile)
            break

#Individual must inherit from Sequence and have methods for MutableSequence except not del to be compatible with tools.mut* methods
def main():
    pop = toolbox.population(n=50)
    CXPB, MUTPB, NGEN = 0.5, 0.2, 16
    fittest = None
    best_fit = 0

    # Evaluate the entire population
    fitnesses = list(map(toolbox.evaluate, pop))
    print(fitnesses)
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = (fit,)
        if fit > best_fit:
            best_fit = fit
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

        # Evaluate the individuals with an invalid fitness
        # TODO: figure out when fitness.valid is set
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = list(map(toolbox.evaluate, invalid_ind))
        print(fitnesses)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = (fit,)
            if fit > best_fit:
                best_fit = fit
                fittest = ind

        # The population is entirely replaced by the offspring
        pop[:] = offspring

    return fittest, best_fit

if __name__ == '__main__':
    fittest, best_fit = main()
    print(best_fit)
    p = factory.from_list(fittest)
    p.to_json('fittest.json')

    """
    p = factory.from_json('fittest.json')
    model = MassAction(p)
    result = measure_oscillation(model)
    print(result)
    """
#    fit_search()


