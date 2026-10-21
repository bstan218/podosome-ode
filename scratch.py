from multiprocessing import Pool

from fit import pod_lifecycle_osc_fitness
from mass_action import MassAction, MAParameters, MAPFactory


factory = MAPFactory()

p = factory.from_json('nice.json').aslist()

fitness = pod_lifecycle_osc_fitness(factory,
                                    individual=p,
                                    exit_on_success=False)


print(fitness)
