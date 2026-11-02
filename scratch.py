from multiprocessing import Pool

from fit import pod_lifecycle_osc_fitness
from mass_action import MassAction, MAParameters, MAPFactory, plot_trajectory


factory = MAPFactory(dt=0.001)

l = [2.649999999999989, 1.9999999999999891, 0.8228265627203881, 9.877901081157498, 3.417111609069584, 7.0017218456479995, 12.618740987357782, 0.06403109470707606, 1.5849134533113847, 1.2647880455762903, 1.6403170974130308, 5.638892745406102, 0.37278332377270224, 0.07315852372708227, 1.0173664470203183, 0.7067036437101383, 2.4322200753993277, 0.43635380880432645, 0.4832318309828007, 1.8462044763126817, 3.5828789773728227, 1.4034868923854338]
p = factory.from_list(l)
#p = factory.from_json('nice.json').aslist()
model = MassAction(p)
plot_trajectory(model)


"""
fitness = pod_lifecycle_osc_fitness(factory,
                                    individual=p,
                                    exit_on_success=False)
print(fitness)
"""
