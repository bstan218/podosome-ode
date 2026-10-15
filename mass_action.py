import json
import argparse
import math
from typing import override, Any, Literal
from dataclasses import dataclass, asdict, fields, astuple
from collections.abc import Sequence

import numpy as np
import matplotlib.pyplot as plt


if __package__:
    from .ode import Equation, Model
else:
    from ode import Equation, Model

@dataclass
class ODEConstantSet(Sequence[float]):
    @override
    def __len__(self):
        return len(fields(self))
    @override
    def __getitem__(self, index):
        return astuple(self)[index]
    def __setitem__(self, index, value) :
        if isinstance(index, slice):
            return NotImplementedError
        key:str = list(asdict(self).keys())[index]
        self.__setattr__(key, value)

@dataclass
class CaConstants(ODEConstantSet):
    Ct:float
    Signal:float
    S0:float
    S1:float
    D0:float
    D1:float

@dataclass
class GaConstants(ODEConstantSet):
    Gt:float
    S0:float
    S1:float
    D0:float
    D1:float

@dataclass
class AConstants(ODEConstantSet):
    S0:float
    S1:float
    D0:float
    D1:float
    
@dataclass
class RConstants(ODEConstantSet):
    S0:float
    S1:float
    D0:float
    D1:float

@dataclass
class MConstants(ODEConstantSet):
    S0:float
    S1:float
    D0:float

@dataclass
class MAParameters:
    Ca_constants: CaConstants
    Ga_constants: GaConstants
    A_constants:  AConstants
    R_constants:  RConstants
    M_constants:  MConstants
    dt : float
    steps : int
    initial_variables : dict[str,float]

    def aslist(self, include_hyper:bool=False):
        if include_hyper:
            raise NotImplementedError
        
        return (list(astuple(self.Ca_constants)) +
                list(astuple(self.Ga_constants)) +
                list(astuple(self.A_constants)) +
                list(astuple(self.R_constants)) +
                list(astuple(self.M_constants)))
    
    def to_json(self, path:str):
        dictionary = asdict(self)
        with open(path, 'w') as f:
            json.dump(dictionary, f)

class MAPFactory:
    ##all hyperparameters should be in method signature. If too many then refactor.
    def __init__(self,
                 null_parameters:Literal['none','S0','D0','S0D0'] = 'none',
                 dt = 0.01,
                 steps = 5000,
                 initial_variables:None|dict[str,float] = None
                 ) -> None:
        if null_parameters != 'none':
            raise NotImplementedError
        self.null_parameters = null_parameters
        self.dt = dt
        self.steps = steps

        if not initial_variables:
            var_names = ['Ca', 'Ga', 'A', 'R', 'M']
            initial_variables = {v : 0 for v in var_names}

        self.initial_variables = initial_variables

    def from_json(self, path:str):
        with open(path, 'r') as f:
            parameters: dict[str, Any] = json.load(f)

            ca_constants = CaConstants(**parameters['Ca_constants'])
            ga_constants = GaConstants(**parameters['Ga_constants'])
            a_constants = AConstants(**parameters['A_constants'])
            r_constants = RConstants(**parameters['R_constants'])
            m_constants = MConstants(**parameters['M_constants'])
            try:
                dt:float =  parameters['dt']
            except KeyError:
                dt:float = self.dt
            try:
                steps:int =  parameters['steps']
            except KeyError:
                steps:int = self.steps
            try:
                initial_variables:dict[str,float] = parameters['initial_variables']
            except KeyError:
                initial_variables = self.initial_variables

            ma_parameters = MAParameters(
                    Ca_constants=ca_constants,
                    Ga_constants=ga_constants,
                    A_constants=a_constants,
                    R_constants=r_constants,
                    M_constants=m_constants,
                    dt=dt,
                    steps=steps,
                    initial_variables=initial_variables)
            return ma_parameters

    def from_list(self, values:list[float], includes_hyper:bool=False):
        if includes_hyper == True:
            raise NotImplementedError

        lenCa = len(fields(CaConstants))
        lenGa = len(fields(GaConstants))
        lenA = len(fields(AConstants))
        lenR = len(fields(RConstants))
        lenM = len(fields(MConstants))

        if len(values) != lenCa + lenGa + lenA + lenR + lenM:
            raise ValueError

        rangeCa = slice(0, lenCa)
        rangeGa = slice(lenCa, lenCa+lenGa)
        rangeA = slice(lenCa+lenGa, lenCa+lenGa+lenA)
        rangeR = slice(lenCa+lenGa+lenA, lenCa+lenGa+lenA+lenR)
        rangeM = slice(lenCa+lenGa+lenA+lenR, None)
        ca_constants = CaConstants(*values[rangeCa])
        ga_constants = GaConstants(*values[rangeGa])
        a_constants = AConstants(*values[rangeA])
        r_constants = RConstants(*values[rangeR])
        m_constants = MConstants(*values[rangeM])
        ma_parameters = MAParameters(
                Ca_constants=ca_constants,
                Ga_constants=ga_constants,
                A_constants=a_constants,
                R_constants=r_constants,
                M_constants=m_constants,
                dt=self.dt,
                steps=self.steps,
                initial_variables=self.initial_variables)
        return ma_parameters

         

    def generate(self,
                 distribution:Literal['uniform','exponential'] = 'exponential',
                 mean:float=1,
                 span:float=1
                 ): 
        rng = np.random.default_rng()
        if distribution == 'uniform':
            gen= lambda: rng.uniform(low=mean-span/2, high=mean+span/2)

        elif distribution == 'exponential':
            gen = lambda: rng.exponential(scale=1/mean)
 
        ca_constants = CaConstants(
                Ct = 1 + gen(),
                Signal = gen(),
                S0 = gen(),
                S1 = gen(),
                D0 = gen(),
                D1 = gen()
                )
        ga_constants = GaConstants(
                Gt = 1 + gen(),
                S0 = gen(),
                S1 = gen(),
                D0 = gen(),
                D1 = gen()
                )
        a_constants = AConstants(
                S0 = gen(),
                S1 = gen(),
                D0 = gen(),
                D1 = gen()
                )
        r_constants = RConstants(
                S0 = gen(),
                S1 = gen(),
                D0 = gen(),
                D1 = gen()
                )
        m_constants = MConstants(
                S0 = gen(),
                S1 = gen(),
                D0 = gen(),
                )
        ma_parameters = MAParameters(
                Ca_constants=ca_constants,
                Ga_constants=ga_constants,
                A_constants=a_constants,
                R_constants=r_constants,
                M_constants=m_constants,
                dt=self.dt,
                steps=self.steps,
                initial_variables=self.initial_variables)
        return ma_parameters


class MassAction(Model):
    def __init__(self, parameters: MAParameters | str | dict[str, Any] ):
        super().__init__(variable_names=['Ca', 'Ga', 'A', 'R', 'M'])
        #TODO: parse parameters based on variable type
        #will need to parse list for ea
        if isinstance(parameters, list):
            raise TypeError('unlike the Model class, MassAction takes parameters instead of variable names.')


        if isinstance(parameters, str):
            factory = MAPFactory()
            parameters = factory.from_json(parameters)

        if isinstance(parameters, dict):
            raise NotImplementedError('sorry no dictionaries')

        Ca_constants = asdict(parameters.Ca_constants)
        Ga_constants = asdict(parameters.Ga_constants)
        A_constants = asdict(parameters.A_constants)
        R_constants = asdict(parameters.R_constants)
        M_constants = asdict(parameters.M_constants)

        self.dt : float= parameters.dt
        self.steps : int = parameters.steps
        self.initial_variables : dict[str,float] = parameters.initial_variables


        Ca_variables = ['Ca', 'Ga']
        dCa_dt = Equation(variable_names=Ca_variables, constants=Ca_constants, calculation=self.calc_dCa_dt)

        Ga_variables = ['Ga','Ca','A']
        dGa_dt= Equation(variable_names=Ga_variables, constants=Ga_constants, calculation=self.calc_dGa_dt)

        A_variables = ['A', 'Ca', 'M']
        dA_dt = Equation(variable_names=A_variables, constants=A_constants, calculation=self.calc_dA_dt)

        R_variables = ['R', 'Ga', 'M']
        dR_dt = Equation(variable_names=R_variables, constants=R_constants, calculation=self.calc_dR_dt)

        M_variables = ['M', 'R']
        dM_dt = Equation(variable_names=M_variables, constants=M_constants, calculation=self.calc_dM_dt)

        self.set_equation('Ca', dCa_dt)
        self.set_equation('Ga', dGa_dt)
        self.set_equation('A', dA_dt)
        self.set_equation('R', dR_dt)
        self.set_equation('M', dM_dt)

        self.trajectory: None | dict[str, list[float]] = None

    @staticmethod
    def calc_dCa_dt(Ca, Ga, Ct, Signal, S0, S1, D0, D1):
        Ci = Ct - Ca
        synthesis = Signal * Ci * (S0 + S1*(Ca**2))
        degredation = Ca * (D0 + D1*Ga)
        return synthesis - degredation

    @staticmethod
    def calc_dGa_dt(Ga, Ca, A, Gt, S0, S1, D0, D1):
        Gi = Gt-Ga
        synthesis = Gi * (S0 + S1*Ca)
        degredation = Ga * (D0 + D1 * A)
        return synthesis - degredation

    @staticmethod
    def calc_dA_dt(A, Ca, M, S0, S1, D0, D1):
        synthesis = S0 + S1*Ca
        degredation = A * (D0 + D1*M)
        return synthesis - degredation

    @staticmethod
    def calc_dR_dt(R, Ga, M, S0, S1, D0, D1):
        synthesis = S0 + S1*Ga
        degredation = R * (D0 + D1*M)
        return synthesis - degredation

    @staticmethod
    def calc_dM_dt(M, R, S0, S1, D0):
        synthesis = S0 + S1*R
        degredation = D0*M
        return synthesis - degredation

    @override
    def generate_trajectory(self, 
                            initial_dependent_variables: dict[str,float] | None = None,
                            dt : float | None = None,
                            steps : int |None = None,
                            stop_at_steady_state : bool =True):
        if initial_dependent_variables == None:
            initial_dependent_variables : dict[str,float] = self.initial_variables
        if dt == None:
            dt : float = self.dt

        if steps == None:
            steps : int = self.steps

        self.trajectory =  super().generate_trajectory(initial_dependent_variables=initial_dependent_variables,
                                           dt=dt,
                                           steps=steps,
                                           stop_at_steady_state=stop_at_steady_state)
        return self.trajectory

    @property
    def contains_nan(self):
        if not self.trajectory:
            _ = self.generate_trajectory()
        if self.trajectory == None:
            raise ValueError
        for t in self.trajectory.values():
            for e in t:
                if math.isnan(e):
                    return True
        return False



#####################################################################################################
# Trajectory Analysis

def plot_trajectory(model:MassAction):
    trajectory = model.generate_trajectory()

    #fig, (ax1, ax2, ax3) = plt.subplots(1,3)
    fig, (ax1, ax2) = plt.subplots(1,2)
    #graph 1
    initial_variables = ['Ca', 'Ga', 'A', 'R', 'M']
    for var in initial_variables:
        ax1.plot(trajectory['t'], trajectory[var], label=var)

    derivatives = [f'd{v}dt' for v in initial_variables]
    for d in derivatives:
        ax2.plot(trajectory['t'], trajectory[d], label=d)

    #ax1.plot(trajectory['t'], trajectory['y'], label='dydt')
    ax1.set_title('value over time')
    ax1.set_xlabel('time')
    ax1.set_ylabel('value')
    ax1.legend()

    ax2.set_title('derivatives over time')
    ax2.set_xlabel('time')
    ax2.set_ylabel('ddt')
    ax2.legend()

    """
    ax3.text(
            0.1,
            0.9,
            f'dCa/dt = {}')
    """

    plt.show()

if __name__ == '__main__':
    DEFAULT = "./default.json"

    parser = argparse.ArgumentParser()

    _ = parser.add_argument(
            "-i",
            type=str,
            required=False
            )

    _ = parser.add_argument(
            "-o",
            type=str,
            required=False
            )

    args = parser.parse_args()

    if args.i == None:
        parameters = DEFAULT
    else:
        parameters = args.i

    model = MassAction(parameters = parameters)

    plot_trajectory(model)


"""
    if args.o:
        output = {
            'Ca_constants' : Ca_constants,
            'Ca_variables' : Ca_variables,
            'Ga_constants' : Ga_constants,
            'Ga_variables' : Ga_variables,
            'A_constants' : A_constants,
            'A_variables' : A_variables,
            'R_constants' : R_constants,
            'R_variables' : R_variables,
            'M_constants' : M_constants,
            'M_variables' : M_variables,
            'initial_variables' : initial_variables,
            'dt' : dt,
            'steps' : steps 
            }
        outf = os.path.join(os.getcwd(), args.o)
        with open(outf, 'w') as f:
            json.dump(output, f, indent=4)
"""


