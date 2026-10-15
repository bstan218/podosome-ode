from multiprocessing import Value
import matplotlib.pyplot as pyplot 
import numpy as np
from collections.abc import Callable
from typing import Literal, Any
from skimage.measure import find_contours

"dx/dt = S - (delta * x)"
def simple_dxdt(x) : 
    S = 1
    D = 1
    return S - (D * x)

def euler_eval(fx, xt, t:float, delta_t:float ):
    x = xt + delta_t * fx(xt)
    return x



class Equation:
    def __init__(self, variable_names: list[str], constants: dict[str, float], calculation:Callable[...,float], type:Literal['direct', 'ode']='ode'):
        self.type:str = type
        self.constants: dict[str, float] = constants
        self.variables: dict[str, float] = {var:0 for var in variable_names}
        self.calculation:Callable[...,float] = calculation
    
    def set_variable(self, name:str, value:float):
        if name not in self.variables:
            raise ValueError('name not in self.variables')

        self.variables[name] = value

    #must have parameters as keywords where key matches each constant or variable name
    def set_calculation(self, calculation:Callable[..., float]):
        self.calculation = calculation

    def calculate(self):
        #TODO: use inspect.signature(calculation) to ensure calculation matches parameters
        return self.calculation(**{**self.variables, **self.constants})

    def nullcline(self, var1:str, var2:str, start=0, stop=20, resolution=500) :
        if var1 not in self.variables or var2 not in self.variables:
            raise ValueError

        range = np.linspace(start,stop,resolution)
        X, Y = np.meshgrid(range, range)
        
        Z = self.calculation(**{var1:X, var2:Y, **self.constants})
        contours = find_contours(Z, 0)
        nullcline_points = []

        for contour in contours:
            rows, cols = contour.T

            x_points = np.interp(cols, np.arange(resolution), range)
            y_points = np.interp(rows, np.arange(resolution), range)

            points = np.column_stack((x_points, y_points))

            nullcline_points.append(points)
        return nullcline_points[0]
        


class Model:
    def __init__(self, variable_names:list[str]):
        self.n_dependent: int = len(variable_names)
        self.dependent_variables: dict[str, float] = {var:0 for var in variable_names}
        self.equations: dict[str, Equation]  = {}

    def set_variable(self, variable_name:str, value:float): 
        self.dependent_variables[variable_name] = value
        for ode in self.equations.values():
            if variable_name in ode.variables:
                ode.variables[variable_name] = value


    def set_equation(self, variable_name:str, equation:Equation):
        self.equations[variable_name] = equation 
        for variable, value in self.dependent_variables.items():
            if variable in equation.variables:
                value = self.dependent_variables[variable]
                equation.set_variable(variable, value)


    def eval(self, dt:float):
        new_values: dict[str, float] = {}
        derivative_trajectory_values:dict[str,float] = {}
        for var, value in list(self.dependent_variables.items()):
            equation:Equation = self.equations[var]
            calculation = equation.calculate()
            if equation.type == 'ode':
                derivative = calculation
                derivative_trajectory_values[var] = derivative
                new_value = value + dt * derivative
                new_values[var] = new_value

            elif equation.type =='direct':
                new_value = calculation
                new_values[var] = new_value

        for var, value in new_values.items():
            self.set_variable(var, value)

        return derivative_trajectory_values

    def get_derivative_keys(self):
        return [f'd{var}dt' for var in self.dependent_variables]

    def generate_trajectory(self, initial_dependent_variables:dict[str,float]={}, dt:float=0.1, steps:int=100, stop_at_steady_state:bool=False):
        if initial_dependent_variables == {}:
            raise ValueError


        for var, value in initial_dependent_variables.items():
            self.set_variable(var, value)
        t = 0
        tt:list[float] = []
        #trajectories is a dictionary of all the trajectories for each dependent variable in the model
        trajectories: dict[str, list[float]] = {var:[] for var,value in self.dependent_variables.items()}
        derivative_trajectories: dict[str, list[float]] = {var:[] for var in self.get_derivative_keys()}
        for _ in range(steps):
            tt.append(t)
            derivative_values = self.eval(dt)
            for var, trajectory in trajectories.items():
                trajectory.append(self.dependent_variables[var])
                derivative_trajectories[f'd{var}dt'].append(derivative_values[var])

            if stop_at_steady_state and sum([abs(dt) for dt in derivative_values.values()]) <= 0.05:
                    return {'t':tt, **trajectories, **derivative_trajectories}

            t += dt

        return {'t':tt, **trajectories, **derivative_trajectories}

    #2D only
    #Need to supply two ODEs, given by their variable keys. 
    def bifurcation_analysis(self, var1:str, var2:str):
        nullcline1 = self.equations[var1].nullcline(var1, var2)
        nullcline2 = self.equations[var2].nullcline(var1, var2)

        return nullcline1, nullcline2



