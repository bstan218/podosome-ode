from ode import Equation, Model
from matplotlib import pyplot
import numpy as np

def G(u, v, J, K):
    numerator = 2 * u * K
    denominator = v - u + v*J + u*K + np.sqrt((v-u+v*J+u*K)**2 - 4*(v-u)*u*K)
    return numerator / denominator

def sig_dxdt(x):
    k0 = 0.4
    k1 = 0.01
    k2 = 1
    k3 = 1
    k4 = 0.2
    j3 = 0.05
    j4 = 0.05
    S=8
    D=1

    E_p_R = G(k3*x, k4, j3, j4)
    return k0*E_p_R + k1*S - k2*D*x

dependent_variable_names = ['x','y']

x_constants = {
        'S0' : 4.,
        'S1' : 1.,
        'S2' : 1.,
        'S3' : 1.
        }

#K < kmax/S2
#H(x) = kmax/2 when x=K
def H(x:float, kmax:float, K:float, n:float=1) -> float:
    return (kmax * x**n) / (K**n + x**n)

def cyc_dxdt(x=0, y=0, S0=0, S1=0, S2=0, S3=0) -> float:
    return S0 + S1*H(x, 3, 1.0, 10) - x*(S2 + S3*y)
def cyc_dxdt_stable_deg(x=0, S0=0, S1=0, S2=0) -> float:
    return S0 + S1*H(x, 7, 2.0, 4) - S2*x 

#not useful, needs hill function to oscillate
def simple_dxdt(x=0, y=0, S1=0, S2=0, S3=0) -> float:
    kmax=1000
    return 10 + S1*kmax - x*(S2 + S3*y)

x_ode = Equation(dependent_variable_names, x_constants, cyc_dxdt, type='ode')

y_constants = {
        'S1' : 0.00,
        'S2' : 1.00,
        'S3' : 2/9        
        }

def dydt(x=0, y=0, S1=0, S2=0, S3=0) -> float:
    return S1 + S2*x - S3*y

y_ode = Equation(dependent_variable_names, y_constants, dydt, type='ode')

model = Model(dependent_variable_names)
model.set_equation('x', x_ode)
model.set_equation('y', y_ode)

trajectory = model.generate_trajectory({'x' : 1, 'y' : 4.}, dt=0.001, steps=50000)
xx_key = 't'
yy_key = 'dxdt'

fig, (ax1, ax2, ax3, ax4) = pyplot.subplots(1,4)
#graph 1
ax1.plot(trajectory['t'], trajectory['dxdt'], label='dxdt')
ax1.plot(trajectory['t'], trajectory['dydt'], label='dydt')
#ax1.plot(trajectory['t'], trajectory['y'], label='dydt')
ax1.set_title('derivative over time')
ax1.set_xlabel('time')
ax2.set_ylabel('d/dt')
ax1.legend()

ax2.plot(trajectory['t'], trajectory['x'], label='x')
ax2.plot(trajectory['t'], trajectory['y'], label='y')
ax2.set_title('values over time')
ax2.set_xlabel('time')
ax2.set_ylabel('value')
ax2.legend()

ax3.plot(trajectory['x'], trajectory['y'])
ax3.set_title('x vs y')
ax3.set_xlabel('x')
ax3.set_ylabel('y')

nullcline1, nullcline2 = model.bifurcation_analysis('y', 'x')
ax4.plot(nullcline1[:, 0], nullcline1[:, 1])
ax4.plot(nullcline2[:, 0], nullcline2[:, 1])
ax4.set_title('Bifurcation Analysis')
#pyplot.xlabel(xx_key)
#pyplot.ylabel(yy_key)
#pyplot.tight_layout()
pyplot.show()

print('completed')

