import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

AU = 1.496e11
x1 = -0.5*AU
x2 = 0.5*AU
y1 = y2 = vx1 = vx2 = t = 0
vy1 = -15000
vy2 = 15000
m1 = m2 = mSol = 1.989e30
G = 6.674e-11
year = 31557600

def func(t, y0, G, m1, m2): #function to pass ODE's into solve_ivp
    x1,y1,x2,y2,vx1,vy1,vx2,vy2 = y0 
    
    r = ((x2-x1)**2+(y2-y1)**2)**0.5 #distance between body 1 and 2
    
    dx1 = vx1
    dy1 = vy1
    dx2 = vx2
    dy2 = vy2
    dvx1 = (G * m2 * (x2-x1)) / (r**3)
    dvy1 = (G * m2 * (y2-y1)) / (r**3)
    dvx2 = (G * m1 * (x1-x2)) / (r**3)
    dvy2 = (G * m1 * (y1-y2)) / (r**3)
    return dx1, dy1, dx2, dy2, dvx1, dvy1, dvx2, dvy2

tmin = 0
tmax = int(1.577e8) #5 years in seconds
tvals = int(1e6) #data points
y0 = [x1,y1,x2,y2,vx1,vy1,vx2,vy2]
t = np.linspace(tmin,tmax,tvals,endpoint=False)
t_span = [tmin,tmax]

solutions = solve_ivp(func, t_span, y0,t_eval=t,args=(G,m1,m2),atol=1e-7,rtol=1e-7) #tolerances low due to high values

fig1 = plt.figure(figsize=(5,6.5))
ax1 = plt.axes([0.15,0.5,0.8,0.4])
ax1.plot(solutions.y[0]/AU,solutions.y[1]/AU,'r',label='Body 1')
ax1.plot(solutions.y[2]/AU,solutions.y[3]/AU,'k',label='Body 2')
ax1.set_xlabel('x [AU]')
ax1.set_ylabel('y [AU]')
ax1.set_xbound(-0.6,0.6)
ax1.set_ybound(-0.4,0.4)
ax1.legend(loc='upper left')

ax2 = plt.axes([0.15,0.15,0.8,0.25])
ax2.plot(t/year,solutions.y[0]/AU,'r')
ax2.plot(t/year,solutions.y[2]/AU,'k')
ax2.set_xlabel('t [yr]')
ax2.set_ylabel('x [AU]')


#Energy conservation check

#extract positions and velocities
x1, y1, x2, y2 = solutions.y[0], solutions.y[1], solutions.y[2], solutions.y[3]
vx1, vy1, vx2, vy2 = solutions.y[4], solutions.y[5], solutions.y[6], solutions.y[7]

#kinetic energies
K1 = 0.5 * m1 * (vx1**2 + vy1**2)
K2 = 0.5 * m2 * (vx2**2 + vy2**2)
K = K1 + K2

#potential energy
r = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
U = -G * m1 * m2 / r

#total energy
E = K + U
E0 = E[0]

#relative energy change delta E as a function of time
dE = (E - E0) / E0

# second run with different tolerances for comparison
solutions2 = solve_ivp(func, t_span, y0,t_eval=t,args=(G,m1,m2),atol=1e-6,rtol=1e-6)

#extract positions and velocities for second run
x12, y12, x22, y22 = solutions2.y[0], solutions2.y[1], solutions2.y[2], solutions2.y[3]
vx12, vy12, vx22, vy22 = solutions2.y[4], solutions2.y[5], solutions2.y[6], solutions2.y[7]

#kinetic energies for second run
K12 = 0.5 * m1 * (vx12**2 + vy12**2)
K22 = 0.5 * m2 * (vx22**2 + vy22**2)
K2 = K12 + K22

#potential energy for second run
r2 = np.sqrt((x22 - x12)**2 + (y22 - y12)**2)
U2 = -G * m1 * m2 / r2

#total energy for second run
E2 = K2 + U2
E02 = E2[0]

#relative energy change delta E as a function of time for second run
dE2 = (E2 - E02) / E02


# plot delta E
plt.figure(figsize=(5,4))
plt.plot(solutions.t / year, dE,label='tols=1e-7')
plt.plot(solutions2.t / year, dE2,label='tols=1e-6')
plt.xlabel('t [yr]')
plt.ylabel('ΔE')
plt.legend(loc='upper left')
plt.title('Relative change in total energy')
plt.grid(True)
plt.tight_layout()
plt.show()
