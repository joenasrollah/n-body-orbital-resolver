import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

####
plotAllBurrau = False ####Set to true to plot all 7 decades, to match the 1967 paper exactly.
####

#Inits
AU = 1
x1 = 1
x2 = -2
x3 = 1
y1 = 3
y2 = -1
y3 = -1
z1 = z2 = z3 = 0
vx1 = vx2 = vx3 = t = vy1 = vy2 = vy3 = vz1 = vz2 = vz3 = 0
m1 = 3
m2 = 4
m3 = 5
G = 1
yearinsecs = 1

def func(t, y0, G, m1, m2): #function to pass ODE's into solve_ivp
    x1,y1,z1,x2,y2,z2,x3,y3,z3,vx1,vy1,vz1,vx2,vy2,vz2,vx3,vy3,vz3 = y0 
    
    r12 = ((x2-x1)**2+(y2-y1)**2+(z2-z1)**2)**0.5 #distance between body 1 and 2
    r13 = ((x3-x1)**2+(y3-y1)**2+(z3-z1)**2)**0.5 #distance between body 1 and 3
    r23 = ((x3-x2)**2+(y3-y2)**2+(z3-z2)**2)**0.5 #distance between body 2 and 3
    
    dx1 = vx1
    dy1 = vy1
    dz1 = vz1
    dx2 = vx2
    dy2 = vy2
    dz2 = vz2
    dx3 = vx3
    dy3 = vy3
    dz3 = vz3
#structure: force between itself and 1up + force between itself and 2up
    dvx1 = (G * m2 * (x2-x1)) / (r12**3) + (G * m3 * (x3-x1)) / (r13**3) #x1
    dvy1 = (G * m2 * (y2-y1)) / (r12**3) + (G * m3 * (y3-y1)) / (r13**3) #y1
    dvz1 = (G * m2 * (z2-z1)) / (r12**3) + (G * m3 * (z3-z1)) / (r13**3) #z1
    dvx2 = (G * m3 * (x3-x2)) / (r23**3) + (G * m1 * (x1-x2)) / (r12**3) #x2
    dvy2 = (G * m3 * (y3-y2)) / (r23**3) + (G * m1 * (y1-y2)) / (r12**3) #y2
    dvz2 = (G * m3 * (z3-z2)) / (r23**3) + (G * m1 * (z1-z2)) / (r12**3) #z2
    dvx3 = (G * m1 * (x1-x3)) / (r13**3) + (G * m2 * (x2-x3)) / (r23**3) #x3
    dvy3 = (G * m1 * (y1-y3)) / (r13**3) + (G * m2 * (y2-y3)) / (r23**3) #y3
    dvz3 = (G * m1 * (z1-z3)) / (r13**3) + (G * m2 * (z2-z3)) / (r23**3) #z3
    return dx1, dy1, dz1, dx2, dy2, dz2, dx3, dy3, dz3, dvx1, dvy1, dvz1, dvx2, dvy2, dvz2, dvx3, dvy3, dvz3

tmin = 0
years = 70 #years
tmax = int(years*yearinsecs) # years in seconds
tvals = int(1e6) #data points
y0 = [x1,y1,z1,x2,y2,z2,x3,y3,z3,vx1,vy1,vz1,vx2,vy2,vz2,vx3,vy3,vz3]
t = np.linspace(tmin,tmax,tvals,endpoint=False)
t_span = [tmin,tmax]

solutions = solve_ivp(func, t_span, y0,t_eval=t,args=(G,m1,m2),atol=1e-12,rtol=1e-12) #tolerances low due to high values

#Begin Plotting the Solutions

fig1 = plt.figure(figsize=(5,6.5))
ax1 = plt.axes([0.15,0.5,0.8,0.4])
ax1.plot(solutions.y[0]/AU,solutions.y[1]/AU,'r',label='Body 1')
ax1.plot(solutions.y[3]/AU,solutions.y[4]/AU,'k',label='Body 2')
ax1.plot(solutions.y[6]/AU,solutions.y[7]/AU,'g',label='Body 3')

for i in range(1, 71): # loop to plot points at every year 
    targetTime = i * yearinsecs
    # Find index where t is closest to targetTime
    index = (np.abs(t - targetTime)).argmin()
    # Plot markers
    ax1.plot(solutions.y[0][index]/AU, solutions.y[1][index]/AU, 'ro', markersize=4)
    ax1.plot(solutions.y[3][index]/AU, solutions.y[4][index]/AU, 'ko', markersize=4)
    ax1.plot(solutions.y[6][index]/AU, solutions.y[7][index]/AU, 'go', markersize=4)
    ax1.annotate(str(i), (solutions.y[0][index]/AU, solutions.y[1][index]/AU), color='red', xytext=(5, 5), textcoords='offset points', fontsize='7')
    ax1.annotate(str(i), (solutions.y[3][index]/AU, solutions.y[4][index]/AU), color='black', xytext=(5, 5), textcoords='offset points', fontsize='7')
    ax1.annotate(str(i), (solutions.y[6][index]/AU, solutions.y[7][index]/AU), color='green', xytext=(5, 5), textcoords='offset points', fontsize='7')
    
ax1.set_xlabel('x [Arb.]')
ax1.set_ylabel('y [Arb.]')
ax1.legend(loc='upper left')

ax2 = plt.axes([0.15,0.15,0.8,0.25])
ax2.plot(t/yearinsecs,solutions.y[9]/AU,'r')
ax2.plot(t/yearinsecs,solutions.y[12]/AU,'k')
ax2.plot(t/yearinsecs,solutions.y[15]/AU,'g')
ax2.set_xlabel('t [yr]')
ax2.set_ylabel('Vx [Arb.]')

#Energy conservation check:
#extract positions and velocities
x1, y1, z1, x2, y2, z2, x3, y3, z3 = solutions.y[0], solutions.y[1], solutions.y[2], solutions.y[3], solutions.y[4], solutions.y[5], solutions.y[6], solutions.y[7], solutions.y[8]
vx1, vy1, vz1, vx2, vy2, vz2, vx3, vy3, vz3 = solutions.y[9], solutions.y[10], solutions.y[11], solutions.y[12], solutions.y[13], solutions.y[14], solutions.y[15], solutions.y[16], solutions.y[17] 

#kinetic energies
K1 = 0.5 * m1 * (vx1**2 + vy1**2 + vz1**2)
K2 = 0.5 * m2 * (vx2**2 + vy2**2 + vz2**2)
K3 = 0.5 * m3 * (vx3**2 + vy3**2 + vz3**2)
K = K1 + K2 + K3

#potential energy calculation
r12 = ((x2-x1)**2+(y2-y1)**2+(z2-z1)**2)**0.5 #distance between body 1 and 2
r13 = ((x3-x1)**2+(y3-y1)**2+(z3-z1)**2)**0.5 #distance between body 1 and 3
r23 = ((x3-x2)**2+(y3-y2)**2+(z3-z2)**2)**0.5 #distance between body 2 and 3

#total Potential energy
U = -G * ((m1 * m2 / r12) + (m1 * m3 / r13) + (m2 * m3 / r23))

#total energy
E = K + U
E0 = E[0]

#relative energy change delta E as a function of time
dE = (E - E0) / E0

# plot delta E
plt.figure(figsize=(5,4))
plt.plot(solutions.t / yearinsecs, dE)
plt.xlabel('t [yr]')
plt.ylabel('ΔE')
plt.title('Relative change in total energy')
plt.grid(True)
plt.tight_layout()
plt.show()

#Burrau Plots: last 4 decades in quadrant figure
valInSp = int(np.round(1e6 / 70)) # values per year

fig2 = plt.figure(figsize=(7,7))
ax_tl = plt.axes([0.10,0.55,0.35,0.35]) # top left
ax_tr = plt.axes([0.55,0.55,0.35,0.35]) # top right
ax_bl = plt.axes([0.10,0.10,0.35,0.35]) # bottom left
ax_br = plt.axes([0.55,0.10,0.35,0.35]) # bottom right

axes_list = [ax_tl, ax_tr, ax_bl, ax_br]
titles = ['30-40 yr','40-50 yr','50-60 yr','60-70 yr']

for j in range(4): # j = 0,1,2,3 -> decades 30-40-50-60-70
    decade = j + 4 # 4th,5th,6th,7th decade
    splStart = int(np.round((decade*10 - 10) * valInSp)) #splice start value
    splEnd   = int(np.round(decade*10 * valInSp))        #splice end value
    ax2 = axes_list[j]
    
    #plot trajectories in this decade
    ax2.plot(solutions.y[0][splStart:splEnd]/AU,solutions.y[1][splStart:splEnd]/AU,'r',label='Body 1')
    ax2.plot(solutions.y[3][splStart:splEnd]/AU,solutions.y[4][splStart:splEnd]/AU,'k',label='Body 2')
    ax2.plot(solutions.y[6][splStart:splEnd]/AU,solutions.y[7][splStart:splEnd]/AU,'g',label='Body 3')
    
    #plot markers and year labels in this decade
    start_year = decade*10 - 10
    end_year = decade*10
    for yr in range(start_year, end_year+1):
        if yr % 5 == 0:
            targetTime = yr * yearinsecs
            index = (np.abs(t - targetTime)).argmin()
            ax2.plot(solutions.y[0][index]/AU, solutions.y[1][index]/AU, 'ro', markersize=4)
            ax2.plot(solutions.y[3][index]/AU, solutions.y[4][index]/AU, 'ko', markersize=4)
            ax2.plot(solutions.y[6][index]/AU, solutions.y[7][index]/AU, 'go', markersize=4)
            ax2.annotate(str(yr), (solutions.y[0][index]/AU, solutions.y[1][index]/AU), color='red', xytext=(5, 5), textcoords='offset points', fontsize='7')
            ax2.annotate(str(yr), (solutions.y[3][index]/AU, solutions.y[4][index]/AU), color='black', xytext=(5, 5), textcoords='offset points', fontsize='7')
            ax2.annotate(str(yr), (solutions.y[6][index]/AU, solutions.y[7][index]/AU), color='green', xytext=(5, 5), textcoords='offset points', fontsize='7')
        
    ax2.set_title(titles[j])

#axis labels and legend
ax_tl.set_ylabel('y [Arb.]')
ax_bl.set_ylabel('y [Arb.]')
ax_br.set_ylabel('y [Arb.]')
ax_tr.set_ylabel('y [Arb.]')

ax_tl.set_xlabel('x [Arb.]')
ax_bl.set_xlabel('x [Arb.]')
ax_br.set_xlabel('x [Arb.]')
ax_tr.set_xlabel('x [Arb.]')
ax_tl.legend(loc='upper left')

if plotAllBurrau == True: #toggleable now as clogs screen if turned on
    #Burrau Plots
    for i in range(1,8):
        valInSp = int(np.round(1e6 / 70)) # values per 10th of a splice
        splStart = int(np.round((i*10 - 10) * valInSp)) #splice start value
        splEnd = int(np.round(i*10 * valInSp)) #splice end value

        #stop any slight overrun due to rounding 1e6/70
        splEnd = min(splEnd, len(t))

        fig2 = plt.figure(figsize=(7,7))
        ax2 = plt.axes([0.15,0.5,0.8,0.4])
        ax2.plot(solutions.y[0][splStart:splEnd]/AU,solutions.y[1][splStart:splEnd]/AU,'r',label='Body 1')
        ax2.plot(solutions.y[3][splStart:splEnd]/AU,solutions.y[4][splStart:splEnd]/AU,'k',label='Body 2')
        ax2.plot(solutions.y[6][splStart:splEnd]/AU,solutions.y[7][splStart:splEnd]/AU,'g',label='Body 3')
        
        for yr in range(int(np.round(splStart/valInSp)), int(np.round(splEnd/valInSp))+1): # loop to plot points at every year i.e. from 0 - 10, 10 - 20
            targetTime = yr * yearinsecs
            # Find index where t is closest to targetTime
            index = (np.abs(t - targetTime)).argmin()
            # Plot markers
            ax2.plot(solutions.y[0][index]/AU, solutions.y[1][index]/AU, 'ro', markersize=4)
            ax2.plot(solutions.y[3][index]/AU, solutions.y[4][index]/AU, 'ko', markersize=4)
            ax2.plot(solutions.y[6][index]/AU, solutions.y[7][index]/AU, 'go', markersize=4)
            ax2.annotate(str(yr), (solutions.y[0][index]/AU, solutions.y[1][index]/AU), color='red', xytext=(5, 5), textcoords='offset points', fontsize='7')
            ax2.annotate(str(yr), (solutions.y[3][index]/AU, solutions.y[4][index]/AU), color='black', xytext=(5, 5), textcoords='offset points', fontsize='7')
            ax2.annotate(str(yr), (solutions.y[6][index]/AU, solutions.y[7][index]/AU), color='green', xytext=(5, 5), textcoords='offset points', fontsize='7')
            
        ax2.set_xlabel('x [Arb.]')
        ax2.set_ylabel('y [Arb.]')
        ax2.legend(loc='upper left')

plt.show()



