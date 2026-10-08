import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

#From readme file: The time to process this file is heavily dependant on the tolerance of the solve_ivp. 
#A tolerance of 1e-6 gives statistically correct simulation but takes over 2 minutes to run. 
#Lowering tolerance will decrease run time.
tolerance = 1e-6
#tolerance = 1e-3

#inits
t = 0

#cluster settings
N = 30 #body count
G = 1 #had issues with g being realistic (small pertubations so took forever) so normalise to 1
yearinsecs = 1 #yr in secs

mStar = 1 #all stars equal mass
m = np.ones(N)*mStar #masses

R0 = 1.0 #cluster radius [arb.], keep units simple
soft = 0.15 #softening length to stop blow ups [arb.]

seed = 67 #set the seed to scatter the stars with
np.random.seed(seed)

def func(t, y0, G, m, N): #function to pass ODE's into solve_ivp

    r = y0[:3*N].reshape((N, 3))  #xyz means every other 3 is a new body. so times by body count to get to next section
    v = y0[3*N:].reshape((N, 3)) 
    a = np.zeros((N,3)) #set up the acceleration 2d array

    for i in range(N): #sum over all bodies
        for j in range(i+1, N): #only do each pair once
            rijVec = r[j] - r[i]
            rij = (rijVec[0]**2 + rijVec[1]**2 + rijVec[2]**2 + soft**2)**0.5 #distance between body i and j
            fac = G / rij**3

            a[i] += m[j] * fac * rijVec
            a[j] -= m[i] * fac * rijVec

    return np.concatenate((v.flatten(), a.flatten())) #returns the flattened vels and accs.

#random positions in a sphere
r0 = np.zeros((N,3))
count = 0
while count < N: #scatters them with the seed before
    x = (2*np.random.rand()-1)*R0
    y = (2*np.random.rand()-1)*R0
    z = (2*np.random.rand()-1)*R0
    if (x**2 + y**2 + z**2)**0.5 <= R0:
        r0[count] = [x,y,z]
        count += 1

#centre of mass correction
rCM = np.zeros(3)
mTot = np.sum(m)
for i in range(N):
    rCM += m[i]*r0[i]
rCM = rCM/mTot #finds centre mss
for i in range(N):
    r0[i] = r0[i] - rCM

#random velocities (will be scaled)
v0 = np.random.normal(0, 1, (N,3))

#remove net momentum
vCM = np.zeros(3)
for i in range(N):
    vCM += m[i]*v0[i]
vCM = vCM/mTot #finds centre velocity 
for i in range(N):
    v0[i] = v0[i] - vCM #make sure all velocities add to 0 

#compute initial K and U to scale to a target virial ratio
def KU(rNow, vNow, G, m, N):
    K = 0
    U = 0

    for i in range(N):
        v2 = vNow[i][0]**2 + vNow[i][1]**2 + vNow[i][2]**2
        K += 0.5*m[i]*v2

    for i in range(N):
        for j in range(i+1, N):
            rij = ((rNow[i][0]-rNow[j][0])**2+(rNow[i][1]-rNow[j][1])**2+(rNow[i][2]-rNow[j][2])**2 + soft**2)**0.5
            U += -G*m[i]*m[j]/rij

    return K, U

#two runs: sub-virial (collapse) and unbound (dispersal)
QList = [0.30, 1.50] #Q = K/|U| (only used to set initial speeds)
labels = ['Sub-virial (collapse)', 'Unbound (disperse)']

#time
tmin = 0
tEnd = 15 #arb. time
tvals = int(700)
t = np.linspace(tmin, tEnd, tvals, endpoint=False)
t_span = [tmin, tEnd]

solList = []
rhList = []
dEList = []

for run in range(2):# run the collapse then the disperse sims

    Q = QList[run]

    K0, U0 = KU(r0, v0, G, m, N)
    scale = (Q*abs(U0)/K0)**0.5
    vScaled = v0*scale

    #print initial energy sign (E0>0 means globally unbound)
    E0check = (K0*scale**2 + U0)
    print('Run', run+1, 'Q =', Q, ', E0 =', E0check)

    y0 = np.concatenate((r0.flatten(), vScaled.flatten()))

    print('Running case', run+1, 'of 2') # print which mode is being computed
    solutions = solve_ivp(func, t_span, y0, t_eval=t, args=(G,m,N), atol=tolerance, rtol=tolerance)
    solList.append(solutions)

    #diagnostics vs time: half-mass radius and total energy error
    rh = []
    E = []

    for k in range(len(solutions.t)):
        rNow = solutions.y[:3*N, k].reshape((N,3))
        vNow = solutions.y[3*N:, k].reshape((N,3))

        #half-mass radius about CM
        rCM = np.zeros(3)
        for i in range(N):
            rCM += m[i]*rNow[i]
        rCM = rCM/mTot

        distList = []
        for i in range(N):
            dx = rNow[i][0]-rCM[0]
            dy = rNow[i][1]-rCM[1]
            dz = rNow[i][2]-rCM[2]
            distList.append((dx**2+dy**2+dz**2)**0.5)

        distSort = np.sort(np.array(distList))
        rh.append(distSort[int(N/2)])

        Kt, Ut = KU(rNow, vNow, G, m, N)
        E.append(Kt + Ut)

    rhList.append(np.array(rh))

    E = np.array(E)
    E0 = E[0]
    dEList.append((E - E0)/E0)

#Figure 7: snapshots (x-y projection), centred on CM
fig7 = plt.figure(figsize=(10,8))

ax1 = plt.axes([0.10,0.58,0.38,0.34]) #top left
ax2 = plt.axes([0.55,0.58,0.38,0.34]) #top right
ax3 = plt.axes([0.10,0.12,0.38,0.34]) #bottom left
ax4 = plt.axes([0.55,0.12,0.38,0.34]) #bottom right

#pick t=0 and final index
k0 = 0
kEnd = -1

#helper limits (hide extreme escapers so the cluster is visible) (top 10 %)
trimFrac = 0.90

#sub-virial
solA = solList[0]
rA0 = solA.y[:3*N, k0].reshape((N,3))
rAE = solA.y[:3*N, kEnd].reshape((N,3))


#shift to centre of mass for better view

#CM shift at t=0 #sub virial
rCM = np.zeros(3)
for i in range(N):
    rCM += m[i]*rA0[i]
rCM = rCM/mTot
for i in range(N):
    rA0[i] = rA0[i] - rCM

#CM shift at t=tEnd
rCM = np.zeros(3)
for i in range(N):
    rCM += m[i]*rAE[i]
rCM = rCM/mTot
for i in range(N):
    rAE[i] = rAE[i] - rCM

#unbound
solB = solList[1]
rB0 = solB.y[:3*N, k0].reshape((N,3))
rBE = solB.y[:3*N, kEnd].reshape((N,3))

#CM shift at t=0 #unbound
rCM = np.zeros(3)
for i in range(N):
    rCM += m[i]*rB0[i]
rCM = rCM/mTot
for i in range(N):
    rB0[i] = rB0[i] - rCM

#CM shift at t=tEnd
rCM = np.zeros(3)
for i in range(N):
    rCM += m[i]*rBE[i]
rCM = rCM/mTot
for i in range(N):
    rBE[i] = rBE[i] - rCM


#plotting the graphs

#axis limits
lim0 = 1.2*R0

dA = []
dB = []
for i in range(N):
    dA.append((rAE[i][0]**2 + rAE[i][1]**2 + rAE[i][2]**2)**0.5)
    dB.append((rBE[i][0]**2 + rBE[i][1]**2 + rBE[i][2]**2)**0.5)

dA = np.sort(np.array(dA))
dB = np.sort(np.array(dB))

limEnd = limEnd = 0.65*max(dA[int(trimFrac*N)-1], dB[int(trimFrac*N)-1])

ax1.plot(rA0[:,0], rA0[:,1], 'r*', markersize=5) #plot sub virial colapse t 0
ax1.set_title(labels[0]+' , t=0')
ax1.set_xlabel('x [arb.]')
ax1.set_ylabel('y [arb.]')
ax1.set_xbound(-lim0, lim0)
ax1.set_ybound(-lim0, lim0)
ax1.grid(True)

ax2.plot(rB0[:,0], rB0[:,1], 'r*', markersize=5)#plot unbound colapse t 0
ax2.set_title(labels[1]+' , t=0')
ax2.set_xlabel('x [arb.]')
ax2.set_ylabel('y [arb.]')
ax2.set_xbound(-lim0, lim0)
ax2.set_ybound(-lim0, lim0)
ax2.grid(True)

ax3.plot(rAE[:,0], rAE[:,1], 'r*', markersize=5) #plot sub virial colapse t 15
ax3.set_title(labels[0]+' , t='+str(np.round(tEnd,2)))
ax3.set_xlabel('x [arb.]')
ax3.set_ylabel('y [arb.]')
ax3.set_xbound(-limEnd, limEnd)
ax3.set_ybound(-limEnd, limEnd)
ax3.grid(True)

ax4.plot(rBE[:,0], rBE[:,1], 'r*', markersize=3) #plot unbound colapse t 15
ax4.set_title(labels[1]+' , t='+str(np.round(tEnd,2)))
ax4.set_xlabel('x [arb.]')
ax4.set_ylabel('y [arb.]')
ax4.set_xbound(-limEnd, limEnd)
ax4.set_ybound(-limEnd, limEnd)
ax4.grid(True)

plt.show()

#Figure 8: diagnostics
fig8 = plt.figure(figsize=(10,4.8))

ax5 = plt.axes([0.10,0.18,0.38,0.72]) #left
ax6 = plt.axes([0.58,0.18,0.38,0.72]) #right

ax5.plot(t, rhList[0], 'k', label='Q='+str(QList[0])) #half mass radius
ax5.plot(t, rhList[1], '--', label='Q='+str(QList[1]))
ax5.set_xlabel('t [arb.]')
ax5.set_ylabel(r'$r_{1/2}$ [arb.]')
ax5.set_title('Half-mass radius')
ax5.grid(True)
ax5.legend(loc='upper right')

ax6.plot(t, dEList[0], 'k', label='Q='+str(QList[0])) #relative energy  error plot
ax6.plot(t, dEList[1], '--', label='Q='+str(QList[1]))
ax6.axhline(0.0, linestyle='--')
ax6.set_xlabel('t [arb.]')
ax6.set_ylabel(r'$\Delta E / E_0$')
ax6.set_title('Relative energy error')
ax6.grid(True)
ax6.legend(loc='upper right')

plt.show()
