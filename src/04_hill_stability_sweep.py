import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

#From readme: This file takes about 60 seconds to run because it repeats the integration many (25) times.
#inits
t = 0

N = 3 #body count (Sun, planet, moon)
mSol = 1.989e30
mEarth = 5.978e24
mMoon = 7.342e22
m = [mSol,mEarth,mMoon] #masses
G = 6.674e-11
AU = 1.496e11
yearinsecs = 31557600 #yr in secs

#aPlanet is fixed, sweep moon distance
aPlanet = 1.0*AU

#Hill radius for reference
rHill = aPlanet * (mEarth/(3*mSol))**(1/3)

#how strict is "bound'
boundFrac = 0.50 #moon must stay within this fraction of Hill radius

def func(t, y0, G, m, N): #function to pass ODE's into solve_ivp

    r = y0[:3*N].reshape((N, 3))  #xyz means every other 3 is a new body. so times by body count to get to next section
    v = y0[3*N:].reshape((N, 3)) 
    a = np.zeros((N,3)) #set up the acceleration 2d array
    
    for i in range(N): #sum over all bodies
        for j in range(N): #then again for the other bodies
            if i != j: #as long as it isnt the same body
                rij = ((r[i][0]-r[j][0])**2+(r[i][1]-r[j][1])**2+(r[i][2]-r[j][2])**2)**0.5 #distance between body i and j 
                rijVec = r[j] - r[i] # the vector version of distance between i and j, can be passed directly to a
                
                a[i] += G * m[j] * rijVec / rij**3
    
    return np.concatenate((v.flatten(), a.flatten())) #returns the flattened vels and accs.

#time for sweep runs (keep it modest so sweep is fast)
tmin = 0
sweepYears = 3
tmax = int(sweepYears*yearinsecs)
tvals = int(5000) #data points
t = np.linspace(tmin,tmax,tvals,endpoint=False)
t_span = [tmin,tmax]

#moon distance sweep (in Hill radius units)
numPoints = 25
aMoonList = np.linspace(0.05*rHill, 0.90*rHill, numPoints)

stableList = []
maxDistList = []

#store one stable and one unstable run to show as examples
storedStable = 0
storedUnstable = 0
rStable = None
rUnstable = None
aStable = 0
aUnstable = 0

for k in range(numPoints): #loop over moon starting distances
    aMoon = aMoonList[k]
    
    #positions
    r0 = np.zeros((N,3))
    r0[0] = [0,0,0] #Sun
    r0[1] = [aPlanet,0,0] #Planet
    r0[2] = [aPlanet + aMoon,0,0] #Moon
    
    #velocities
    v0 = np.zeros((N,3))
    v0[0] = [0,0,0]
    
    #planet circular orbit around Sun
    vPlanet = np.sqrt(G*m[0]/aPlanet)
    v0[1] = [0,vPlanet,0]
    
    #moon circular orbit around planet (add planet orbital velocity)
    vMoon = np.sqrt(G*m[1]/aMoon)
    v0[2] = [0,vPlanet + vMoon,0]
    
    y0 = np.concatenate((r0.flatten(), v0.flatten()))
    solutions = solve_ivp(func, t_span, y0,t_eval=t,args=(G,m,N),atol=1e-8,rtol=1e-8)
    
    #distance moon-planet
    xP = solutions.y[3]
    yP = solutions.y[4]
    zP = solutions.y[5]
    xM = solutions.y[6]
    yM = solutions.y[7]
    zM = solutions.y[8]
    
    rMP = ((xM-xP)**2 + (yM-yP)**2 + (zM-zP)**2)**0.5
    maxDist = np.max(rMP)
    
    maxDistList.append(maxDist)
    
    #bound/escape classification
    if maxDist < boundFrac*rHill:
        stableList.append(1)
        if storedStable == 0:
            rStable = rMP
            aStable = aMoon
            storedStable = 1
    else:
        stableList.append(0)
        if storedUnstable == 0:
            rUnstable = rMP
            aUnstable = aMoon
            storedUnstable = 1

stableArray = np.array(stableList)
maxDistArray = np.array(maxDistList)

#Final figure
fig5 = plt.figure(figsize=(10,8))

ax1 = plt.axes([0.10,0.58,0.38,0.34]) #top left
ax2 = plt.axes([0.55,0.58,0.38,0.34]) #top right
ax3 = plt.axes([0.10,0.12,0.38,0.34]) #bottom left
ax4 = plt.axes([0.55,0.12,0.38,0.34]) #bottom right

#top left: example stable r_mp(t)
if rStable is not None:
    ax1.plot(t/yearinsecs, rStable/rHill, 'k')
    ax1.axhline(boundFrac, linestyle='--')
    ax1.set_title(r'Example stable moon, $a_m/R_H$ = '+str(np.round(aStable/rHill,3)))
ax1.set_xlabel('t [yr]')
ax1.set_ylabel(r'$r_{mp} / R_H$')
ax1.grid(True)

#top right: example unstable r_mp(t)
if rUnstable is not None:
    ax2.plot(t/yearinsecs, rUnstable/rHill, 'k')
    ax2.axhline(boundFrac, linestyle='--')
    ax2.set_title(r'Example unstable moon, $a_m/R_H$ = '+str(np.round(aUnstable/rHill,3)))
ax2.set_xlabel('t [yr]')
ax2.set_ylabel(r'$r_{mp} / R_H$')
ax2.grid(True)

#bottom left: bound(1) / not bound(0) vs initial distance
ax3.plot(aMoonList/rHill, stableArray, 'ko')
ax3.set_xlabel(r'Initial moon distance  $a_m / R_H$')
ax3.set_ylabel('Bound after '+str(sweepYears)+' yr')
ax3.set_title('Stability outcome from parameter sweep')
ax3.set_ybound(-0.1,1.1)
ax3.grid(True)

#bottom right: max separation vs initial distance
ax4.plot(aMoonList/rHill, maxDistArray/rHill, 'ko')
ax4.axhline(boundFrac, linestyle='--')
ax4.set_xlabel(r'Initial moon distance  $a_m / R_H$')
ax4.set_ylabel(r'$\max(r_{mp}) / R_H$')
ax4.set_title('Maximum moon separation reached')
ax4.grid(True)

plt.show()






