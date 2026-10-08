import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

#From readme: This file is a parameter sweep so it repeats the integration many times.
#Estimated time to run: ~3-4 Minutes. Took 3 on powerful cpu
#it prints the % completion in the console.
#moved into new file as processing time was too long

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

#aPlanet is fixed
aPlanet = 1.0*AU

#Hill radius for reference
rHill = aPlanet * (mEarth/(3*mSol))**(1/3)

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

#Figure 6: moon stability map by phase averaging and planet-centric unbinding
#escape condition: moon becomes unbound when its specific orbital energy about the planet goes positive

tmin6 = 0
fig6Years = 3
tmax6 = int(fig6Years*yearinsecs)
t_span6 = [tmin6,tmax6]

#sweep grid
aMoonFracList6 = np.linspace(0.20, 0.90, 12) #a_m/R_H
incDegList6 = np.linspace(0, 180, 10) #degrees (include retrograde)

#phase averaging
numPhase = 6
phiList = np.linspace(0, 2*np.pi, numPhase, endpoint=False)
cosPhiList = np.cos(phiList)
sinPhiList = np.sin(phiList)

surviveFracMat6 = np.zeros((len(incDegList6), len(aMoonFracList6)))
medianTimeMat6 = np.zeros((len(incDegList6), len(aMoonFracList6)))

#planet circular orbit about Sun at t=0
vPlanet = np.sqrt(G*m[0]/aPlanet)

def eventUnbind(t, y, *args): #stop when moon becomes unbound from planet
    dx = y[6] - y[3]
    dy = y[7] - y[4]
    dz = y[8] - y[5]
    rMP = (dx**2 + dy**2 + dz**2)**0.5

    dvx = y[15] - y[12]
    dvy = y[16] - y[13]
    dvz = y[17] - y[14]
    vRel2 = dvx**2 + dvy**2 + dvz**2

    eps = 0.5*vRel2 - G*m[1]/rMP #specific orbital energy about the planet
    return eps

eventUnbind.terminal = True
eventUnbind.direction = 1

for j in range(len(incDegList6)): #loop over inclination
    incDeg = incDegList6[j]
    incRad = incDeg*np.pi/180
    cosInc = np.cos(incRad)
    sinInc = np.sin(incRad)
    print(str(j*10) + "% Complete") #print completion amount
    for k in range(len(aMoonFracList6)): #loop over moon distance
        aMoon = aMoonFracList6[k] * rHill
        vMoon = np.sqrt(G*m[1]/aMoon)

        boundCount = 0
        timeList = []

        for p in range(len(phiList)): #loop over initial moon orbital phase
            cphi = cosPhiList[p]
            sphi = sinPhiList[p]

            #positions
            r0 = np.zeros((N,3))
            r0[0] = [0,0,0] #Sun
            r0[1] = [aPlanet,0,0] #Planet

            #moon relative position about planet (tilted by inclination)
            xRel = aMoon*cphi
            yRel = aMoon*sphi*cosInc
            zRel = aMoon*sphi*sinInc
            r0[2] = [aPlanet + xRel, yRel, zRel]

            #velocities
            v0 = np.zeros((N,3))
            v0[0] = [0,0,0]
            v0[1] = [0,vPlanet,0]

            #moon circular velocity in tilted plane
            vxRel = -vMoon*sphi
            vyRel =  vMoon*cphi*cosInc
            vzRel =  vMoon*cphi*sinInc
            v0[2] = [v0[1][0] + vxRel, v0[1][1] + vyRel, v0[1][2] + vzRel]

            y0 = np.concatenate((r0.flatten(), v0.flatten()))

            solutions6 = solve_ivp(func, t_span6, y0,args=(G,m,N),events=eventUnbind,
                                   atol=1e-7,rtol=1e-7,method='DOP853')

            if len(solutions6.t_events[0]) == 0:
                boundCount += 1
                timeList.append(fig6Years)
            else:
                timeList.append(solutions6.t_events[0][0]/yearinsecs)

        surviveFracMat6[j][k] = boundCount/len(phiList)
        medianTimeMat6[j][k] = np.median(np.array(timeList))
print("100% Complete!")
#Final figure 6
fig6 = plt.figure(figsize=(10,4.8))

ax6a = plt.axes([0.08,0.18,0.40,0.72]) #left panel
ax6b = plt.axes([0.58,0.18,0.38,0.72]) #right panel

im1 = ax6a.imshow(surviveFracMat6, origin='lower', aspect='auto', interpolation='nearest', extent=[aMoonFracList6[0], aMoonFracList6[-1], incDegList6[0], incDegList6[-1]], vmin=0, vmax=1)
ax6a.set_xlabel(r'Initial moon distance  $a_m / R_H$')
ax6a.set_ylabel(r'Inclination  $i$ [deg]')
ax6a.set_title('Survival fraction over '+str(numPhase)+' phases')
ax6a.axhline(90, linestyle='--')
ax6a.grid(False)
plt.colorbar(im1, ax=ax6a, fraction=0.046, pad=0.04)

im2 = ax6b.imshow(medianTimeMat6, origin='lower', aspect='auto', interpolation='nearest', extent=[aMoonFracList6[0], aMoonFracList6[-1], incDegList6[0], incDegList6[-1]], vmin=0, vmax=fig6Years)
ax6b.set_xlabel(r'Initial moon distance  $a_m / R_H$')
ax6b.set_ylabel(r'Inclination  $i$ [deg]')
ax6b.set_title('Median unbinding time (yr)')
ax6b.axhline(90, linestyle='--')
ax6b.grid(False)
plt.colorbar(im2, ax=ax6b, fraction=0.046, pad=0.04)

plt.show()
