import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

#Set up burrau initial conditions
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
years = 70
tmax = int(years*yearinsecs) # years in seconds
tvals = int(1e6) #data points
y0 = [x1,y1,z1,x2,y2,z2,x3,y3,z3,vx1,vy1,vz1,vx2,vy2,vz2,vx3,vy3,vz3]
t = np.linspace(tmin,tmax,tvals,endpoint=False)
t_span = [tmin,tmax]

solutions = solve_ivp(func, t_span, y0,t_eval=t,args=(G,m1,m2),atol=1e-12,rtol=1e-12) #tolerances low due to high values
solutionsPlanar = solutions #store planar run


#only plot first 10 years to reduce clutter
yearsToPlot = 10
valInSp = int(np.round(tvals / years)) # values per year
splStart = 0
splEnd = int(np.round(yearsToPlot * valInSp)) #splice end value for first 10 years



#Plot in 3D to show no 3D action in burrau
fig3d = plt.figure(figsize=(10, 8))
colours = ['y', 'b', 'r']
ax3d = fig3d.add_subplot(111, projection='3d')

N = 3
m = [3,4,5]
for i in range(N): #split by 3 for each orthogonal 
    ax3d.plot(solutions.y[3*i][splStart:splEnd]/AU, solutions.y[3*i+1][splStart:splEnd]/AU, solutions.y[3*i+2][splStart:splEnd]/AU, label='Body '+ str(i+1) + ( '%.3e' % m[i]))
    ax3d.scatter(solutions.y[3*i][splEnd-1]/AU, solutions.y[3*i+1][splEnd-1]/AU, solutions.y[3*i+2][splEnd-1]/AU, marker='o', color=colours[i])

ax3d.set_xlabel('x [arb.]')
ax3d.set_ylabel('y [arb.]')
ax3d.set_zlabel('z [arb.] ')
ax3d.legend()
plt.show()


######

##### Give each body a velocity in the Z- direction to prove 3D dynamics
vz1 = 0
vz2 = -0.03
vz3 = 0.03
y0 = [x1,y1,z1,x2,y2,z2,x3,y3,z3,vx1,vy1,vz1,vx2,vy2,vz2,vx3,vy3,vz3] #recompile the initial conditions into y0
solutions = solve_ivp(func, t_span, y0,t_eval=t,args=(G,m1,m2),atol=1e-12,rtol=1e-12) #tolerances low due to high values
solutionsKick = solutions #store kicked run

#plot 3d to show change in orbits
fig3d2 = plt.figure(figsize=(10, 8))
colours = ['y', 'b', 'r']
ax3d2 = fig3d2.add_subplot(111, projection='3d')

N = 3
m = [3,4,5]
for i in range(N): #split by 3 for each orthogonal 
    ax3d2.plot(solutions.y[3*i][splStart:splEnd]/AU, solutions.y[3*i+1][splStart:splEnd]/AU, solutions.y[3*i+2][splStart:splEnd]/AU, label='Body '+ str(i+1) + ( '%.3e' % m[i]))
    ax3d2.scatter(solutions.y[3*i][splEnd-1]/AU, solutions.y[3*i+1][splEnd-1]/AU, solutions.y[3*i+2][splEnd-1]/AU, marker='o', color=colours[i])

ax3d2.set_xlabel('x [arb.]')
ax3d2.set_ylabel('y [arb.]')
ax3d2.set_zlabel('z [arb.]')
ax3d2.legend()

plt.show()

#Plot both 3D cases on one figure for exporting
figBoth = plt.figure(figsize=(12, 6))
axL = figBoth.add_subplot(121, projection='3d')
axR = figBoth.add_subplot(122, projection='3d')

colours = ['y', 'b', 'r']
N = 3
m = [3,4,5]

#left plot: planar case
for i in range(N): #split by 3 for each orthogonal 
    axL.plot(solutionsPlanar.y[3*i][splStart:splEnd]/AU, solutionsPlanar.y[3*i+1][splStart:splEnd]/AU, solutionsPlanar.y[3*i+2][splStart:splEnd]/AU, label='Body '+ str(i+1))
    axL.scatter(solutionsPlanar.y[3*i][splEnd-1]/AU, solutionsPlanar.y[3*i+1][splEnd-1]/AU, solutionsPlanar.y[3*i+2][splEnd-1]/AU, marker='o', color=colours[i])
#add z=0 plane that touches the walls
xlims = axL.get_xlim() 
ylims = axL.get_ylim()
X, Y = np.meshgrid([xlims[0], xlims[1]], [ylims[0], ylims[1]]) #use the ends of the data for edge of plane
Z = 0*X
axL.plot_surface(X, Y, Z, color='grey', alpha=0.2, linewidth=0, shade=False) #add the plane 
axL.set_xlim(xlims[0], xlims[1])
axL.set_ylim(ylims[0], ylims[1])
axL.set_xlabel('x [arb.]')
axL.set_ylabel('y [arb.]')
axL.set_zlabel('z [arb.]')
axL.set_title('Burrau in 3D, Vz = 0, first 10 years') 
axL.grid(True)
axL.legend(loc='upper left', fontsize=8)

#right plot: kicked case, same as planar just using the kicked data
for i in range(N): #split by 3 for each orthogonal 
    axR.plot(solutionsKick.y[3*i][splStart:splEnd]/AU, solutionsKick.y[3*i+1][splStart:splEnd]/AU, solutionsKick.y[3*i+2][splStart:splEnd]/AU, label='Body '+ str(i+1))
    axR.scatter(solutionsKick.y[3*i][splEnd-1]/AU, solutionsKick.y[3*i+1][splEnd-1]/AU, solutionsKick.y[3*i+2][splEnd-1]/AU, marker='o', color=colours[i])
#add z=0 plane that touches the walls
xlims = axR.get_xlim()
ylims = axR.get_ylim()
X, Y = np.meshgrid([xlims[0], xlims[1]], [ylims[0], ylims[1]])
Z = 0*X
axR.plot_surface(X, Y, Z, color='grey', alpha=0.2, linewidth=0, shade=False)
axR.set_xlim(xlims[0], xlims[1])
axR.set_ylim(ylims[0], ylims[1])
axR.set_xlabel('x [arb.]')
axR.set_ylabel('y [arb.]')
axR.set_zlabel('z [arb.]')
axR.set_title('Burrau in 3D, Vz = Perturbed, first 10 years')
axR.grid(True)
axR.legend(loc='upper left', fontsize=8)


plt.show()
