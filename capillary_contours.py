import numpy as np
import matplotlib.pyplot as plt
#from matplotlib import cm
from scipy.optimize import curve_fit
import MDAnalysis
import os


"""
This script computes the line tension,$\gamma$, from the Fourier spectrum of the hight fluctuations of the interfaces,
$\left \langle  |\delta h_k|^2  \right \rangle $.
According to capillary theory (Safran (1994), Statistical thermodynamics of surfaces and interfaces,
Addison-Wesley, New York),
$\left \langle  |\delta h_k|^2  \right \rangle \approx k_BT/ L \gamma k^2 $
for small $k$.  $L$ is the box size, $k = \frac{2 \pi m}{L}$, $m = 0, \pm 1, \pm2, ...$

Despite our interface is stable, problem is that our box Ly dimension is pretty small for correctly see
the proportionality of $\gamma$ to $1/k^2$.
An ansatz for problems of this type has been put forward by  Triebllo and Ceriotti:
    https://iopscience.iop.org/article/10.1088/1361-648X/aa893d/pdf
In this script their method is implemented as well.

"""


input_dir = "ANALYSIS/directors/contours/"
output_dir = "ANALYSIS/line_tension_capillary_theory/"


if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def mean_sorter(a1, a2):
    mean1 = np.mean(a1)
    mean2 = np.mean(a2)
    if mean1 <= mean2:
        return a1, a2
    else:
        return a2, a1

def DFT(f, y, Ly, delta_y):
    """Compute the discrete Fourier Transform of the 1D array x  
        #definition:
        #fk=∑_y=0 ^(Ly−1) f(y)⋅e(−i  k y) *dy"""    
    N = f.shape[0]
    #delta_y = np.diff(y)
    k = (np.arange(0, N)* 2*np.pi/Ly).reshape(-1,1) #(np.arange(0, N, 2*np.pi/Ly)[:N-1]).reshape(N-1,1)
    M = np.exp((-1j *  k * y[:N]))
    delta_y_f= np.multiply(delta_y, f[:N])
    return np.dot(M, delta_y_f)/ Ly
   
Kb= 1.38064 * 10**(-23)# J K^−1 
T=298 #K

contours_left_down = []
contours_right_down = []
contours_left_up = []
contours_right_up = []

top  = 'ANALYSIS/recentered_x.gro'
traj = 'ANALYSIS/recentered_x.xtc'
u = MDAnalysis.Universe(top,traj)

for j in range(0, u.trajectory.n_frames,1):
    cont_down =  np.load(input_dir + 'contours_lower.'+ str(j)+'.npy', allow_pickle=True)
    sorted_contours_down = sorted(cont_down, key=len, reverse=True)    
    left_down, right_down = mean_sorter(sorted_contours_down[0], sorted_contours_down[1])
    
    if (right_down[0,1]- right_down[-1,1] > 0):
        right_down = np.flip(right_down, axis =0)
    if (left_down[0,1]- left_down[-1,1] > 0):
        left_down = np.flip(left_down, axis =0)
    
    contours_left_down.append(left_down/10)  #to have it in nm
    contours_right_down.append(right_down/10)
    
    cont_up =  np.load(input_dir + 'contours_upper.'+ str(j)+'.npy', allow_pickle=True)
    sorted_contours_up = sorted(cont_up, key=len, reverse=True)    
    left_up, right_up = mean_sorter(sorted_contours_up[1], sorted_contours_up[0])
    
    if (right_up[0,1]- right_up[-1,1] > 0):
        right_up = np.flip(right_up, axis =0)
    if (left_up[0,1]- left_up[-1,1] > 0):
        left_up = np.flip(left_up, axis =0)
    
    contours_left_up.append(left_up/10)  #to have it in nm
    contours_right_up.append(right_up/10)
    

contours_up = contours_left_up  + contours_left_down  
contours_down = contours_right_up  + contours_right_down  



# =================== check the contours ==========================================================
#plt.figure()
#for i in range(0, (len(contours_left_up)), 10):
#    f_id=contours_left_up[i][:,0]
#    y_id = contours_left_up[i][:,1]   
#     #f_iu=contours_up[i][:,0]
#     #y_iu = contours_up[i][:,1]   
# 
#    plt.plot(f_id, y_id)
#     #plt.plot(f_iu, y_iu, "--")
## plt.show()
# 
#    
#for i in range(0, (len(contours_left_down)), 10):
#    f_id=contours_left_down[i][:,0]
#    y_id = contours_left_down[i][:,1]   
#     #f_iu=contours_up[i][:,0]
#     #y_iu = contours_up[i][:,1]   
# 
#    plt.plot(f_id, y_id)
#    
#    
## plt.figure()
#for i in range(0, (len(contours_right_down)), 10):
#    f_id=contours_right_down[i][:,0]
#    y_id = contours_right_down[i][:,1]   
#     #f_iu=contours_up[i][:,0]
#     #y_iu = contours_up[i][:,1]   
# 
#    plt.plot(f_id, y_id)
#     #plt.plot(f_iu, y_iu, "--")
# #plt.show() 
# 
# 
# 
# 
# 
#plt.figure()
#for i in range(0, (len(contours_up)), 1):
#    f_id=contours_down[i][:,0]
#    y_id = contours_down[i][:,1]   
#    f_iu=contours_up[i][:,0]
#    y_iu = contours_up[i][:,1]   
# 
#    plt.plot(f_id, y_id)
#    plt.plot(f_iu, y_iu, "--")
#plt.show()
# =============================================================================


def func(x, a):
     """ for fitting """
     return a * (1/ (x**2))
 
    
def func_tribello(x,a, b):
     """ for fitting reference: J.Phys. Condens. Matter 29 445001 Extrcting the intrefacial free energy  
     and anisotropy from a smooth fluctuating dividing surface"""
     return a* (1/ (x**2)) * np.exp((-x**2)/2*(b**2))    

fft_down= []
for i in range(0, len(contours_down), 1):
    Ly_i = contours_down[i][-1,1] - (contours_down[i][0,1])    
    f_i=contours_down[i][:,0]
    y_i = contours_down[i][:,1]   
    dy = np.diff(y_i)
    invalid_points = np.where(dy<=0)[0]
    if len(invalid_points) != 0:        
        print("weird contours. skipping them")
        continue
    valid_points = np.where(dy>0)[0]
    dy = dy[valid_points]
    f_i=f_i[valid_points]
    
    k_i = (np.arange(0, len(dy)) *  (2*np.pi/Ly_i) ).reshape(len(dy) ,1) 
    fft_i = DFT(f_i, y_i, Ly_i, dy)
    fft_down.append((fft_i[0:50]))
    
    #plt.plot(f_i, y_i[valid_points])
    
squared_abs_h_k_down  = np.mean((np.abs(fft_down)**2), axis=0)


fft_up= []
box_dimension =[]
for i in range(0, len(contours_up), 1):
    Ly_i = contours_up[i][-1,1] - (contours_up[i][0,1])    
    f_i=contours_up[i][:,0]
    y_i = contours_up[i][:,1]   
    dy = np.diff(y_i)
    invalid_points = np.where(dy<=0)[0]
    if len(invalid_points) != 0:        
        print("weird contours. skipping them")
        continue
    valid_points = np.where(dy>0)[0]
    dy = dy[valid_points]
    f_i=f_i[valid_points]
    
    k_i = (np.arange(0, len(dy)) *  (2*np.pi/Ly_i) ).reshape(len(dy) ,1) 
    fft_i = DFT(f_i, y_i, Ly_i, dy)
    fft_up.append((fft_i[0:50]))
    box_dimension.append((Ly_i))
    
    #plt.plot(f_i, y_i[valid_points])
    
squared_abs_h_k_up  = np.mean((np.abs(fft_up)**2), axis=0)


x_up = np.array(k_i[1:20]).reshape(1,-1)
y_up = np.array((squared_abs_h_k_up[1:20]*Ly_i)).reshape(1,-1)
popt_up, pcov_up = curve_fit(func, x_up.ravel(), y_up.ravel(), p0=[5])

popt_g, pcov_g= curve_fit(func_tribello, x_up.ravel(), y_up.ravel(), p0=[0.05, 0.78])

x2_up = np.array(k_i[2:5]).reshape(1,-1)
print(Ly_i)
y2_up = np.array((squared_abs_h_k_up[2:5]*Ly_i)).reshape(1,-1)
popt_up2, pcov_up2 = curve_fit(func, x2_up.ravel(), y2_up.ravel(), p0=[5])

plt.loglog(k_i[1:50], (squared_abs_h_k_up[1:50]*Ly_i), "-o", label='right_side')
plt.loglog(k_i[1:10], popt_up*(1/(k_i[1:10]**2)))
plt.loglog(k_i[1:10], popt_g*(1/(k_i[1:10]**2)))

plt.loglog(k_i[1:15], (squared_abs_h_k_up[1:15]*Ly_i), "-o", label='right_side')
plt.loglog(k_i[1:10], popt_g[0]*(1/(k_i[1:10]**2)) * (np.exp(-(k_i[1:10]**2)/2*(popt_g[1])**2) ))



plt.ylabel(r'$\left \langle  |\delta h_k|^2  \right \rangle L \quad [nm^{3}]$')
plt.xlabel(r'$k= \frac{2\pi m}{L} \quad [nm^{-1}]$')
plt.legend()



x_down = np.array(k_i[1:3]).reshape(1,-1)
y_down = np.array((squared_abs_h_k_down[1:3]*Ly_i)).reshape(1,-1)
popt_down, pcov_down = curve_fit(func, x_down.ravel(), y_down.ravel(), p0=[5])

plt.loglog(k_i[1:30], (squared_abs_h_k_down[1:30]/squared_abs_h_k_down[0]), "-.", label='left_Side')
plt.loglog(k_i[1:10], popt_down*(1/(k_i[1:10]**2)) )
plt.ylabel(r'$\left \langle  |\delta h_k|^2  \right \rangle L \quad [nm^{3}]$')
plt.xlabel(r'$k= \frac{2\pi m}{L} \quad [nm^{-1}]$')
plt.legend()




#plt.loglog(k_i[0:50], (squared_abs_h_k_down[0:50]/squared_abs_h_k_down[0])*Ly_i , "-.", label='left_Side')
#plt.loglog(k_i[0:50], (squared_abs_h_k_up[0:50]/squared_abs_h_k_up[0])*Ly_i, "-o", label='right_side')



squared_abs_h_k_all  = (np.mean((np.abs(fft_up )**2) , axis=0) + np.mean((np.abs(fft_down)**2) , axis=0) )/2

fft= fft_up +fft_down
squared_abs_h_k_all  = (np.mean((np.abs(fft) **2) , axis=0)) 

x_all = np.array(k_i[1:50]).reshape(1,-1)
y_all = np.array((squared_abs_h_k_all[1:50])).reshape(1,-1)

popt_all, pcov_all = curve_fit(func, x_all.ravel(), y_all.ravel(), p0=[5])

pot_g, pcov_g = curve_fit(func_tribello, x_all.ravel(), y_all.ravel(), p0=[0.08, 0.78])

y_up = np.array((squared_abs_h_k_up[1:50])).reshape(1,-1)
y_down = np.array((squared_abs_h_k_down[1:50])).reshape(1,-1)

pot_up, pcov_up = curve_fit(func_tribello, x_all.ravel(), y_up.ravel(), p0=[0.08, 0.78])

pot_down, pcov_down = curve_fit(func_tribello, x_all.ravel(), y_down.ravel(), p0=[0.08, 0.75])

plt.loglog(k_i[1:10], (squared_abs_h_k_all[1:10]), "-o", color ='green')
plt.loglog(k_i[0:50], (squared_abs_h_k_up[0:50]), "-o", label='right_side_up')
plt.loglog(k_i[0:50], (squared_abs_h_k_down[0:50] ), "-o", label='right_side_down')
#plt.loglog(k_i[2:5], 0.3*(1/(k_i[2:5]**2))* np.exp(-(k_i[2:5]**2)/2*(0.8**2)) )
#plt.loglog(k_i[1:30], (squared_abs_h_k_all[1:30]), "-o", label="all")
#plt.loglog(k_i[1:5], 0.068*(1/(k_i[1:5]**2)))
plt.loglog(k_i[1:10], (squared_abs_h_k_down[1:10]), "-o", label='right_side_down')
plt.loglog(k_i[1:5], pot_down[0]*(1/(k_i[1:5]**2))* np.exp(-(k_i[1:5]**2)/2*(pot_down[1]**2)))

plt.loglog(k_i[1:15], (squared_abs_h_k_all[1:15]), "-o", color ='green', linewidth=1)
plt.loglog(k_i[1:5], pot_g[0]*(1/(k_i[1:5]**2))* np.exp(-(k_i[1:5]**2)/2*(pot_g[1]**2)), color='black',  linewidth=1)
plt.ylabel(r'$\left \langle  |\delta h_k|^2  \right \rangle L \quad [nm^{3}]$')
plt.xlabel(r'$k= \frac{2\pi m}{L} \quad [nm^{-1}]$')
plt.savefig(output_dir +'capillary_wave_tribello.png', bbox_inches='tight', transparent=True, dpi=300)

plt.loglog(k_i[1:10], (squared_abs_h_k_up[1:10]), "-o", label='right_side_up')
plt.loglog(k_i[1:10], pot_up[0]*(1/(k_i[1:10]**2))* np.exp(-(k_i[1:10]**2)/2*(pot_up[1]**2)))

plt.loglog(k_i[1:5], 0.075*(1/(k_i[1:5]**2))* np.exp(-(k_i[1:5]**2)/2*(0.55**2)) )

plt.legend()

plt.loglog(k_i[1:10], pot_g*(1/(k_i[1:10]**2)) )

# with Tribello and Ceriotti method
gamma = (Kb * T/ Ly_i)* (1/ pot_g[0]) * (10**9) #pN
gamma_up = (Kb * T/ Ly_i)* (1/ pot_up[0]) * (10**9)
gamma_down = (Kb * T/ Ly_i)* (1/ pot_down[0]) * (10**9)

plt.plot(k_i[1:30], (squared_abs_h_k_all[1:30]))
plt.plot(k_i[1:5], 0.08130541*(1/(k_i[1:5]**2))* np.exp(-(k_i[1:5]**2)/2*(pot_g[1]**2)) ) #* np.exp(-(k_i[1:5]**2)/2*(pot_g[1]**2))
plt.plot(k_i[1:5], 0.06888*(1/(k_i[1:5]**2)))
