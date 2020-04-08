#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 11:12:29 2019

@author: anna
Import the tilt and splay angle data.
make an histogram and fit the histogram with a gaussian.
use the range of the gaussian to fit the PMF
Compute the Kt and Kc as described in Phys. Chem. Chem. Phys. 2017, 19, 16806.

Additional Arguments: Estimate of apl of the disordered and ordered phase
"""
import numpy as np
from numpy import trapz
from scipy.optimize import curve_fit
import MDAnalysis
import matplotlib.pyplot as plt
import sys

apl_Ld = 0.6 #sys.argv[1]
apl_Lo = 0.4 #sys.argv[2]

top  = 'ANALYSIS/recentered_x.gro'
traj = 'ANALYSIS/recentered_x.xtc'
u = MDAnalysis.Universe(top,traj)



def _gauss(x, *p):
        A, mu, sigma = p
        return A * np.exp(-(x - mu)**2 / (2. * sigma**2))
    
def _FitGaussian(bincenters, pa):
      mu0 = np.sum(bincenters * pa) / np.sum(pa)
      A0 = np.max(pa)             
      sigma0 = np.sqrt(np.sum(((bincenters - mu0)**2.0) * pa) / np.sum(pa))
#      sigma0 = 0.1
      #print(mu0, A0, sigma0)
      (A, mu, sigma), v = curve_fit(_gauss, bincenters, pa, [A0, mu0, sigma0])
      return A, mu, abs(sigma)
  
    
    

# 
# =============================================================================
def _parabole(x, a, b, x0):
    return  a + (b) * (x-x0)**2.0

def first_quadrant(x):
    if (x >= 90) :
        x= 180 - x
    else:
        x= x
    return x

def _FindIndexOfClosestValue(l, v):
    return min(enumerate(l), key=lambda x: abs(x[1] - v))[0]

def _FitParabole(bincenters, fa, fitting_range):
    first = _FindIndexOfClosestValue(bincenters, fitting_range[0])
    last = _FindIndexOfClosestValue(bincenters, fitting_range[1])
  
    mask = fa != np.inf
    a = min(fa)
    x0 = bincenters[np.argmin(fa)] #argmin return the indices of minimum value 
    xm = bincenters[mask][np.argmax(fa[mask])]
    fm = max(fa[mask])
    b = (fm - a) / (xm - x0)**2.0
    r, v = curve_fit(_parabole, bincenters[first:last], fa[
                         first:last], [a, b, x0])
    return r


def splay_modulus( leaflet, angles_in_radians, area_per_lipid, status,nbins=100, Plot=True):
    
    """ compute the distribution of splay angles using an histogram """
    histo, bins = np.histogram(angles_in_radians, bins= nbins , density=True) #bins=len(angles_first_quadrant)
    bincenters = 0.5 * (bins[1:] + bins[:-1])
    
    if status == "disordered":
        cutoff = 35
        g_range_sa = np.where(bincenters < np.radians(cutoff))[0]
        A, mu, sigma = _FitGaussian(bincenters[g_range_sa], histo[g_range_sa])
        
    else: 
        cutoff = 20
        g_range_sa = np.where(bincenters < np.radians(cutoff))[0]
        A, mu, sigma = _FitGaussian(bincenters[g_range_sa], histo[g_range_sa])
        #plt.plot(bincenters_Lo, _gauss(bincenters_Lo, Ao, muo, sigmao ))
        
    y=np.sin(bincenters)
    Area=trapz(y, bincenters)
    sin_normalized=y/Area
    #plt.plot(bincenters_Ld, sin_normalized)    
    """ normlize the probability with the sin(theta) """
    pa2 = histo / sin_normalized  
    """ PMF in KbT units """
    PMF = -np.log(pa2)
    #plt.plot(bincenters, PMF)


    ranges = [ (max(mu - i * sigma, 0), mu + i * sigma) 
                  for i in [ 1, 1.25, 1.5, 1.75, 2.0]] 
    
    print ("Using the following ranges to fit the PMF:", ranges)
    res_list = [_FitParabole(bincenters, PMF, fitting_range)
                    for fitting_range in ranges]

    K_list = [(2. * r[1])/ area_per_lipid for r in res_list]
    DeltaK = np.std(K_list)
    K = K_list[0]
    
    
    if Plot: 
        fig, ax = plt.subplots(3, 1, sharex=True, sharey=False)
        fig.subplots_adjust(hspace=0.05, wspace=0.05)
        ax[0].fill_between(bincenters, _gauss(bincenters,A, mu, sigma), alpha=0.5)
        ax[0].plot(bincenters, histo)  
        xcoords = [mu - sigma, mu, mu + sigma]
        for xc in xcoords:
            ax[0].axvline(x=xc, linestyle='--')

        ax[1].plot(bincenters, pa2,'-')
        ax[2].plot(bincenters, PMF,'-')
        ax[2].plot(bincenters, _parabole(bincenters, res_list[0][0],res_list[0][1], res_list[0][2] ), 'g--', label =r'$k$ = %3.1f $\pm$  %3.1f [$k_BT$]' %(K,DeltaK ))
        ax[2].grid('True')
        plt.xlim(0,np.pi/2)
        plt.legend()
        plt.savefig('ANALYSIS/tilts_local_normals/Splay_modulus_'+ str(leaflet)+ '_' + str(status) +'.png', dpi=300)
        plt.savefig('ANALYSIS/tilts_local_normals/Splay_modulus_'+ str(leaflet)+ '_' + str(status) +'.svg')
    return K, DeltaK, K_list
   
    

def tilt_modulus( leaflet, angles_in_radians, status, nbins=100,  Plot=True):
    """ 
    It will first fit a gaussian y=A exp[(x-mu)/sigma^2] to the distribution of tilts 
    to determine the fitting range then used to fit the corresponding potential of mean force (PMF).
    Different fitting ranges are used to estimate the error on the extracted tilt modulus.
    The function will calculate one tilt modulus for each lipid species and one splay modulus for each pair
    of lipid species. It will then combine these to calculate the overall tilt modulus and splay modulus (bending rigidity).
    More details about this procedure can be found in ref. [2]_ 
    """
    
    
    """ set the angles in range [0,90] degrees """
    angles_in_degree = np.degrees(angles_in_radians)   #all_tilts[disordered_indx]
    angles_first_quadrant = np.array([first_quadrant(x) for x in angles_in_degree])

    
    """ compute the distribution of tilt angles using an histogram """    
    histo, bins = np.histogram(np.radians(angles_first_quadrant), bins= nbins , density=True) #bins=len(angles_first_quadrant)
    bincenters = 0.5 * (bins[1:] + bins[:-1])
    
    if status == "disordered":
        cutoff = 30
        g_range_sa = np.where(bincenters < np.radians(cutoff))[0]
        A, mu, sigma = _FitGaussian(bincenters[g_range_sa], histo[g_range_sa])
        
    else: 
        cutoff = 30
        g_range_sa = np.where(bincenters < np.radians(cutoff))[0]
        A, mu, sigma = _FitGaussian(bincenters[g_range_sa], histo[g_range_sa])
        #plt.plot(bincenters_Lo, _gauss(bincenters_Lo, Ao, muo, sigmao ))
        
    y=np.sin(bincenters)
    Area=trapz(y, bincenters)
    sin_normalized=y/Area
    #plt.plot(bincenters_Ld, sin_normalized)    
    """ normlize the probability with the sin(theta) """
    pa2 = histo / sin_normalized  
    """ PMF in KbT units """
    PMF = -np.log(pa2)
    #plt.plot(bincenters, PMF)


    ranges = [ (max(mu - i * sigma, 0), mu + i * sigma) 
                  for i in [ 1, 1.25, 1.5, 1.75, 2.0]] 
    
    print ("Using the following ranges to fit the PMF:", ranges)
    res_list = [_FitParabole(bincenters, PMF, fitting_range)
                    for fitting_range in ranges]

    K_list = [(2. * r[1]) for r in res_list]
    DeltaK = np.std(K_list)
    K = K_list[0]
    
    
    if Plot: 
        fig, ax = plt.subplots(3, 1, sharex=True, sharey=False)
        fig.subplots_adjust(hspace=0.05, wspace=0.05)
        ax[0].fill_between(bincenters, _gauss(bincenters,A, mu, sigma), alpha=0.5)
        ax[0].plot(bincenters, histo)  
        xcoords = [mu - sigma, mu, mu + sigma]
        for xc in xcoords:
            ax[0].axvline(x=xc, linestyle='--')
    #ax[0].plot(X_plot[:, 0], _gauss(bincenters,A_test2, mu_test2, test_sigma2 ),'-')
        ax[1].plot(bincenters, pa2,'-')
        ax[2].plot(bincenters, PMF,'-')
        ax[2].plot(bincenters, _parabole(bincenters, res_list[0][0],res_list[0][1], res_list[0][2] ), 'g--', label =r'$k_t$ = %3.1f $\pm$  %3.1f [$k_BT/ nm^2$]' %(K,DeltaK ))
        ax[2].grid('True')
        plt.xlim(0,np.pi/2)
        plt.legend()
        plt.savefig('ANALYSIS/tilts_local_normals/Tilt_modulus_'+ str(leaflet)+ '_' + str(status)+ '.png', dpi=300)
        plt.savefig('ANALYSIS/tilts_local_normals/Tilt_modulus_'+ str(leaflet)+ '_' + str(status)+ '.svg')

    return K, DeltaK, K_list







##======== better using the arctan2 method  ==================================##
def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

import numpy.linalg as la
def compute_angle(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'
        The sign of the angle is dependent on the order of v1 and v2
        so acos(norm(dot(v1, v2))) does not work and atan2 has to be used, see:
        https://stackoverflow.com/questions/21483999/using-atan2-to-find-angle-between-two-vectors
    """
    cosang = np.dot(v1, v2)
    sinang = la.norm(np.cross(v1, v2))
    angle = np.arctan2(sinang, cosang)
    return angle 




def compute_splays(first_neighbors_splay, time, all_tilts_vect_upper):
    angles_splay = np.zeros(( len(first_neighbors_splay[0]), 4))
    time = np.full(len(first_neighbors_splay[0]), time)
    for i in range(len(first_neighbors_splay[0])):        
        angles_splay[i, :] = compute_angle(all_tilts_vect_upper[first_neighbors_splay[0][i]], all_tilts_vect_upper[first_neighbors_splay[1][i]]), first_neighbors_splay[0][i], first_neighbors_splay[1][i], time[i]
    return angles_splay

    

###=======================Main ==============================================######
Kb= 0.0083144621
T =298

input_dir = "ANALYSIS/directors/"
input_tilts_dir = "ANALYSIS/tilts_local_normals/"
input_phase_assignment="ANALYSIS/directors/plots/" 









assigned_up_all = []
assigned_down_all = [] 
leaflet = 'upper' 

import pandas as pd
assignment_up_all = []
assignment_down_all = []

appended_data_up = []
appended_data_down = []
for ts in range (0,u.trajectory.n_frames,1) : 
    infile_up = 'ANALYSIS/directors/Dataframeup'+ str(ts)
    data_up = pd.read_pickle(infile_up)
    # store DataFrame in list
    appended_data_up.append(data_up)
    # see pd.concat documentation for more info
    Data_up = pd.concat(appended_data_up)
    
    infile_down = 'ANALYSIS/directors/Dataframedown'+ str(ts)
    data_down = pd.read_pickle(infile_down)
    # store DataFrame in list
    appended_data_down.append(data_down)
    # see pd.concat documentation for more info
    Data_down = pd.concat(appended_data_down)       
    
    """ read in the Lo/Ld assignment: ATTENTION: for the lipids you have saved the value two times(one time for chain): CLEAN UP!
    taking only one value per chain!
    Assignment : 1 = Lo, 0 = Ld
    """
    assignment_up = np.load(input_phase_assignment + 'resid_phases'+ 'upper' +'.'+ str(ts) + '.npy')    
    assignment_down = np.load(input_phase_assignment + 'resid_phases'+ 'lower' +'.'+ str(ts) + '.npy') 

    chl_res_up = np.load(input_dir + 'cholesterol_'+'upper'+'_tail_' + str(ts) + '.npy')
    dlipc_res_up = np.load(input_dir + 'dlipc_' + 'upper'+'_tail_' + str(ts) + '.npy')  
    dspc_res_up = np.load(input_dir + 'dspc_' + 'upper'+'_tail_'  +  str(ts) + '.npy')
    ssm_res_up = np.load(input_dir + 'ssm_' + 'upper'+'_tail_'  +  str(ts) + '.npy')
    
    chl_res_down = np.load(input_dir + 'cholesterol_'+'lower'+'_tail_' + str(ts) + '.npy')
    dlipc_res_down = np.load(input_dir + 'dlipc_' + 'lower'+'_tail_' + str(ts) + '.npy')  
    dspc_res_down = np.load(input_dir + 'dspc_' + 'lower'+'_tail_'  +  str(ts) + '.npy')
    ssm_res_down = np.load(input_dir + 'ssm_' + 'lower'+'_tail_'  +  str(ts) + '.npy')
    
    
    cleaned_assignment_up = np.vstack((assignment_up[0:len(chl_res_up) + len(dlipc_res_up)],
                                                     assignment_up[len(chl_res_up) + len(dlipc_res_up)*2 : len(chl_res_up) + len(dlipc_res_up)*2 +len(ssm_res_up)],
                                                     assignment_up[len(chl_res_up) + len(dlipc_res_up)*2 + len(ssm_res_up)*2 : len(chl_res_up) + len(dlipc_res_up)*2 +len(ssm_res_up)*2 + len(dspc_res_up)] )) 
    assigned_up_all.append(cleaned_assignment_up) 

    cleaned_assignment_down = np.vstack((assignment_down[0:len(chl_res_down) + len(dlipc_res_down)],
                                                     assignment_down[len(chl_res_down) + len(dlipc_res_down)*2 : len(chl_res_down) + len(dlipc_res_down)*2 +len(ssm_res_down)],
                                                     assignment_down[len(chl_res_down) + len(dlipc_res_down)*2 + len(ssm_res_down)*2 : len(chl_res_down) + len(dlipc_res_down)*2 +len(ssm_res_down)*2 + len(dspc_res_down)] )) 
    assigned_down_all.append(cleaned_assignment_down)     
    
    

    assignment_down_all.append(cleaned_assignment_down)
    
    ass_down_all = np.vstack((assigned_down_all))   
    ass_up_all = np.vstack((assigned_up_all)) 
    

Data_down['Assign'] = ass_down_all[:,1]
Data_up['Assign'] = ass_up_all[:,1]

Data_up_Lo = Data_up[Data_up['Assign'] ==1]
Data_down_Lo = Data_down[Data_down['Assign'] ==1]

Data_up_Ld = Data_up[Data_up['Assign'] ==0]
Data_down_Ld = Data_down[Data_down['Assign'] ==0]    
try:   
    disordered_Kc_up = splay_modulus('up', Data_up_Ld['Splay'].values,  area_per_lipid= apl_Ld, status="disordered", Plot=True, nbins=10 )
except Exception as e:
    print(e)
try:    
    ordered_Kc_up = splay_modulus('up', Data_up_Lo['Splay'].values,  area_per_lipid= apl_Lo, status="ordered", Plot=True, nbins=20 )    
except Exception as e:
    print(e)
try:    
    disordered_Kt_up =  tilt_modulus('up', Data_up_Ld['Tilt_angles'].values , status="disordered", Plot=True, nbins=20 )
except Exception as e:
    print(e)
try:    
    ordered_Kt_up =  tilt_modulus('up', Data_up_Lo['Tilt_angles'].values, status="ordered", Plot=True, nbins=20 )
except Exception as e:
    print(e)
try:    
    disordered_Kc_down = splay_modulus('down', Data_down_Ld['Splay'].values,  area_per_lipid= apl_Ld, status="disordered", Plot=True, nbins=10 )
except Exception as e:
    print(e)

try:
    ordered_Kc_down = splay_modulus('down', Data_down_Lo['Splay'].values,  area_per_lipid= apl_Lo, status="ordered", Plot=True, nbins=20 )    
except Exception as e:
    print(e)
try:    
    disordered_Kt_down =  tilt_modulus('down', Data_down_Ld['Tilt_angles'].values , status="disordered", Plot=True, nbins=20 )
except Exception as e:
    print(e)
try:
    ordered_Kt_down =  tilt_modulus('down', Data_down_Lo['Tilt_angles'].values, status="ordered", Plot=True, nbins=20 )
except Exception as e:
    print(e)


    
    
    
  
 




