#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# """
#     @author: anna
#     Calculate the center of mass of the ("phase separated" aggregated )system using the method described in
#     L. Bai and D. Breen, J. Graphics, GPU, Game Tools 13, 53 (2008)
#     http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.151.8565&rep=rep1&type=pdf
#     
#     Let's consider a system of N particles of
#     equal mass in a one-dimensional periodic system of size x_max == Lx.
#     Naturally, for three-dimensional systems, the calculations have
#     to be performed for each dimension separately.
#     We start by mapping each coordinate to an angle:
#         \begin{equation}
#         \theta_i = 2\pi \frac{x_i}{L_x}
#         \end {equation}
#         
#     Then, this angle is interpreted to be on a unit circle, and the
#     corresponding two-dimensional coordinates are calculated,
#     \begin{equation}
#     \xi_i = cos(\theta_i)
#     \zeta_i = sin(\theta_i)
#     \end{equation}
#     
#     Now, we calculate the standard center of mass in the twodimensional space,
#     \begin{equation}
#     \bar \xi_i = \frac{1}{N}\sum_{i=1}^N \xi_i
#     \bar \zeta_i  =  \frac{1}{N}\sum_{i=1}^N \zeta_i
#     \end{equation}
#     Finally, we can map back the common center to an angle
#     \begin{equation}
#     \bat \theta = atan2(-\bar \zeta, -\bar \xi ) + \pi
#     \end{equation}
#     and then map back this average angle to a length coordinate
#     in the range $[0, L_x)$ 
#         \begin{equation}
#         x_{COM} = L_x \frac{\bat \theta}{2\pi}
#         \end{equation}
#     . The negation of the arguments in
#     combination with the shift of the function by $\pi$ ensures that $\theta$
#     falls within $[0,2\pi)$.
#     
#     The algorithm is completely unambiguous, even for cases where the mass
#     distribution is wide in comparison to the periodic box. This
#     is not true when trying to calculate the center of mass for
#     such a case using the usual minimum image convention with
#     respect to some more or less random reference point. The
#     algorithm will only fail in the case of a completely uniform
#     mass distribution, for which the center of mass is not defined
#     in a periodic system. Even then, the algorithm will return some
#     value (depending on the implementation of atan2), which is
#     as good as any other for that particular situation (This is for instance the case for the 
#     y direction in our membrane systems).
#     
#     Once the position of the center of mass is known, 
#     fold back the  coordinate in the box and write down the trajectory.
# """
# 
# =============================================================================
import math
import MDAnalysis
import numpy as np

top = 'ANALYSIS/selection.pdb'
traj = 'ANALYSIS/selection.xtc' 

u = MDAnalysis.Universe(top,traj)

def getCMx(xpos, Lx):
   
    PI= math.pi
    
    radius_i = Lx/ (2. * PI)
    
    x_cm_i = np.mean(radius_i * np.cos(xpos / Lx * 2. * PI), axis=0)
    z_cm_i = np.mean(radius_i * np.sin(xpos / Lx * 2. * PI), axis=0)

    return (np.arctan2(-z_cm_i, -x_cm_i) + PI) / (2. * PI) * Lx

def set_pos_back_to_box(res_idxs, CMx, Lx, pos_x_all_new):
    pos_x_all_new = pos_x_all_new.copy()
    for i in range(len(res_idxs)):
        idx = res_idxs[i]
        pos_res = u.select_atoms('resnum %i'%idx).center_of_geometry()
        if pos_res[0] -  CMx + (0.5 * Lx) >= Lx:
            #print('True')
            idx_atoms = u.select_atoms('resnum %i'%idx).ix
            pos_x_all_new[idx_atoms] = pos_x_all_new[idx_atoms] - Lx 
        elif pos_res[0] -  CMx + (0.5 * Lx) < 0:
            idx_atoms = u.select_atoms('resnum %i'%idx).ix
            pos_x_all_new[idx_atoms] = pos_x_all_new[idx_atoms] + Lx
            #print('second')
    return pos_x_all_new


trj = "ANALYSIS/recentered_x.xtc"  
pdb = "ANALYSIS/recentered_x.pdb" 
gro = "ANALYSIS/recentered_x.gro" 

resnames  = np.unique(u.select_atoms('all and name P or name O2').resnames)
with MDAnalysis.Writer(trj, multiframe=True, bonds=None, n_atoms=u.atoms.n_atoms) as XTC:
    

    for ts in u.trajectory:
        if len(resnames) > 3 :
            if len(resnames [resnames =='DLIP']) > 0:            
                pos_x_all = u.select_atoms('all').positions[:,0]
                pos_DLIPC_x = u.select_atoms('resname DLIP').positions[:,0] #to put DLIP in the middle    
                Lx = u.trajectory.ts.triclinic_dimensions[0][0]
                CMx = getCMx(pos_DLIPC_x[:],Lx)
                pos_x_all_new = pos_x_all -  CMx + (0.5 * Lx)
                DIPC_idxs = u.select_atoms('resname DLIP and name P').resnums
                pos_x_all_new = set_pos_back_to_box(DIPC_idxs, CMx, Lx, pos_x_all_new)              
                DSPC_idxs = u.select_atoms('resname DSPC and name P').resnums
                pos_x_all_new = set_pos_back_to_box(DSPC_idxs, CMx, Lx, pos_x_all_new)          
                SSM_idxs = u.select_atoms('resname SSM and name P').resnums
                pos_x_all_new = set_pos_back_to_box(SSM_idxs, CMx, Lx, pos_x_all_new)                
                CHL_idxs = u.select_atoms('resname CHL and name O2').resnums
                pos_x_all_new = set_pos_back_to_box(CHL_idxs, CMx, Lx, pos_x_all_new)
            else : 
                pos_x_all = u.select_atoms('all').positions[:,0]
                pos_DLIPC_x = u.select_atoms('resname DAPC').positions[:,0] #to put DLIP in the middle    
                Lx = u.trajectory.ts.triclinic_dimensions[0][0]
                CMx = getCMx(pos_DLIPC_x[:],Lx)
                pos_x_all_new = pos_x_all -  CMx + (0.5 * Lx)
                DIPC_idxs = u.select_atoms('resname DAPC and name P').resnums
                pos_x_all_new = set_pos_back_to_box(DIPC_idxs, CMx, Lx, pos_x_all_new)              
                DSPC_idxs = u.select_atoms('resname DSPC and name P').resnums
                pos_x_all_new = set_pos_back_to_box(DSPC_idxs, CMx, Lx, pos_x_all_new)          
                SSM_idxs = u.select_atoms('resname SSM and name P').resnums
                pos_x_all_new = set_pos_back_to_box(SSM_idxs, CMx, Lx, pos_x_all_new)                
                CHL_idxs = u.select_atoms('resname CHL and name O2').resnums
                pos_x_all_new = set_pos_back_to_box(CHL_idxs, CMx, Lx, pos_x_all_new) 
        else:
            if len(resnames [resnames =='DLIP']) > 0:
                pos_x_all = u.select_atoms('all').positions[:,0]
                pos_DLIPC_x = u.select_atoms('resname DLIP').positions[:,0] #to put DLIP in the middle    
                Lx = u.trajectory.ts.triclinic_dimensions[0][0]
                CMx = getCMx(pos_DLIPC_x[:],Lx)
                pos_x_all_new = pos_x_all -  CMx + (0.5 * Lx)
                DIPC_idxs = u.select_atoms('resname DLIP and name P').resnums
                pos_x_all_new = set_pos_back_to_box(DIPC_idxs, CMx, Lx, pos_x_all_new)              
                DSPC_idxs = u.select_atoms('resname DSPC and name P').resnums
                pos_x_all_new = set_pos_back_to_box(DSPC_idxs, CMx, Lx, pos_x_all_new)                          
                CHL_idxs = u.select_atoms('resname CHL and name O2').resnums
                pos_x_all_new = set_pos_back_to_box(CHL_idxs, CMx, Lx, pos_x_all_new)
            else:
                pos_x_all = u.select_atoms('all').positions[:,0]
                pos_DLIPC_x = u.select_atoms('resname DAPC').positions[:,0] #to put DLIP in the middle    
                Lx = u.trajectory.ts.triclinic_dimensions[0][0]
                CMx = getCMx(pos_DLIPC_x[:],Lx)
                pos_x_all_new = pos_x_all -  CMx + (0.5 * Lx)
                DIPC_idxs = u.select_atoms('resname DAPC and name P').resnums
                pos_x_all_new = set_pos_back_to_box(DIPC_idxs, CMx, Lx, pos_x_all_new)              
                DSPC_idxs = u.select_atoms('resname DSPC and name P').resnums
                pos_x_all_new = set_pos_back_to_box(DSPC_idxs, CMx, Lx, pos_x_all_new)                          
                CHL_idxs = u.select_atoms('resname CHL and name O2').resnums
                pos_x_all_new = set_pos_back_to_box(CHL_idxs, CMx, Lx, pos_x_all_new)                
                
        foo1 = np.asarray([pos_x_all_new, u.select_atoms('all').positions[:,1], 
                                       u.select_atoms('all').positions[:,2]])
        foo2=foo1.transpose()
    
        u.atoms.positions = foo2
        
        XTC.write(u.atoms)
                



with MDAnalysis.Writer(pdb, multiframe=True, bonds=None, n_atoms=u.atoms.n_atoms) as PDB:
    for ts in [u.trajectory[0]]:
        if len(resnames) > 3 :
            if len(resnames [resnames =='DLIP']) > 0:            
                pos_x_all = u.select_atoms('all').positions[:,0]
                pos_DLIPC_x = u.select_atoms('resname DLIP').positions[:,0] #to put DLIP in the middle    
                Lx = u.trajectory.ts.triclinic_dimensions[0][0]
                CMx = getCMx(pos_DLIPC_x[:],Lx)
                pos_x_all_new = pos_x_all -  CMx + (0.5 * Lx)
                DIPC_idxs = u.select_atoms('resname DLIP and name P').resnums
                pos_x_all_new = set_pos_back_to_box(DIPC_idxs, CMx, Lx, pos_x_all_new)              
                DSPC_idxs = u.select_atoms('resname DSPC and name P').resnums
                pos_x_all_new = set_pos_back_to_box(DSPC_idxs, CMx, Lx, pos_x_all_new)          
                SSM_idxs = u.select_atoms('resname SSM and name P').resnums
                pos_x_all_new = set_pos_back_to_box(SSM_idxs, CMx, Lx, pos_x_all_new)                
                CHL_idxs = u.select_atoms('resname CHL and name O2').resnums
                pos_x_all_new = set_pos_back_to_box(CHL_idxs, CMx, Lx, pos_x_all_new)
            else : 
                pos_x_all = u.select_atoms('all').positions[:,0]
                pos_DLIPC_x = u.select_atoms('resname DAPC').positions[:,0] #to put DLIP in the middle    
                Lx = u.trajectory.ts.triclinic_dimensions[0][0]
                CMx = getCMx(pos_DLIPC_x[:],Lx)
                pos_x_all_new = pos_x_all -  CMx + (0.5 * Lx)
                DIPC_idxs = u.select_atoms('resname DAPC and name P').resnums
                pos_x_all_new = set_pos_back_to_box(DIPC_idxs, CMx, Lx, pos_x_all_new)              
                DSPC_idxs = u.select_atoms('resname DSPC and name P').resnums
                pos_x_all_new = set_pos_back_to_box(DSPC_idxs, CMx, Lx, pos_x_all_new)          
                SSM_idxs = u.select_atoms('resname SSM and name P').resnums
                pos_x_all_new = set_pos_back_to_box(SSM_idxs, CMx, Lx, pos_x_all_new)                
                CHL_idxs = u.select_atoms('resname CHL and name O2').resnums
                pos_x_all_new = set_pos_back_to_box(CHL_idxs, CMx, Lx, pos_x_all_new) 
        else:
            if len(resnames [resnames =='DLIP']) > 0:
                pos_x_all = u.select_atoms('all').positions[:,0]
                pos_DLIPC_x = u.select_atoms('resname DLIP').positions[:,0] #to put DLIP in the middle    
                Lx = u.trajectory.ts.triclinic_dimensions[0][0]
                CMx = getCMx(pos_DLIPC_x[:],Lx)
                pos_x_all_new = pos_x_all -  CMx + (0.5 * Lx)
                DIPC_idxs = u.select_atoms('resname DLIP and name P').resnums
                pos_x_all_new = set_pos_back_to_box(DIPC_idxs, CMx, Lx, pos_x_all_new)              
                DSPC_idxs = u.select_atoms('resname DSPC and name P').resnums
                pos_x_all_new = set_pos_back_to_box(DSPC_idxs, CMx, Lx, pos_x_all_new)                          
                CHL_idxs = u.select_atoms('resname CHL and name O2').resnums
                pos_x_all_new = set_pos_back_to_box(CHL_idxs, CMx, Lx, pos_x_all_new)
            else:
                pos_x_all = u.select_atoms('all').positions[:,0]
                pos_DLIPC_x = u.select_atoms('resname DAPC').positions[:,0] #to put DLIP in the middle    
                Lx = u.trajectory.ts.triclinic_dimensions[0][0]
                CMx = getCMx(pos_DLIPC_x[:],Lx)
                pos_x_all_new = pos_x_all -  CMx + (0.5 * Lx)
                DIPC_idxs = u.select_atoms('resname DAPC and name P').resnums
                pos_x_all_new = set_pos_back_to_box(DIPC_idxs, CMx, Lx, pos_x_all_new)              
                DSPC_idxs = u.select_atoms('resname DSPC and name P').resnums
                pos_x_all_new = set_pos_back_to_box(DSPC_idxs, CMx, Lx, pos_x_all_new)                          
                CHL_idxs = u.select_atoms('resname CHL and name O2').resnums
                pos_x_all_new = set_pos_back_to_box(CHL_idxs, CMx, Lx, pos_x_all_new)                
                
        foo1 = np.asarray([pos_x_all_new, u.select_atoms('all').positions[:,1], 
                                       u.select_atoms('all').positions[:,2]])
        foo2=foo1.transpose()
    
        u.atoms.positions = foo2
        
        PDB.write(u.atoms)
                

with MDAnalysis.Writer(gro, reindex=False , bonds=None, n_atoms=u.atoms.n_atoms) as w:
    for ts in [u.trajectory[0]]:
        if len(resnames) > 3 :
            if len(resnames [resnames =='DLIP']) > 0:            
                pos_x_all = u.select_atoms('all').positions[:,0]
                pos_DLIPC_x = u.select_atoms('resname DLIP').positions[:,0] #to put DLIP in the middle    
                Lx = u.trajectory.ts.triclinic_dimensions[0][0]
                CMx = getCMx(pos_DLIPC_x[:],Lx)
                pos_x_all_new = pos_x_all -  CMx + (0.5 * Lx)
                DIPC_idxs = u.select_atoms('resname DLIP and name P').resnums
                pos_x_all_new = set_pos_back_to_box(DIPC_idxs, CMx, Lx, pos_x_all_new)              
                DSPC_idxs = u.select_atoms('resname DSPC and name P').resnums
                pos_x_all_new = set_pos_back_to_box(DSPC_idxs, CMx, Lx, pos_x_all_new)          
                SSM_idxs = u.select_atoms('resname SSM and name P').resnums
                pos_x_all_new = set_pos_back_to_box(SSM_idxs, CMx, Lx, pos_x_all_new)                
                CHL_idxs = u.select_atoms('resname CHL and name O2').resnums
                pos_x_all_new = set_pos_back_to_box(CHL_idxs, CMx, Lx, pos_x_all_new)
            else : 
                pos_x_all = u.select_atoms('all').positions[:,0]
                pos_DLIPC_x = u.select_atoms('resname DAPC').positions[:,0] #to put DLIP in the middle    
                Lx = u.trajectory.ts.triclinic_dimensions[0][0]
                CMx = getCMx(pos_DLIPC_x[:],Lx)
                pos_x_all_new = pos_x_all -  CMx + (0.5 * Lx)
                DIPC_idxs = u.select_atoms('resname DAPC and name P').resnums
                pos_x_all_new = set_pos_back_to_box(DIPC_idxs, CMx, Lx, pos_x_all_new)              
                DSPC_idxs = u.select_atoms('resname DSPC and name P').resnums
                pos_x_all_new = set_pos_back_to_box(DSPC_idxs, CMx, Lx, pos_x_all_new)          
                SSM_idxs = u.select_atoms('resname SSM and name P').resnums
                pos_x_all_new = set_pos_back_to_box(SSM_idxs, CMx, Lx, pos_x_all_new)                
                CHL_idxs = u.select_atoms('resname CHL and name O2').resnums
                pos_x_all_new = set_pos_back_to_box(CHL_idxs, CMx, Lx, pos_x_all_new) 
        else:
            if len(resnames [resnames =='DLIP']) > 0:
                pos_x_all = u.select_atoms('all').positions[:,0]
                pos_DLIPC_x = u.select_atoms('resname DLIP').positions[:,0] #to put DLIP in the middle    
                Lx = u.trajectory.ts.triclinic_dimensions[0][0]
                CMx = getCMx(pos_DLIPC_x[:],Lx)
                pos_x_all_new = pos_x_all -  CMx + (0.5 * Lx)
                DIPC_idxs = u.select_atoms('resname DLIP and name P').resnums
                pos_x_all_new = set_pos_back_to_box(DIPC_idxs, CMx, Lx, pos_x_all_new)              
                DSPC_idxs = u.select_atoms('resname DSPC and name P').resnums
                pos_x_all_new = set_pos_back_to_box(DSPC_idxs, CMx, Lx, pos_x_all_new)                          
                CHL_idxs = u.select_atoms('resname CHL and name O2').resnums
                pos_x_all_new = set_pos_back_to_box(CHL_idxs, CMx, Lx, pos_x_all_new)
            else:
                pos_x_all = u.select_atoms('all').positions[:,0]
                pos_DLIPC_x = u.select_atoms('resname DAPC').positions[:,0] #to put DLIP in the middle    
                Lx = u.trajectory.ts.triclinic_dimensions[0][0]
                CMx = getCMx(pos_DLIPC_x[:],Lx)
                pos_x_all_new = pos_x_all -  CMx + (0.5 * Lx)
                DIPC_idxs = u.select_atoms('resname DAPC and name P').resnums
                pos_x_all_new = set_pos_back_to_box(DIPC_idxs, CMx, Lx, pos_x_all_new)              
                DSPC_idxs = u.select_atoms('resname DSPC and name P').resnums
                pos_x_all_new = set_pos_back_to_box(DSPC_idxs, CMx, Lx, pos_x_all_new)                          
                CHL_idxs = u.select_atoms('resname CHL and name O2').resnums
                pos_x_all_new = set_pos_back_to_box(CHL_idxs, CMx, Lx, pos_x_all_new)                
                
        foo1 = np.asarray([pos_x_all_new, u.select_atoms('all').positions[:,1], 
                                       u.select_atoms('all').positions[:,2]])
        foo2=foo1.transpose()
    
        u.atoms.positions = foo2
        
        w.write(u.atoms)
                
print('Re-centered trajectory written')