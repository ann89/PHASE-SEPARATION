#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 11:45:44 2019

@author: anna
This script use the MDAnlysis wrapper to write an .xtc trajectory out of an atoms selection. 
The same can be achieved using gromacs and a proper index file.
"""

import MDAnalysis
import os
import numpy as np
# =============================================================================
""" path to trajectory """
top  = '../trajectory_every1ns_from7.5us.gro'
traj = '../trajectory_every100ns_from7.5us.xtc'
# 
# 
u = MDAnalysis.Universe(top,traj)
# =============================================================================
dirName = 'ANALYSIS' 
try:
    # Create target Directory
    os.mkdir(dirName)
    print("Directory " , dirName ,  " Created ") 
except FileExistsError:
    print("Directory " , dirName ,  " already exists")



trj = 'ANALYSIS/selection.xtc'
pdb = 'ANALYSIS/selection.pdb'
gro = 'ANALYSIS/selection.gro'
#Definition of the atoms to consider # 

chain_C_DAPC_DLIPC_DSPC_1 = 'or name C22 or name C23 or name C24 or name C25 or name C26 or name C27\
                or name C28 or name C29 or name C210 or name C211 or name C212 or name C213 or name C214\
                or name C215 or name C216  or name C217 or name C218 or name 0C210 or name 1C21 or name 2C21 or name 3C21\
                or name 4C21 or name 5C21 or name 6C21 or name 7C21 or name 8C21'
                
chain_C_DAPC_DLIPC_DSPC_2 = 'or name C32 or name C33 or name C34 or name C35 or name C36 or name C37\
                or name C38 or name C39 or name C310 or name C311 or name C312 or name C313 or name C314\
                or name C315 or name C316  or name C317 or name C318 or name 0C31 or name 1C31 or name 2C31 or name 3C31\
                or name 4C31 or name 5C31 or name 6C31 or name 7C31 or name 8C31' 
                
chain_H_DAPC_DLIPC_DSPC_1 = 'or name H2R or name H3R or name H4R or name H5R or name H6R or name H7R or name H8R\
                             or name H8R or name H9R or name H10R or name H11R or name H12R or name H13R or name H14R or name H15R or name H16R\
                             or name H17R or name H18R or name RH10 or name RH11 or name RH12 or name RH13 or name RH14 or name RH15 or name RH16\
                             or name RH17 or name RH18'
chain_H_DAPC_DLIPC_DSPC_2 = 'or name H2X or name H3X or name H4X or name H5X or name H6X or name H7X or name H8X\
                             or name H8X or name H9X or name H10X or name H11X or name H12X or name H13X or name H14X or name H15X or name H16X\
                             or name H17X or name H18X or name XH10 or name XH11 or name XH12 or name XH13 or name XH14 or name XH15 or name XH16\
                             or name XH17 or name XH18'  
        

                
chain_C_SSM_1 = 'or name C2S  or name C3S or name C4S or name C5S or name C6S or name C7S or name C8S or name C9S or name C10S\
                 or name C11S or name C12S or name C13S or name C14S or name C15S or name C16S or name C17S or name C18S\
                 or name SC10 or name SC11 or name SC12 or name SC13 or name SC14 or name SC15 or name SC16 or name SC17\
                 or name SC18'
                 
chain_C_SSM_2 = 'or name C2F  or name C3F or name C4F or name C5F or name C6F or name C7F or name C8F or name C9F or name C10F\
                 or name C11F or name C12F or name C13F or name C14F or name C15F or name C16F or name C17F or name C18F\
                 or name FC10 or name FC11 or name FC12 or name FC13 or name FC14 or name FC15 or name FC16 or name FC17\
                 or name FC18'
                 
chain_H_SSM_1 = 'or name H2S or name H3S or name H4S or name H5S or name H6S or name H7S or name H8S\
                 or name H8S or name H9S or name H10S or name H11S or name H12S or name H13S or name H14S or name H15S or name H16S\
                 or name H17S or name H18S or name SH10 or name SH11 or name SH12 or name SH13 or name SH14 or name SH15 or name SH16\
                 or name SH17 or name SH18'               
                 
chain_H_SSM_2 = 'or name H2F or name H3F or name H4F or name H5F or name H6F or name H7F or name H8F\
                 or name H8F or name H9F or name H10F or name H11F or name H12F or name H13F or name H14F or name H15F or name H16F\
                 or name H17F or name H18F or name FH10 or name FH11 or name FH12 or name FH13 or name FH14 or name FH15 or name FH16\
                 or name FH17 or name FH18'                  
                
                
                
                
DAPC_selection ='(resname DAPC and (name P or name C24 or name 6C21 or name C3 or name C2 or name C21\
                                     or name C34 or name 6C31 or name C26 or name C27 or name C28 \
                                     or name C29 or name C36 or name C37 or name C38 or name C39'\
                                     + ' ' + str(chain_C_DAPC_DLIPC_DSPC_1) + ' ' +str(chain_C_DAPC_DLIPC_DSPC_2)\
                                     + ' ' + str(chain_H_DAPC_DLIPC_DSPC_1) +' ' + str(chain_H_DAPC_DLIPC_DSPC_2) + ' ))'


DLIPC_selection ='(resname DLIPC and (name P or name C22 or name 6C21 or   name C3 or name C2 or name C21\
                                     or name C32 or name 6C31 or name C26 or name C27 or name C28 \
                                     or name C29 or name C36 or name C37 or name C38 or name C39'\
                                     + ' ' + str(chain_C_DAPC_DLIPC_DSPC_1) + ' ' +str(chain_C_DAPC_DLIPC_DSPC_2)\
                                     + ' ' + str(chain_H_DAPC_DLIPC_DSPC_1) +' ' + str(chain_H_DAPC_DLIPC_DSPC_2) + ' ))'                                     

# # 
DSPC_selection = '(resname DSPC and (name P or name C24 or name 6C21 or  name C3 or name C2 or name C21\
                                     or name C34 or name 6C31 or name C26 or name C27 or name C28 \
                                     or name C29 or name C36 or name C37 or name C38 or name C39'\
                                     + ' ' + str(chain_C_DAPC_DLIPC_DSPC_1) + ' ' +str(chain_C_DAPC_DLIPC_DSPC_2)\
                                     + ' ' + str(chain_H_DAPC_DLIPC_DSPC_1) +' ' + str(chain_H_DAPC_DLIPC_DSPC_2) + ' ))'
# # 
SSM_selection = '(resname SSM and (name P or name C6S or name C16S or name C3S or name C2S or name C1F\
                                     or name C6F or name C16F or name C8S or name C9S or name C10S \
                                     or name C11S or name C8F or name C9F or name C10F or name C11F'\
                                     + ' ' + str(chain_C_SSM_1) + ' ' +str(chain_C_SSM_2)\
                                     + ' ' + str(chain_H_SSM_1) +' ' + str(chain_H_SSM_2) + ' ))'                                     

CHL_selection = '(resname CHL and (name O2 or name C1 or name C65 or name C24))'
 


resnames  = np.unique(u.select_atoms('all and name P or name O2').resnames)

if len(resnames) > 3 :
    if len(resnames [resnames =='DLIPC']) > 0:
        selection = u.select_atoms( DLIPC_selection + str('or ') + DSPC_selection + str('or ') + SSM_selection + str('or ') + CHL_selection) 
    else:
        selection = u.select_atoms( DAPC_selection + str('or ') + DSPC_selection + str('or ') + SSM_selection + str('or ') + CHL_selection) 
else:
    if len(resnames [resnames =='DLIPC']) > 0:
        selection = u.select_atoms( DLIPC_selection + str('or ') + DSPC_selection +  str('or ') + CHL_selection) 
    else:
        selection = u.select_atoms( DAPC_selection + str('or ') + DSPC_selection +  str('or ') + CHL_selection) 
     

            

with MDAnalysis.Writer(pdb, multiframe=True, bonds=None, n_atoms=selection.n_atoms) as PDB:
      for ts in [u.trajectory[0]]:
          PDB.write(selection)
          
with MDAnalysis.Writer(gro, reindex=False, bonds=None, n_atoms=selection.n_atoms) as w:
      for ts in [u.trajectory[0]]:
          w.write(selection)   
          
          
with MDAnalysis.Writer(trj, multiframe=True, bonds=None, n_atoms=selection.n_atoms) as XTC:
      for ts in u.trajectory:
          XTC.write(selection)


print('END: Write trajectory, pdb and gro files of selected atoms')


