#ANALYSIS WORKFLOW
# 0. To check phase separation one could compute the mixing entropy with the simple method
#of G.B. Brandani, M. Schor, C.E. MacPhee, H. Grubmuller, U. Zachariae, & D. Marenduzzo “Quantifying Disorder through Conditional Entropy: An Application to Fluid Mixing,” PLoS One 8 e65617 (2013).

	python mixing_entropy.py up
	python mixing_entropy.py down
 
#OPTIONAL 1. Consider a selection of the membrane system to speed up the subsequent analysis (returns the trajectory of the selected bilayer components and the structure files (.gro and .pdb))
	python Trajectory_selection.py 
# 2. Remove COM translations along the x direction and return a trajectory with the liquid phase centered in the box:
	 python COM-recenter.py	
# 3. Compute Director order parameter, Deuterium Order (average per chain and per carbon), Tilt and Splay angles (as defined in Phy. Chem. Chem. Phys. 2017, 19, 16806 and similar others) to later compute Kc and Kt
	python DirectorDeuteriumTiltSplay_CHL-DLIPC-DSPC-SSM.py up #up for upper leaflet
        python DirectorDeuteriumTiltSplay_CHL-DLIPC-DSPC-SSM.py  down #down for lower leaflet
# 4. Compute contours between Lo and Ld and assign lipids to the respective phases
	python FindContours.py up
	python FindContours.py down 

#From here on, the analysis can be done in the order that one prefers.

# 5. Analyze cholesterol flip flop (return a .csv file with the id of lipids who are  )
     python Flip-flop-Distance-Angle-cutoffs.py

# 6. Compute Bending rigidity and tilt modulus with the method of Khelashvili (Phy. Chem. Chem. Phys. 2017, 19, 16806 and similar others) 
#	an estimate of the apl of the disorder and ordered phases is needed as input 
	python Kc_Kt_moduli.py

7. compute MSD of lipids always in Lo and/or always in Ld phase. Needed  a 'no-jump' trajectory as input and the lipid assignment performed in 4.
	python MSD-phases-Disp_Autocorr.py  

8. Compute Line tension 
	python capillary_contours.py
 


