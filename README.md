# StarV
Event-driven Monitoring and Verification Codesign for Learning-enabled Cyber-Physical Systems with Star Reachability
 - Operating System: Ubuntu 18, 20
 - Python version: 3.0+
 - Independencies: gurobipy, glpk, polytope, pypoman, tabulate, mathplotlib
 
# Dependencies installation
 
## Install gurobipy: 
  
    a) use the following command for python 3.0+
  
        python -m pip install - https://pypi.gurobi.com gurobipy
    
    b) obtain the relevant license and activate using grbgetkey (have to download gurobi install files from website to access    grbgetkey as that's not installed using pip
    
    c) copy the gurobi.lic file wherever you initially installed it to the following directory: [your python dir]/site-packages/gurobipy/.libs **note there is an existing restricted install license in the directory, simply replace it.
    
## Install glpk: 
   
        pip3 install glpk
        notes: error may come: ERROR: could not build wheels for glpk which use PEP 517 and cannot be installed directly
        sollution: sudo apt install libglpk-dev python3.8-dev libgmp3-dev, pip install glpk
   
## Install polytope: (polytope operations)
        
        pip install polytope
   
## Install pypoman: (plot star sets) 
   
        pip install pypoman 
     
## Install tabulate: (print latex table)

        pip install tabulate
        
## Install mathplotlib: 

        pip install mathplotlib
        
        
        
# Artifacts 

## HSCC2023: Quantitative Verification of Neural Networks using ProbStars
