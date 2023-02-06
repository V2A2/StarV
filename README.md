# StarV
Event-driven Monitoring and Verification Codesign for Distributed Learning-enabled Cyber-Physical Systems with Star Reachability
 - Operating System: Ubuntu 18., 20.
 - Python version: 3.0+
 - Independencies: gurobipy, glpk, polytope, pypoman, tabulate, mathplotlib
 - Key features: 
    1) Qualitative and quantitative verification algorithms for deep neural networks and distributed Le-CPS 
    2) Probabilistic Star Temporal Logic (under development)
    3) Monitoring algorithms (under development)
    4) Target ROS-based applications (under development)
 
 
# StarV installation, tests and artifacts

  - No installation, just clone the repository:
  
        git clone https://github.com/V2A2/StarV
        
  - Go inside StarV and run tests or artifacts
 
# Dependencies installation
 
- Install gurobipy: 
  
    a) use the following command for python 3.0+
  
        python -m pip install - https://pypi.gurobi.com gurobipy
    
    b) obtain the relevant license and activate using grbgetkey (have to download gurobi install files from website to access    grbgetkey as that's not installed using pip
    
    c) copy the gurobi.lic file wherever you initially installed it to the following directory: [your python dir]/site-packages/gurobipy/.libs **note there is an existing restricted install license in the directory, simply replace it.
    
- Install glpk: 
   
        pip3 install glpk
        notes: error may come: ERROR: could not build wheels for glpk which use PEP 517 and cannot be installed directly
        sollution: sudo apt install libglpk-dev python3.8-dev libgmp3-dev, pip install glpk
   
- Install polytope: (polytope operations)
        
        pip install polytope
   
- Install pypoman: (plot star sets) 
   
        pip install pypoman 
     
- Install tabulate: (print latex table)

        pip install tabulate
        
- Install mathplotlib: 

        pip install mathplotlib
        
        
        
# Artifacts 

## HSCC2023: 

- Paper: Quantitative Verification of Neural Networks using ProbStars [https://www.dropbox.com/s/fd6fpydoy5rx3w3/hscc23probstar.pdf?dl=0]

- Run following commands, a new folder named artifact will be generated to store all the results (figures and tables), the tables and figures will be also printed on the screen. 

- Our experiment is done on a computer with the following configuration: Intel Core i7-10700 CPU @ 2.9GHz x 8 Processors, 63.7 GiB Memory, 64-bit Ubuntu 18.04.6 LTS OS. 

- Figure 5 & Table 1:
   
        python3 HSCC2023_run_tinyNet.py
        
- Table 2: 

        python3 HSCC2023_run_ACASXu_small.py
        
- Table 3: 
      
        python3 HSCC2023_run_RocketNet.py
        
- Table 5:

        python3 HSCC2023_run_ACASXu_full.py
