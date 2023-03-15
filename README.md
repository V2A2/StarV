# StarV
Event-driven Monitoring and Verification Codesign for Distributed Learning-enabled Cyber-Physical Systems with Star Reachability
 - Operating System: Ubuntu 18., 20.
 - RAM: at least 64 GB
 - Python version: 3.8
 - Dependencies: gurobipy, glpk, polytope, pypoman, tabulate, mathplotlib, numpy, scipy, ipyparallel, torchvision
 - Key features: 
    1) Qualitative and quantitative verification algorithms for deep neural networks and distributed Le-CPS 
    2) Probabilistic Star Temporal Logic (under development)
    3) Monitoring algorithms (under development)
    4) Target ROS-based applications (under development)

-  Our experiment is done on a computer with the following configuration: Intel Core i7-10700 CPU @ 2.9GHz x 8 Processors, 63.7 GiB Memory, 64-bit Ubuntu 18.04.6 LTS OS.

# Pre-installed Virtual Machine

We provide the virtual machine (VM) that has pre-installed StarV, gurobi, and all depedendent packages. We created the VM with VMware Workstation Pro 17.

1) Virtual Machine

   The VM (10.1 GB) can be dowloaded from the link below. 

        https://www.dropbox.com/sh/dif9387wrz2n7kw/AADI1zG9k0zCBnJ7VuhswGf7a?dl=0

2) Virtual Machine Log In

        username: starv
        password: starv 

   StarV directory: `/home/starv/Desktop/StarV`.

3) Update Gurobi license.

   Acquire the gurobi license from https://www.gurobi.com/academia/academic-program-and-licenses/

   At `/home/nnv/opt/gurobi1001/linux64/bin` copy the `grbgetkey` line from the site and enter it into a terminal.

   **Please `save/replace` gurobi lincense (gurobi.lic) at `/home/starv/gurobi.lic`.


# StarV installation, tests and artifacts

  - No installation, just clone the repository:
  
        git clone https://github.com/V2A2/StarV
        
  - Go inside StarV and run tests or artifacts
  

# Gurobi Installation on Ubuntu

1) Dowload Gurobi and extract.

   Go to https://www.gurobi.com/downloads/ and download the correct version of Gurobi.

        wget https://packages.gurobi.com/10.0/gurobi10.0.1_linux64.tar.gz

   https://www.gurobi.com/documentation/10.0/remoteservices/linux_installation.html recommends installing Gurobi `/opt` for a shared installtion.

        mv gurobi10.0.1_linux64.tar.gz ~/opt/

   Note: One might have to create the ~/opt/ directory using mkdir ~/opt first.

   Move into the directory and extract the content.

        cd ~/opt/
        tar -xzvf gurobi10.0.1_linux64.tar.gz
        rm gurobi10.0.1_linux64.tar.gz


2) Setting up the environment variables.

    Open the `~/.bashrc` file.

        vim ~/.bashrc

    Add the following lines, replacing {PATH_TO_YOUR_HOME} with the _aboslute_ path to your home directory, and save the file:

        export GUROBI_HOME="{PATH_TO_YOUR_HOME}/opt/gurobi1001/linux64"
        export GRB_LICENSE_FILE="{PATH_TO_YOUR_HOME}/gurobi.lic"
        export PATH="${PATH}:${GUROBI_HOME}/bin"
        export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${GUROBI_HOME}/lib"

    Note: If one installed Gurobi or the license file into a different directory, one has to adjust the paths in the first two lines.

    After saving, reload .bashrc:

        source ~/.bashrc

3) Acquire your license from https://www.gurobi.com/academia/academic-program-and-licenses/

    At `~/opt/gurobi1001/linux64/bin` copy the `grbgetkey` line from the site and enter it into a terminal. Please save the gurobi license `gurobi.lic` in the corresponding directory to `GRB_LICENSE_FILE="{PATH_TO_YOUR_HOME}/gurobi.lic"`.
 
# Dependencies installation

- Install Ubuntu packages:

        sudo apt-get install python3-dev python3-pip libgmp-dev libglpk-dev libgmp3-dev 

- Install Python dependencies:

        pip3 install -r requirements.txt

  The requirement.txt contains following packages:

        gurobipy==10.0.1
        glpk
        pycddlib
        polytope
        pypoman
        tabulate
        matplotlib
        numpy
        scipy
        ipyparallel
        torchvision


        
        
# Artifacts 

## HSCC2023: 

- Paper: Quantitative Verification of Neural Networks using ProbStars [https://www.dropbox.com/s/fd6fpydoy5rx3w3/hscc23probstar.pdf?dl=0]

- Run following commands, a new folder named artifact will be generated to store all the results (figures and tables), the tables and figures will be also printed on the screen. 

- Our experiment is done on a computer with the following configuration: Intel Core i7-10700 CPU @ 2.9GHz x 8 Processors, 63.7 GiB Memory, 64-bit Ubuntu 18.04.6 LTS OS.

- Only 4 cores are used in the verification due to memory consumption (ACASXu full and RocketNet)

- Figure 5 & Table 1:
   
        python3 HSCC2023_run_tinyNet.py
        
- Table 2: 

        python3 HSCC2023_run_ACASXu_small.py
        
- Table 3: 
      
        python3 HSCC2023_run_RocketNet.py
        
- Table 5:

        python3 HSCC2023_run_ACASXu_full.py