# StarV
Event-driven Monitoring and Verification Codesign for Distributed Learning-enabled Cyber-Physical Systems with Star Reachability
 - Operating System: Ubuntu 20.04
 - RAM: at least 64 GB
 - Python version: 3.8 and later versions
 - Dependencies: gurobipy, glpk, polytope, pypoman, tabulate, mathplotlib, numpy, scipy, ipyparallel, torchvision
 - Key features:
    1) Qualitative and quantitative verification algorithms for deep neural networks and distributed Le-CPS
    2) Probabilistic Star Temporal Logic (under development)
    3) Monitoring algorithms (under development)
    4) Target ROS-based applications (under development)

-  Our experiment is done on a computer with the following configuration: Intel Core i7-6950X CPU @ 3.00GHz x 20 Processors, 125.7 GiB Memory, 64-bit Ubuntu 20.04.6 LTS OS.

# StarV Installation

  - Download the ICSE 2025 artifact package

  - Go inside StarV and run tests or ICSE artifacts following instructions in the Artifact section

        StarV
        │   README.md
        |   requirements.txt
        |   gurobi.lic
        │   Test and evaluation scripts
        │   ...    
        │
        └───.devcontainer
        │
        └───StarV
        │   │   All main algorithm scripts
        │   
        └───artifacts
            └───ICSE2025_Algebra
            └───...
  

# Running with Docker

  - Acquire Gurobi Web License Service (WLS) license from https://www.gurobi.com/features/web-license-service/ for Docker container

  - Place ```gurobi.lic``` at ```/StarV``` main repository

        StarV
        |   gurobi.lic
        │   ...    
        │
        └───.devcontainer
            |   build_docker.sh
            |   launch_docker.sh
            |   ...
        |
        └───StarV
        │   
        └───artifacts
  

  - At ```/StarV``` main repository, build the docker:

        sh .devcontainer/build_docker.sh

  - Launch the docker image:

        sh .devcontainer/launch_docker.sh

  - Follow the ICSE 2025 Artifact instructions below to reproduce the evaluation results

# Running with Local Machine

## Gurobi Installation on Ubuntu

1) Download Gurobi and extract.

   Go to https://www.gurobi.com/downloads/ and download the correct version of Gurobi.

        wget https://packages.gurobi.com/11.0/gurobi11.0.2_linux64.tar.gz

   https://www.gurobi.com/documentation/11.0/remoteservices/linux_installation.html recommends installing Gurobi `/opt` for a shared installation.

        mv gurobi11.0.2_linux64.tar.gz ~/opt/

   Note: One might have to create the ~/opt/ directory using mkdir ~/opt first.

   Move into the directory and extract the content.

        cd ~/opt/
        tar -xzvf gurobi11.0.2_linux64.tar.gz
        rm gurobi11.0.2_linux64.tar.gz


2) Setting up the environment variables.

    Open the `~/.bashrc` file.

        vim ~/.bashrc

    Add the following lines, replacing {PATH_TO_YOUR_HOME} with the _aboslute_ path to one's home directory, and save the file:

        export GUROBI_HOME="{PATH_TO_YOUR_HOME}/opt/gurobi1102/linux64"
        export GRB_LICENSE_FILE="{PATH_TO_YOUR_HOME}/gurobi.lic"
        export PATH="${PATH}:${GUROBI_HOME}/bin"
        export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${GUROBI_HOME}/lib"

    Note: If one installed Gurobi or the license file into a different directory, one has to adjust the paths in the first two lines.

    After saving, reload .bashrc:

        source ~/.bashrc

3) Acquire one's license from https://www.gurobi.com/academia/academic-program-and-licenses/

    At `~/opt/gurobi1102/linux64/bin` copy the `grbgetkey` line from the site and enter it into a terminal. Please save the gurobi license `gurobi.lic` in the corresponding directory to `GRB_LICENSE_FILE="{PATH_TO_YOUR_HOME}/gurobi.lic"`.

## Dependencies Installation

- Install Ubuntu packages:

        sudo apt-get install python3-dev python3-pip libgmp-dev libglpk-dev libgmp3-dev

- Install Python dependencies:

        pip3 install -r requirements.txt

  The requirement.txt contains the following packages:

        gurobipy==11.0.2
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

## ICSE 2025:

- Paper: Algebra for Qualitative and Quantitative Verification of Complex Learning-enabled Cyber-Physical Systems

- Run the following command, a new folder named "artifact/ICSE2025_Algebra" will be generated to store all the results (figures and tables).

- Our experiment is done on a computer with the following configuration: Intel Core i7-6950X CPU @ 3.00GHz x 20 Processors, 125.7 GiB Memory, 64-bit Ubuntu 20.04.6 LTS OS.

- Only four cores are used in the verification due to memory consumption (ACC and AEBS System)

- All the evaluation results can be reproduced by running the following command:

        python3 2025_ICSE.py

   - Figure 5 (a, b) can be reproduced by ```generate_exact_Q2_verification_results(net_id='5x20')``` function

   - Figure 5 (c) can be reproduced by ```generate_approx_Q2_verification_results(net_id='5x20', pf=0.01)``` function

   - Figure 6 (a, b, c) can be reproduced by ```generate_exact_reachset_figs(net_id='5x20')``` function

   - Figure 7 can be reproduced by ```generate_exact_reachset_figs_AEBS()``` function

   - Figure 8 (a) can be reproduced by ```generate_AEBS_Q2_verification_results(initSet_id=0, pf=0.005)``` function

   - Figure 8 (b) can be reproduced by ```generate_AEBS_Q2_verification_results(initSet_id=1, pf=0.005)``` function

   - Figure 8 (c) can be reproduced by ```generate_AEBS_Q2_verification_results(initSet_id=2, pf=0.005)``` function

   - Figure 8 (d) can be reproduced by ```generate_AEBS_Q2_verification_results(initSet_id=3, pf=0.005)``` function

   - Table 1 can be reproduced by ```generate_VT_Conv_vs_pf_net()``` function

   - Table 2 can be reproduced by ```generate_AEBS_VT_Conv_vs_pf_initSets()``` function
