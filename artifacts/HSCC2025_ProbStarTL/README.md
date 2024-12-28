# StarV
Event-driven Monitoring and Verification Codesign for Distributed Learning-enabled Cyber-Physical Systems with Star Reachability
 - Operating System: Ubuntu 18., 20.
 - RAM: at least 32 GB
 - Python version: $\leq$ 3.11
 - Key features: 
    1) Qualitative and quantitative verification algorithms for deep neural networks and distributed Le-CPS 
    2) Probabilistic Star Temporal Logic 
    3) Monitoring algorithms (under development)
    4) Target ROS-based applications (under development)

-  Our experiment is done on a computer with the following configuration: iMAC 3.8 GHz 8-Core Intel Core i7 with a 128GB memory with a virtual 64-bit Ubuntu 20.04.4 LTS system.

# Getting Gurobi license.
1) Docker:

   Acquire the Gurobi Academic Web License Service (WLS) license from https://www.gurobi.com/features/academic-wls-license/ or https://www.gurobi.com/features/web-license-service/

   Place the Gurobi license inside StarV directory: `StarV/gurobi.lic`.

2) Local Machine:

   Acquire the gurobi license from https://www.gurobi.com/academia/academic-program-and-licenses/

# StarV Installation

  - Download the HSCC 2025 artifact package

  - Go inside StarV and run tests or HSCC2025_SparseImageStar artifacts following instructions in the Artifact section

        StarV
        │   README.md
        |   requirements.txt
        |   HSCC2025_ProbStarTL.py
        |   gurobi.lic
        │   Test scripts
        │   ...    
        │
        └───.devcontainer
        │
        └───StarV
        │   │   All main algorithm scripts
        │   
        └───artifacts
            └───HSCC2025_ProbStarTL
            └───...
  

# Running with Docker

  - Acquire the Gurobi Academic Web License Service (WLS) license from https://www.gurobi.com/features/academic-wls-license/ or https://www.gurobi.com/features/web-license-service/

  - Place ```gurobi.lic``` at ```StarV``` main repository

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
  

  - At ```StarV``` main repository, build the docker:

        sh .devcontainer/build_docker.sh

  - Launch the docker image:

        sh .devcontainer/launch_docker.sh

  - Follow the HSCC 2025 SparseImageStar Artifact instructions below to reproduce the evaluation results


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
        pycddlib<=2.1.8
        polytope
        pypoman
        tabulate
        matplotlib
        numpy<=1.26.4
        scipy
        ipyparallel
        torchvision
        plotly==5.14.1
        onnx
        onnx2pytorch
        onnxruntime
        scikit-learn

        
# Artifacts 

## HSCC2025 SparseImageStar: 

- Paper: Memory-Efficient Verification for Deep Convolutional Neural Networks using SparseImageStar [https://www.dropbox.com/s/fd6fpydoy5rx3w3/hscc23probstar.pdf?dl=0]

-  Our experiment is done on a computer with the following configuration: iMAC 3.8 GHz 8-Core Intel Core i7 with a 128GB memory with a virtual 64-bit Ubuntu 20.04.4 LTS system.

- Run following commands, a new folder named `StarV/artifacts/HSCC2025_ProbStarTL`  will be generated to store all the results (figures and tables). 


### Reproduce all evaluation results:

At StarV directory, `StarV`,

    python3 HSCC2025_ProbStarTL.py

### Reproduce individual results:

At StarV directory, `StarV`, run following functions to reproduce individual the evaluation result.

- Table 2: Verification results for Le-ACC with the network controller 𝑁5×20 (i.e., 5 layers, 20 neurons per layer)

      verify_temporal_specs_ACC()

- Figure 2: Conservativeness analysis of 𝜑1.

      analyze_conservativeness()

- Figure 3: Verification timing performance of 𝜑′4.
   
      analyze_timing_performance()

- Figure 4: Verification complexity depends on
        1) the number of traces (which varies for different networks and different initial conditions),
        2) the number of CDNFs, and
        3) the lengths of CDNFs.

      analyze_verification_complexity()               # Figure 4a
      analyze_verification_complexity_2()             # Figure 4b

- Figure 5: Length of CDNFs for 𝜑′ 4 verification with 𝑇 = 20 and the visualization of a trace satisfying the specification.
        
      analyze_verification_complexity_3()             # Figure 5a
      visualize_satisfied_traces()                    # Figure 5b

- Table 3: Verification results (robustness intervals) of NeuroSymbolic [ 12] are consistent with the proposed ProbStarTL verification results (probabilities of satisfaction). 
        
      verify_temporal_specs_ACC_trapeziu_full()

- Table 4: Quantitative verification results of AEBS system against property 𝜑 = ⋄[0,𝑇 ] (𝑑𝑘 ≤ 𝐿 ∧ 𝑣𝑘 ≥ 0.2) 

      verify_AEBS()

- Figure 6: 50-step reachable sets (𝑑𝑘 vs. 𝑣𝑘 ) of AEBS system (in green) and the unsafe region (in red) for different initial conditions (scenarios) 𝑑0 × 𝑣0. 𝑑𝑘

      generate_exact_reachset_figs_AEBS()