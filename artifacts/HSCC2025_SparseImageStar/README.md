# StarV
Event-driven Monitoring and Verification Codesign for Distributed Learning-enabled Cyber-Physical Systems with Star Reachability
 - Operating System: Ubuntu 18., 20.
 - RAM: at least 128 GB
 - Python version: $\leq$ 3.11
 - Key features: 
    1) Qualitative and quantitative verification algorithms for deep neural networks and distributed Le-CPS 
    2) Probabilistic Star Temporal Logic 
    3) Monitoring algorithms (under development)
    4) Target ROS-based applications (under development)

-  Our experiment is done on a computer with the following configuration: Intel Core i7-6940X CPU @ 3.0GHz x 20 Processors, 125.7 GiB Memory, 64-bit Ubuntu 20.04.6 LTS OS.

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
        |   HSCC2025_SpraseImageStar.py
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
            └───HSCC2025_SparseImageStar
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

-  Our experiment is done on a computer with the following configuration: Intel Core i7-6940X CPU @ 3.0GHz x 20 Processors, 125.7 GiB Memory, 64-bit Ubuntu 20.04.6 LTS OS.

- Run following commands, a new folder named `StarV/artifacts/HSCC2025_SparseImageStar/results`  will be generated to store all the results (figures and tables). 


### Reproduce all evaluation results:

At StarV directory, `StarV`,

    python3 HSCC2025_SparseImageStar.py

### Reproduce individual results:

At StarV directory, `StarV`, run following functions to reproduce individual the evaluation result.

- Table 1: Verification results of the Small MNIST CNN (CAV2020)

      verify_convnet_network(net_type='Small', dtype='float64')
      plot_table_covnet_network(net_type = 'Small')

- Table 2: Verification results of the Medium MNIST CNN (CAV2020)

      verify_convnet_network(net_type='Medium', dtype='float64')
      plot_table_covnet_network(net_type = 'Medium')

- Table 3: Verification results of the Large MNIST CNN (CAV2020)
   
      verify_convnet_network(net_type='Large', dtype='float64')
      plot_table_covnet_network(net_type = 'Large')

- Table 4: Verification results of VGG16 in seconds (vnncomp2023)

      verify_vgg16_network(dtype='float64')
      verify_vgg16_converted_network(dtype='float64')
      verify_vgg16_network_spec_cn()
      plot_table_vgg16_network()

- Figure 4: Memory usage and computation time comparison between ImageStar and SparseImageStar (SIM) in verifying the vggnet16 network (vnncomp2023) with spec 11 image
        
      memory_usage_vgg16(spec=11)

- Figure 5: Memory usage and computation time comparison between ImageStar and SparseImageStar (SIM) in verifying the oval21 network with 𝑙∞ norm attack on all pixels.
        
      memory_usage_oval21()

- Figure 6: Memory usage and computation time comparison between ImageStar and SparseImageStar (SIM) in verifying the vggnet16 network (vnncomp2023) with spec c4 image

      memory_usage_vgg16_spec_cn(spec=4)

### Reproduce NNV results:

### Reproduce NNENUM results:

### Reproduce $\alpha, \beta$-CROWN (VNNCOMP2024) results:

Github link: https://github.com/Verified-Intelligence/alpha-beta-CROWN_vnncomp2024

Clone $\alpha, \beta$-CROWN (VNNCOMP2024) verifier

    git clone --recursive https://github.com/Verified-Intelligence/alpha-beta-CROWN_vnncomp2024.git

Setup the conda environment:

- Remove the old environment, if necessary.

      conda deactivate; conda env remove --name alpha-beta-crown

- install all dependents into the alpha-beta-crown environment

      conda env create -f complete_verifier/environment.yaml --name alpha-beta-crown

- activate the environment

      conda activate alpha-beta-crown

### Reproduce ERAN (DeepPoly) results:

### Reproduce Marabou results: