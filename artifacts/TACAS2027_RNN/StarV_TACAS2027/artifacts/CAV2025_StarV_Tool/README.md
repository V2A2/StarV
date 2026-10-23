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

  - Download the CAV 2025 artifact package

  - Go inside StarV and run tests or CAV2025_SparseImageStar artifacts following instructions in the Artifact section

        StarV
        │   README.md
        |   requirements.txt
        |   CAV2025_SpraseImageStar.py
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
            └───CAV2025_SparseImageStar
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

  - Follow the CAV 2025 StarV Tool Artifact instructions below to reproduce the evaluation results


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

- The requirement.txt contains the following packages:

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

## CAV2025 StarV Tool: 

- Paper: Memory-Efficient Verification for Deep Convolutional Neural Networks using StarV Tool [https://www.dropbox.com/s/fd6fpydoy5rx3w3/CAV23probstar.pdf?dl=0]

- Our experiment is done on a computer with the following configuration: Intel Core i7-6940X CPU @ 3.0GHz x 20 Processors, 125.7 GiB Memory, 64-bit Ubuntu 20.04.6 LTS OS.

- Run following commands, a new folder named `StarV/artifacts/CAV2025_StarV_Tool/results`  will be generated to store all the results (figures and tables). 


### Reproduce all evaluation results:

At StarV directory, `StarV`,

    python3 CAV2025_StarV_Tool.py

### Reproduce individual results:

At StarV directory, `StarV`, run following functions to reproduce individual the evaluation result.

- Figure 2: L15 and G15 verification results 

      verify_MNIST_LSTM_GRU(type='gru', hidden=15)
      plot_rnns_results(type='gru', hidden=15)
      verify_MNIST_LSTM_GRU(type='lstm', hidden=15)
      plot_rnns_results(type='lstm', hidden=15)

- Table 2: Verification results of the MNIST CNN

      verify_convnet_network(net_type='Small', dtype='float64')
      verify_convnet_network(net_type='Medium', dtype='float64')
      verify_convnet_network(net_type='Large', dtype='float64')
      plot_table_covnet_network_all()

- Table 3: Verification results of VGG16
        
      verify_vgg16_network_spec_cn()
      plot_table_vgg16_network()

- Figure 3: Memory usage and computation time comparison between ImageStar and SparseImageStar (SIM) in verifying the vggnet16 network (vnncomp2023) with spec 11 image
        
      memory_usage_vgg16(spec=11)  

- Table 4: Combined AcasXu ReLU networks, ProbStar vs MC vs other tools

      comparison_ACASXu()

- Table 5: Verification results (robustness intervals) of NeuroSymbolic  

      verify_temporal_specs_ACC_trapeziu_full()

- Table 6: Verification results of all models for Quantitative verification of Massive linear system

      full_evaluation_results()
      verification_Hylaa_tool()
      generate_table_3_vs_Hylaa_tool()