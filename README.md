# StarV

Event-driven Monitoring and Verification Codesign for Distributed Learning-enabled Cyber-Physical Systems with Star Reachability

* Operating System: Ubuntu 20, 22. `<!-- - RAM: at least 64 GB -->`
* Python version: 3.8+
* Dependencies: gurobipy, glpk, polytope, pypoman, tabulate, mathplotlib, numpy, scipy, ipyparallel, torchvision, plotly, onnx, onnx2pytorch, onnxruntime, scikit-learn
* Key features:
  1. Qualitative and quantitative verification algorithms for deep neural networks and distributed Le-CPS
  2. Probabilistic Star Temporal Logic (under development)
  3. Monitoring algorithms (under development)
  4. Target ROS-based applications (under development)

<!-- -  Our experiment is done on a computer with the following configuration: Intel Core i7-10700 CPU @ 2.9GHz x 8 Processors, 63.7 GiB Memory, 64-bit Ubuntu 18.04.6 LTS OS. -->

## Installation

Clone the StarV repository:

```bash
git clone https://github.com/V2A2/StarV
```

File structure:

```
StarV (root directory)
│    README.md
│    requirements.txt
│    gurobi.lic (for Docker usage)
│    setup.py
│    ...  
│
└───.devcontainer
│   │    Docker scripts
│
└───StarV (dev directory)
│   │    All main algorithm scripts
│   
└───artifacts
│   └───HSCC2023_ProbStar
│       │    HSCC2023_ProbStar.py
│       │    README.md
│       └─── results
│   └─── ... other artifacts ...
└───tests
│   │   all testing scripts
│   │   ...   
└───tutorials
│   └───reachable_sets
│       │    tutorial scripts
│   └───layers
│       │    tutorial scripts
│   └───...
```

### Option 1: Running with Local Machine

#### Gurobi Installation on Ubuntu

**Dowload Gurobi and extract.**
Go to https://www.gurobi.com/downloads/ and download the correct version of Gurobi.
Or use the following command:

```bash
wget https://packages.gurobi.com/10.0/gurobi10.0.1_linux64.tar.gz
```

https://www.gurobi.com/documentation/10.0/remoteservices/linux_installation.html recommends installing Gurobi `/opt` for a shared installtion. Note: One might have to create the ~/opt/ directory using mkdir ~/opt first.

```bash
mv gurobi10.0.1_linux64.tar.gz ~/opt/
```

Move into the directory and extract the content.

```bash
cd ~/opt/
tar -xzvf gurobi10.0.1_linux64.tar.gz
rm gurobi10.0.1_linux64.tar.gz
```

**Setting up the environment variables.**
Open the `~/.bashrc` file.

```bash
vim ~/.bashrc
```

Add the following lines, replacing {PATH_TO_YOUR_HOME} with the _aboslute_ path to your home directory, and save the file:

```bash
export GUROBI_HOME="{PATH_TO_YOUR_HOME}/opt/gurobi1001/linux64"
export GRB_LICENSE_FILE="{PATH_TO_YOUR_HOME}/gurobi.lic"
export PATH="${PATH}:${GUROBI_HOME}/bin"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${GUROBI_HOME}/lib"
```

Note: If one installed Gurobi or the license file into a different directory, one has to adjust the paths in the first two lines.
After saving, reload .bashrc:

```bash
source ~/.bashrc
```

**Acquire your license from https://www.gurobi.com/academia/academic-program-and-licenses/**
At `~/opt/gurobi1001/linux64/bin` copy the `grbgetkey` line from the site and enter it into a terminal. Please save the gurobi license `gurobi.lic` in the corresponding directory to `GRB_LICENSE_FILE="{PATH_TO_YOUR_HOME}/gurobi.lic"`.

#### Installation and setup

**Install Ubuntu packages:**

```bash
sudo apt-get install python3-dev python3-pip libgmp-dev libglpk-dev libgmp3-dev 
```

**set up conda environment:**

```bash
# Remove the old environment, if necessary.
conda deactivate; conda env remove -n starv
# install all dependents into the alpha-beta-crown environment
conda create -n starv python=3.8
# activate the environment
conda activate starv
# deactivate the environment when you are done
conda deactivate
```

**Install StarV as a local Python package:** while installing StarV package, it will install all Python dependency packages listed in the requirement.txt.

```bash
# activate the environment
conda activate starv
# install starV as a local Python package
pip3 install -e .
```

The requirement.txt contains the following Python dependency packages:

```bash
gurobipy==11.0.3
glpk
pycddlib<=2.1.8
polytope
pypoman
tabulate
matplotlib
numpy<=1.26.4
scipy<1.13.1
ipyparallel
torchvision
plotly==5.14.1
onnx
onnx2pytorch
onnxruntime
scikit-learn
```

### Option 2: Running with Docker

Acquire Gurobi Web License Service (WLS) license from https://www.gurobi.com/features/web-license-service/ for Docker container
Place ``gurobi.lic`` at ``/StarV`` root repository

```
StarV (root directory)
│    gurobi.lic (for Docker usage)
│    setup.py
│    ...  
│
└───.devcontainer
│   │    build_docker.sh
│   │    launch_docker.sh
│   │    ...
│
└───StarV (dev directory)│   
│   
└─── ...
```

At the ``/StarV`` **root repository**, build the docker:

```bash
sh .devcontainer/build_docker.sh
```

Launch the docker image:

```bash
sh .devcontainer/launch_docker.sh
```

## Tests

```
StarV (root directory)
│    ...  
└───.devcontainer
└───StarV (dev directory)
└───artifacts
└───tests
│   │   all testing scripts
│   │   ...   
└───tutorials
```

To run the testing scripts, navigate to the ``StarV/tests/`` and execute python scripts. For example, to run testing script for ProbStar set, you just run:

```bash
conda activate starv # activate the conda environment if you followed the option 1 for installation, ignore this command if you installed it with Docker
cd tests
python3 test_set_probstar.py
```

## Tutorials

```
StarV (root directory)
│    ...  
└───.devcontainer
└───StarV (dev directory)
└───artifacts
└───tests  
└───tutorials
│   └───reachable_sets
│       │    tutorial scripts
│   └───layers
│       │    tutorial scripts
│   └───...
```

To run the tutorial scripts, navigate to the ``StarV/tutorials/reachable_sets/`` and execute python scripts. For example, to run tutorial script for Star set, you just run:

```bash
conda activate starv # activate the conda environment if you followed the option 1 for installation, ignore this command if you installed it with Docker
cd tutorials/reachable_sets
python3 tutorial_star.py
```

## Artifacts

### HSCC2023:

Paper: Quantitative Verification of Neural Networks using ProbStars [https://www.dropbox.com/s/fd6fpydoy5rx3w3/hscc23probstar.pdf?dl=0]

Run the following command at the StarV **root directory** to regenerate all the results, figures and tables will be saved in ``StarV/artifacts/HSCC2023_ProbStar/``.
Details for this artifact can be found in ``StarV/artifacts/HSCC2023_ProbStar/README.md``.

```bash
python3 artifacts/HSCC2023_ProbStar/HSCC2023_ProbStar.py
```
