# Verification of Recurrent Neural Networks with Star Reachability

## HSCC2023

## Source Code:
https://github.com/transafeailab/nnv

## 0. Pre-installed Virtual Machine

We provide the virtual machine (VM) that has pre-installed NNV and RnnVerify and their dependent packages. We created the VM with VMware Workstation Pro 17.

### 0.1 Virtual Machine

The VM (16.4 GB) can be dowloaded from the link below. 

    https://www.dropbox.com/sh/dapm02ewj07aasm/AAA63Tc4RC5Xk-pG5vhdC8msa?dl=0


### 0.2 Virtual Machine Log In

    username: nnv
    password: nnv 

NNV directory: `/home/nnv/Desktop/nnv`.

RnnVerify directory: `/home/nnv/Desktop/RnnVerify`.

MATLAB directory: `/usr/local/MATLAB/R2022b/bin/./matlab`.

One can launch MATLAB by executing the file: `/home/nnv/Desktop/MATLAB`.

### 0.3 Activate MATLAB

Unfortunately, we can prove a pre-installed virtual machine but cannot provide MATLAB and gurobi licenses. To use a pre-installed virtual machine, one must activate MATLAB and update the Gurobi license.

    /usr/local/MATLAB/R20XXx/bin/glnxa64/MathWorksProductAuthorizer.sh

### 0.4 Update Gurobi license.

Acquire the gurobi license from https://www.gurobi.com/academia/academic-program-and-licenses/

At `/home/nnv/opt/gurobi1000/linux64/bin` copy the `grbgetkey` line from the site and enter it into a terminal.

**Please `save/replace` gurobi lincense (gurobi.lic) at `/home/nnv/gurobi.lic`.


### 0.5 Set up Gurobi for MATLAB

To set up Gurobi for MATLAB, launch the Gurobi MATLAB setup script, `gurobi_steup.m` located at `/home/nnv/opt/gurobi1000/linux64/matlab` in MATLAB.


## 1. NNV Installation

### 1.1 System requirements

OS: Ubuntu 18.04 (due to RnnVerify)

RAM: at least 32 GB

MATLAB version: 2021b or later

Gurobi optimizer: 9.12 version (or later; 10.00 is tested and works)

Note: Different versions of Gurobi may produce slightly different verification results than the paper. The verification results of the paper are produced with Gurobi 9.12 version. Different verification results may be computed if MATLAB optimization is used instead of Gurobi.

Detailed installation of NNV is available at:

https://github.com/transafeailab/nnv#installation

### 1.2 Required MATLAB toolboxes

    - Control System
    - Deep Learning 
    - Optimization
    - Parallel Computing
    - Sytem Identification

### 1.3 Clone NNV repository

    git clone https://github.com/transafeailab/nnv.git

### 1.4 Install NNV (install.m)

In MATLAB, navigate to the `/nnv` folder. Execute the `install.m` script, which will set up various dependencies (mostly via tbxmanager). This should also set up the path correctly within Matlab, so all dependencies are available.

https://github.com/transafeailab/nnv/blob/master/install.m

**If MATLAB is `restarted`, to work again, `install.m` must be executed again. Alternatively, one can `savepath` to update the path after executing the install (but in this case, MATLAB may need to have been launched with administrative privilege).

**In order to add `/nnv` to MATLAB `permenantly`, on MATLAB Command Window (for `savepath`, `sudo` or `administrative privilege` may be required to launch MATLAB):

    addpath(genpath('{PATH_TO_NNV}'))
    savepath



### 1.5 Install Gurobi for MATLAB

1) Dowload Gurobi and extract.

Go to https://www.gurobi.com/downloads/ and download the correct version of Gurobi.

    wget https://packages.gurobi.com/10.0/gurobi10.0.0_linux64.tar.gz

https://www.gurobi.com/documentation/10.0/remoteservices/linux_installation.html recommends installing Gurobi `/opt` for a shared installtion.

    mv gurobi10.0.0_linux64.tar.gz ~/opt/

Note: One might have to create the ~/opt/ directory using mkdir ~/opt first.

Move into the directory and extract the content.

    cd ~/opt/
    tar -xzvf gurobi10.0.0_linux64.tar.gz
    rm gurobi10.0.0_linux64.tar.gz


2) Setting up the environment variables.

Open the `~/.bashrc` file.

    vim ~/.bashrc

Add the following lines, replacing {PATH_TO_YOUR_HOME} with the _aboslute_ path to your home directory, and save the file:

    export GUROBI_HOME="{PATH_TO_YOUR_HOME}/opt/gurobi1000/linux64"
    export GRB_LICENSE_FILE="{PATH_TO_YOUR_HOME}/gurobi.lic"
    export PATH="${PATH}:${GUROBI_HOME}/bin"
    export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${GUROBI_HOME}/lib"

Note: If one installed Gurobi or the license file into a different directory, one has to adjust the paths in the first two lines.

After saving, reload .bashrc:

    source ~/.bashrc

3) Acquire your license from https://www.gurobi.com/academia/academic-program-and-licenses/

At `~/opt/gurobi1000/linux64/bin` copy the `grbgetkey` line from the site and enter it into a terminal. Please save the gurobi license `gurobi.lic` in the corresponding directory to `GRB_LICENSE_FILE="{PATH_TO_YOUR_HOME}/gurobi.lic"`.

4) Setting up Gurobi for MATLAB

https://www.gurobi.com/documentation/10.0/quickstart_linux/matlab_setting_up_grb_for_.html

To set up Gurobi for MATLAB, one needs to launch the Gurobi MATLAB setup script, `gurobi_steup.m` located at `~/opt/gurobi1000/linux64/matlab` in MATLAB.

**Add the directory below to MATLAB such that `linprog` function will launch based on `gurobi` optimization instead of MATLAB optimization.

`~/opt/gurobi1000/linux64/examples/matlab`

**On MATLAB Command Window (for `savepath`, `sudo` or `administrative privilege` may be required to launch MATLAB):

    addpath(genpath('{PATH_TO_YOUR_HOME}/opt/gurobi1000/linux64/examples/matlab'))
    savepath
            

## 2. (Optional) RnnVerify Installation 

### 2.1 Clone RnnVerify

    git clone https://github.com/yuvaljacoby/RnnVerify.git

### 2.2 Install RnnVerify

Detailed installation of RnnVerify is available at:

https://github.com/yuvaljacoby/RnnVerify

However, provided installation guide is outdated. We provide our installation guide for RnnVerify. The achieved RnnVerify results may differ slightly from those in the paper due to differences in Python packages.

### 2.2.1 System Requirement

OS: Ubuntu 18.04

Python: 3.8

Gurobi optimizer: 9.12 version (or later; 10.00 is tested and works)


### 2.3 RnnVerify Installation on Ubuntu 18.04 (clean version)

Install Ubuntu 18.04 on virtual machine or on your device.

### 2.3.1 Update the system

    sudo apt update && sudo apt upgrade -y

### 2.3.2 Install Python 3.8

Default Python version of Ubuntu 18.04 is Python 3.6.9.

You may check it by

    python3 --version

Install Python 3.8.

    sudo apt-get install build-essential libpq-dev libssl-dev openssl libffi-dev zlib1g-dev -y
    sudo apt-get install -y software-properties-common
    sudo add-apt-repository ppa:deadsnakes/ppa
    sudo apt-get update
    sudo apt-get install python3-pip python3.8-dev
    sudo apt-get install python3.8

Update `python3` to point to `python3.8`.

    sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.6 1
    sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 2
    sudo update-alternatives --config python3

Type `1` after `sudo update-alternatives --config python3`.

    sudo ln -sf /usr/bin/python3.8 /usr/bin/python3

(Recommended) Updating python3.8 will cause trouble for default `Terminal` (gnome-terminal) on Ubuntu.

    sudo nano /usr/bin/gnome-terminal

Change `#!/usr/bin/python3` to `#!/usr/bin/python3.6`.

(Optional) After updating to python3.8, there will be errors related to python3-apt.

    sudo apt remove --purge python3-apt
    sudo apt autoclean
    sudo apt install python3-apt

(Optional)

    sudo apt-get install gnome-control-center
    sudo apt-get install update-manager



### 2.3.3 Install Python packages.

Install pip3.

    sudo apt install python3-pip

Update pip3.

    pip3 install --upgrade pip

Install pybind11 for marabou installation.

    pip3 install pybind11

Change `requirements.txt` (located inside RnnVerify) as below:

    ipython==7.9.0
    Keras==2.4.0
    matplotlib==3.5.3
    numpy==1.17.3
    pandas==0.25.2
    pytest==5.2.2
    pytest-forked==1.1.3
    pytest-repeat==0.8.0
    pytest-rerunfailures==9.0
    pytest-sugar==0.9.2
    pytest-timeout==1.3.3
    pytest-xdist==1.31.0
    seaborn==0.9.0
    tabulate==0.8.3
    tqdm==4.36.1
    PTable==0.9.2
    tensorflow==2.2.0
    gurobipy==10.0.0 

Install required Python packages for RnnVerify.

    pip3 install -r requirements.txt

Please install other packages if the above installation gives an error as below:

    ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
    launchpadlib 1.10.6 requires testresources, which is not installed.

Install `launchpadlib`.

    pip3 install launchpadlib==1.10.6

Install `cmake`.

    sudo apt install cmake


Install `libboost-all-dev`.

    sudo apt-get install libboost-all-dev

Install following commands (may not be needed; install if `cmake` when `compiling  Marabou` results in error related to expect and/or protobuf):

    sudo apt-get remove python-pexpect python3-pexpect
    pip3 install --upgrade pexpect
    pip3 install protobuf==3.20.0


### 2.3.4 Compile Marabou

    mkdir build
    cd build
    cmake ..
    cmake --build .

To compile Marabou `sudo` may be required for `cmake`. 

    mkdir build
    cd build
    sudo cmake ..
    sudo cmake --build .


## 3. Core results

1) Figure 3. Average verification times of different approaches for the small RNN network. Approaches include exact-star, RNNSVerify, approx-star, RnnVerify.

2) Table 1. Exact verification results for the small RNN.

3) Figure 4. The ranges of all outputs in the first output set in 8 counter output sets for $x_1$ with $T_{max} = 20$ (Table).

4) Table 2. Verification results for the RNN using the approximate and relaxed reachability.

5) Figure 5. Verification performance with different attack bount $\epsilon$ on network $\mathcal{N}_{8,0}$.

6) Figure 5. Verification performance with different attack bount $\epsilon$ on network $\mathcal{N}_{8,0}$.

7) Table 4. Full verification results for $\mathcal{N}_{2,0}$.

8) Table 5. Full verification results for $\mathcal{N}_{2,2}$.

9) Table 6. Full verification results for $\mathcal{N}_{4,0}$.

10) Table 7. Full verification results for $\mathcal{N}_{4,2}$.

11) Table 8. Full verification results for $\mathcal{N}_{4,4}$.

12) Table 9. Full verification results for $\mathcal{N}_{8,0}$.

## 4. Reproducing the results in the paper

### 4.1 Reprodcinng all NNV results by a single run
-	Navigate to `/nnv/examples/artifact/HSCC2023/`
-	Run `reproduce_HSCC2022.m`

### 4.1.1 Reroducing small NNV results (includes Figure 3, Table 1, Figure 4, Table 2, Figure 5)
-	Navigate to `/nnv/examples/artifact/HSCC2023/
-	Run `reproduce_HSCC2023_small_results.m`

### 4.1.2 Reroducing full NNV results (includes Figure 3, Table 1, Figure 4, Table 4, Table 5, Table 6, Table 7, Table 8, Table 9, Figure 5)
-	Navigate to `/nnv/examples/artifact/HSCC2023/
-	Run `reproduce_HSCC2023_full_results.m`

### 4.2 Reproducing NNV results individually:
#### 4.2.1 Figure 3
-	Navigate to `/nnv/examples/artifact/HSCC2023/small_RNN`
-	Run `verify_small_RNN.m`

#### 4.2.2 Table 1
-	Navigate to `/nnv/examples/artifact/HSCC2023/small_RNN`
-	Run `verify_small_RNN.m`

#### 4.2.3 Figure 4
-	Navigate to `/nnv/examples/artifact/HSCC2023/small_RNN`
-	Run `verify_small_RNN.m`

#### 4.2.4 Table 2
##### For verification results for $\mathcal{N}_{2,0}$:
-	Navigate to `/nnv/examples/artifact/HSCC2023/N_2_0`
-	Run `verify_N2_0.m`

##### For verification results for $\mathcal{N}_{2,2}$:
-	Navigate to `/nnv/examples/artifact/HSCC2023/N_2_2`
-	Run `verify_N2_2.m`

##### For verification results for $\mathcal{N}_{4,0}$:
-	Navigate to `/nnv/examples/artifact/HSCC2023/N_4_0`
-	Run `verify_N4_0.m`

##### For verification results for $\mathcal{N}_{4,2}$:
-	Navigate to `/nnv/examples/artifact/HSCC2023/N_4_2`
-	Run `verify_N4_2.m`

##### For verification results for $\mathcal{N}_{4,4}$:
-	Navigate to `/nnv/examples/artifact/HSCC2023/N_4_4`
-	Run `verify_N4_4.m`

##### For verification results for $\mathcal{N}_{8,0}$:
-	Navigate to `/nnv/examples/artifact/HSCC2023/N_8_0`
-	Run `verify_N4_4.m`

#### 4.2.5 Figure 5
-	Navigate to `/nnv/examples/artifact/HSCC2023/N_8_0`
-	Run `epsilon_vs_robustness_and_time.m`

#### 4.2.6 Table 4
-	Navigate to `/nnv/examples/artifact/HSCC2023/N_2_0`
-	Run `verify_N2_0_full.m`

#### 4.2.7 Table 5
-	Navigate to `/nnv/examples/artifact/HSCC2023/N_2_2`
-	Run `verify_N_2_2_full.m`

#### 4.2.8 Table 6
-	Navigate to `/nnv/examples/artifact/HSCC2023/N_4_0`
-	Run `verify_N_4_0_full.m`

#### 4.2.9 Table 7
-	Navigate to `/nnv/examples/artifact/HSCC2023/N_4_2`
-	Run `verify_N_4_2_full.m`

#### 4.2.10 Table 8
-	Navigate to `/nnv/examples/artifact/HSCC2023/N_4_4`
-	Run `verify_N_4_4_full.m`

#### 4.2.11 Table 9
-	Navigate to `/nnv/examples/artifact/HSCC2023/N_8_0`
-	Run `verify_N_8_0_full.m`

### 4.3 (Optional) Reproduing RnnVerify results:
#### 4.3.1 Figure 3
-   Navigate to RnnVerify directory
-   Run `PYTHONPATH=. python3 rnn_experiment/algorithms_compare/experiment.py exp 25`

#### 4.3.2 Table 2, 4, 5, 6, 7, 8, and 9
-   Navigate to RnnVerify directory
-   Run `PYTHONPATH=. python3 rnn_experiment/self_compare/experiment.py exp all 20`

#### 4.3.3 Figure 5
-   Place `/nnv/examples/artifact/HSCC2023/diff_attack_bounds.py` file to RnnVerify directory
-   Navigate to RnnVerify directory
-   Run `PYTHONPATH=. python3 diff_attack_bounds.py`



`**` means important