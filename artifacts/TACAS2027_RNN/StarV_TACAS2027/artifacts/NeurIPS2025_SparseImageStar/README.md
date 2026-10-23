<p align="right">
<a href=""><img src="./v2a2_logo.png" width="10%"></a>
</p>

# NeurIPS 2025
# Memory-Efficient Verification for Deep Convolutional Neural Networks using SparseImageStar

* Operating System: Ubuntu 20, 22 
* Python version: 3.11+
* Lincense Requirement: Gurobi License
* Hardware Requirements for all the results: 
   - 128GB RAM
* Dependencies: gurobipy, glpk, pycddlib, polytope, pypoman,
      tabulate, matplotlib, numpy, scipy, ipyparallel, torchvision,
      plotly, onnx, onnx2pytorch, onnxruntime, scikit-learn, pandas
* The provided Docker image is generated with Linux, amd64.
* Key features:
  1. SparseImageStar in CSR and COO format and associated reachability algorithms for different types of layers used in CNNs.
  2. Novel SpGEMM convolution and average pooling operations with the index-shifting technique.

<p align="center">
<a href=""><img src="./overview.png" width="90%"></a>
</p>



## Running with provided Docker image
The pre-built Docker image (`starv_artifact_image.tar`) is located at `NeurIPS2025_SparseImageStar_Submission`. 

To load the Docker image, please navigate to `NeurIPS2025_SparseImageStar_Submission` directory and follow the instructions below:

On the first terminal:
```bash
# Load Docker image
docker load -i starv_artifact_image.tar
# Launch Docker image
docker run --rm -v "./StarV:/work" -it starv_artifact_image:latest bash
```

All the packages required are installed in this Docker image but Gurobi license is not provided. <span style="color: red">Acquire Gurobi Web License Service (WLS) license from https://www.gurobi.com/features/web-license-service/ for Docker container.</span>

On the second terminal:
```bash
docker ps
# Obtain the Docker Container ID for the Docker image: starv_artifact_image:latest
docker cp {PATH_TO_YOUR_GUROBI_LICENSE}/gurobi.lic {DOCKER_CONTAINER_ID}:/opt/gurobi/gurobi.lic
```

Please continue using the first terminal for the rest of the artifact process. The reviewer using Docker image can skip the **Normal Installation** section.



## Normal Installation

File structure:

```
StarV (root directory)
│    README.md (for Artifact)
│    requirements.txt
│    gurobi.lic (Gurobi Web License Service (WLS) license for Docker usage)
│    setup.py
│    smoke_test.sh
│    ...  
│
└───.devcontainer
│   │   Docker scripts
│
└───StarV (dev directory)
│   └───set
│   │   │   sparseimagestar2dcsr.py
│   │   │   sparseimagestar2dcoo.py
│   │   │   imagestar.py
│   │   │   ...
│   └───fun
│   │   │   poslin.py
│   │   │   ...
│   └───layer
│   │   │   Conv2DLayer.py
│   │   │   AvgPool2DLayer.py
│   │   │   BatchNorm2DLayer.py
│   │   │   ReLULayer.py
│   │   │   MaxPool2DLayer.py
│   │   │   FullyConnectedLayer.py
│   │   │   FlattenLayer.py
│   │   │   ...
│   └───net
│   │   │   network.py
│   └───verifier
│   │   │   certifier.py
│   │   │   ...
│   │   ...  
└───artifacts
│   └───NeurIPS2025_SparseImageStar
│   │   │   NeurIPS2025_SparseImageStar.py
│   │   └───results
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

Instead of using a Docker image to reproduce the results, there are alternative options to install the artifact package locally. 

- Option 1: Run on a local machine using a conda environment. This requires the reviewer to acquire a Named-User Gurobi license from https://www.gurobi.com/academia/academic-program-and-licenses/ during installation. Please refer to **Installation Option 1: Running with Local Machine** for detailed instructions.

- Option 2: Run using Docker. This requires the reviewer to acquire a Gurobi Web License Service (WLS) license from https://www.gurobi.com/features/web-license-service/ to build the Docker image. Follow the commands provided in the **Installation Option 2: Running with Docker** to build the image and set up the container.

### Option 1: Running with Local Machine

#### Gurobi Installation on Ubuntu

**Dowload Gurobi and extract.**
Go to https://www.gurobi.com/downloads/ and download the correct version of Gurobi.
Or use the following command:

```bash
wget https://packages.gurobi.com/11.0/gurobi11.0.2_linux64.tar.gz
```

https://www.gurobi.com/documentation/11.0/remoteservices/linux_installation.html recommends installing Gurobi to `/opt` for a shared installtion. Note: one might have to create the ~/opt/ directory using mkdir ~/opt first.

```bash
mv gurobi11.0.2_linux64.tar.gz ~/opt/
```

Move into the directory and extract the content.

```bash
cd ~/opt/
tar -xzvf gurobi11.0.2_linux64.tar.gz
rm gurobi11.0.2_linux64.tar.gz
```

**Setting up the environment variables.**
Open the `~/.bashrc` file.

```bash
vim ~/.bashrc
```

Add the following lines, replacing {PATH_TO_YOUR_HOME} with the _aboslute_ path to your home directory, and save the file:

```bash
export GUROBI_HOME="{PATH_TO_YOUR_HOME}/opt/gurobi1102/linux64"
export GRB_LICENSE_FILE="{PATH_TO_YOUR_HOME}/gurobi.lic"
export PATH="${PATH}:${GUROBI_HOME}/bin"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${GUROBI_HOME}/lib"
```

Note: If one installed Gurobi or the license file into a different directory, one has to adjust the paths in the first two lines.
After saving, reload .bashrc:

```bash
source ~/.bashrc
```

<span style="color: red">Acquire the gurobi license from https://www.gurobi.com/academia/academic-program-and-licenses/.</span>
At `~/opt/gurobi1102/linux64/bin` copy the `grbgetkey` line from the site and enter it into a terminal. Please save the gurobi license `gurobi.lic` in the corresponding directory to `GRB_LICENSE_FILE="{PATH_TO_YOUR_HOME}/gurobi.lic"`.

#### Installation and setup

**Install Ubuntu packages:**

```bash
sudo apt-get install python3-dev python3-pip libgmp-dev libglpk-dev libgmp3-dev 
```

**set up conda environment:**

```bash
# remove the old environment, if necessary
conda deactivate; conda env remove -n starv
# create the starv conda environment
conda create -n starv python=3.11
# activate the environment
conda activate starv
# deactivate the environment when the reviewer is done
conda deactivate
```

**Install StarV as a local Python package:** while installing the StarV package, it will install all Python dependency packages listed in the requirement.txt. Run the following command at the ``/StarV`` **root directory**:

```bash
# activate the environment
conda activate starv
# install starV as a local Python package
pip3 install -e .
```

The requirement.txt contains the following Python dependency packages:

```bash
gurobipy==11.0.2
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
pandas==2.2.3
```

### Option 2: Running with Docker

<span style="color: red">Acquire the Gurobi Web License Service (WLS) license from https://www.gurobi.com/features/web-license-service/ for the Docker container.</span>
Place ``gurobi.lic`` at ``/StarV`` root directory.

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
└───StarV (dev directory)
│   
└─── ...
```

At the ``/StarV`` **root directory**, build the docker:

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

To run the testing scripts, navigate to the ``StarV/tests/`` and execute python scripts. For example, to run testing script for ProbStar set, the reviewer can just run:

```bash
conda activate starv # activate the conda environment if the reviewer followed the option 1 for installation, ignore this command if the reviewer installed it with Docker
cd tests
python3 test_set_probstar.py
```

## Tutorials


```
StarV (root directory)
│   manual.pdf
│   ...  
└───.devcontainer
└───StarV (dev directory)
└───artifacts
└───tests  
└───tutorials
│   └───reachable_sets
│       │    tutorial scripts
│   └───reachability_analysis
│       │    tutorial scripts
│   └───...
```

To run the tutorial scripts, navigate to the ``StarV/tutorials/reachable_sets/`` and execute python scripts. For example, to run tutorial script for Star set, the reviewer can just run:

```bash
conda activate starv # activate the conda environment if the reviewer followed the option 1 for installation, ignore this command if the reviewer installed it with Docker
cd tutorials/reachable_sets
python3 tutorial_star.py
```


## Smoke Tests

```
StarV (root directory)
│    smoke_test.sh
│    ...  
└───.devcontainer
└───StarV (dev directory)
└───artifacts
└───tests 
└───tutorials
```

To run the scripts for artifact smoke test, at the StarV root directory, `StarV`, run the following command:

```bash
conda activate starv # activate the conda environment if the reviewer is following the option 1 for normal installation. The reviewer may ignore this command if using the provided Docker image or following the option 2 for normal installation. 
sh smoke_test.sh
```


## Artifacts

### NeurIPS 2025: Memory-Efficient Verification for Deep Convolutional Neural Networks using SparseImageStar 

- Our experiment is done on a computer with the following configuration: Intel Core i7-6940X CPU @ 3.0GHz x 20 Processors, 125.7 GiB Memory, 64-bit Ubuntu 20.04.6 LTS OS.

- By running the commands below, all the main results such as figures and tables in the paper will be generated at `StarV/artifacts/NeurIPS2025_SparseImageStar/`. All the data for results  will be stored at `StarV/artifacts/NeurIPS2025_SparseImageStar/results/`.

- Reproducing all paper results may take around 19 hours. 

#### Reproduce all evaluation results:

Please **activate the starv conda environment** if the reviewer is following the option 1 for normal installation. The reviewer may ignore this command if using the provided Docker image or following the option 2 for normal installation. 

```bash
conda activate starv
```

At the StarV root directory, `StarV`, run the following command to reproduce:
- **All results (main + appendix)** in the paper (~19 hours).   

       python3 artifacts/NeurIPS2025_SparseImageStar/NeurIPS2025_SparseImageStar.py


#### Reproduce individual results:

To reproduce individual evaluation result in the main paper, please run the following python function individually:

- Figure 1: The worse-case memory usage of generator and center images in ImageStar

      worst_case_vgg16()

- Table 1: Verification results of the MNIST CNN (CAV2020)

      verify_mnist_cnn()

- Table 2: Verification results of VGG16 in seconds (vnncomp2023)

      verify_vgg16()

- Figure 4: Memory usage comparison in verifying the VGG16 (vnncomp2023) with spec 11 image

      memory_usage_vgg16(spec=11)

- Figure 5: Memory usage and computation time comparison in verifying the oval21 network with 𝑙∞ norm attack on all pixels.

      memory_usage_oval21()

- Figure 7: Memory usage and computation time comparison between SparseImageStar CSR and COO in verifying the VGG16 (vnncomp2023) with spec c4 image

      memory_usage_vgg16_spec_cn(spec=4)
