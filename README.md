[![Build Status](https://travis-ci.org/danielecook/python-cli-skeleton.svg?branch=master)](https://travis-ci.org/danielecook/python-cli-skeleton) [![Coverage Status](https://coveralls.io/repos/github/danielecook/python-cli-skeleton/badge.svg?branch=master)](https://coveralls.io/github/danielecook/python-cli-skeleton?branch=master)

# chemotaxis-cli

*A command-line interface to measure chemotaxis index (CI) from chemotaxis assay images*

## Install

1. [Install anaconda](https://docs.anaconda.com/anaconda/install/index.html) if its not already installed.
2. Create a new [conda environment](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) with python 2.7.12 and activate it. The env name used below `chemotaxis_py2.7.12` is optional, name it as you like.
    
    ```bash
    # create a new conda environemnt
    conda create --name chemotaxis_py2.7.12 python=2.7.12 
    
    # activate it
    conda activate chemotaxis_py2.7.12
    ```
    
3. Clone the chemotaxis-cli repo to the directory of your choice.
    
    ```bash
    # navigate to the directory you want to clone the repo to
    cd <path_to_your_dir>
    
    # clone the chemotaxis-cli repository there
    git clone https://github.com/AndersenLab/chemotaxis-cli.git
    ```
    
4. Install the requirements and `ct` into your environment.
    
    ```bash
    # install the requirements with pip
    pip install -r chemotaxis-cli/requirements.txt
    
    # install ct
    python chemotaxis-cli/setup.py install
    ```
    
5. Run `ct` from within the `chemotaxis_py2.7.12` env following example usage below. If you don’t plan to run `ct` its a good idea to `deactivate` your environment. [See details of conda environment management here.](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html)

---

## Example Usage

1. Activate the conda environment you installed `ct` into.
    
    ```bash
    # activate your chemotaxis env
    conda activate chemotaxis_py2.7.12
    ```
    
2. Navigate to the directory with your chemotaxis assay plate images, then run the `ct` command on images to calculate the chemotaxis index (CI). The command below can be edited to run on example images in the `chemotaxis-cli` repo you cloned in the install.
    
    ```bash
    # navigagte to your directory with images. You can use our examples if desired.
    cd <your_path_to_chemotaxis_cli>/examples/example1
    
    # Run ct on all .jpg files within a directory.
    ct *.jpg  --radius 920 --fp_sigma 1 --crop 20 --center 5 --small 100 --large 1200 --debug --header > results1.txt
    
    # Run on the second example with different parameters
    cd <your_path_to_chemotaxis_cli>/examples/example2
    ct *.jpg  --radius 781 --fp_sigma 2 --crop 20 --center 5 --small 100 --large 1200 --debug --header > results2.txt
    ```
    
3. Process the  `.txt` file as you please. The output variables are defined below:

| fname | q1 | q2 | q3 | q4 | n | total_q | total | ci |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| the file name without file extension | pixel count in top left quadrant | pixel count in top right quadrant | pixel count in bottom left quadrant | pixel count in bottom right quadrant | pixel count in origin (center) | sum of q1-q4 | sum of all | chemotaxis index (CI) |

## ct arguments

- `--radius` sets the radius of n pixels used to find plates in the image. It can be helpful to measure this pixel value in imageJ for one image and provide it to `ct` so the program has an optimal starting radius for finding plates.
- `--fp_sigma` sets the standard deviation of the Gaussian filter used in canny edge detection when finding the plate center and radius. Higher values will find only the most obvious plate features. The default is 1.
- `--crop` will crop the plate radius by n pixels when creating the plate mask. This is useful if `ct` identifies the correct plate radius but the plate edges are identified as objects.
- `--center` sets the denominator value for the equation (plate radius / x), which is used to specify the radius of the origin (center) mask. The origin (center) mask is used to avoid counting nematode objects that have not migrated away from the origin (center) of the plate. The default `--center` value is 5.
- `--small` sets the pixel count used to filter out small objects.
- `--large` sets the pixel count used to filter out large objects.
- `--debug` will create many additional debug files that will help the user assess what objects are being counted and filtered.
- `--header` will attach a column header to output text file.
- `--rev` will reverse the sign of the chemotaxis index. This is helpful if the test and control compounds are switched from the default locations. Normally test compounds are in the top left and bottom right quadrants and controls are in the top right and bottom left quadrants.
