[![Build Status](https://travis-ci.org/danielecook/python-cli-skeleton.svg?branch=master)](https://travis-ci.org/danielecook/python-cli-skeleton) [![Coverage Status](https://coveralls.io/repos/github/danielecook/python-cli-skeleton/badge.svg?branch=master)](https://coveralls.io/github/danielecook/python-cli-skeleton?branch=master)

# chemotaxis-cli

*A command-line interface to measure chemotaxis index (CI) from chemotaxis assay images*
---
![ct_repo_workflow.png](https://github.com/AndersenLab/chemotaxis-cli/blob/master/img/ct_repo_workflow.png)
---
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
- `--fp_sigma` float, sets the standard deviation of the Gaussian filter used in canny edge detection when finding the plate center and radius. Higher values will find only the most obvious plate features. The default is 1.
- `--crop` will crop the plate radius by n pixels when creating the plate mask. This is useful if `ct` identifies the correct plate radius but the plate edges are identified as objects.
- `--center` sets the denominator value for the equation (plate radius / x), which is used to specify the radius of the origin (center) mask. The origin (center) mask is used to avoid counting nematode objects that have not migrated away from the origin (center) of the plate. The default `--center` value is 5.
- `--small` sets the pixel count used to filter out small objects.
- `--large` sets the pixel count used to filter out large objects.
- `--debug` will create many additional debug files that will help the user assess what objects are being counted and filtered.
- `--header` will attach a column header to output text file.
- `--rev` will reverse the sign of the chemotaxis index. This is helpful if the test and control compounds are switched from the default locations. Normally test compounds are in the top left and bottom right quadrants and controls are in the top right and bottom left quadrants.

## Experimental workflow

1. Bleach synchronize strains to be assayed.
2. Use the COPAS BIOSORT with our custom chemotaxis plate holder to sort 50 synchronized L4 animals to the origin of chemotaxis assay plates.
    1. The [CAD files and drawings required to fabricate our 3 part plate holder are here](https://github.com/AndersenLab/chemotaxis-cli/tree/master/customPlateHolder).
    2. The parts are assembled from bottom to top in the following order: base, support, then plates.
3. Spot control and test compounds onto the plates using a multi-channel pipet. The spot locations are etched into the chemotaxis plate holder for convenience.
4. If necessary, wick the M9 fluid used to dispense the nematodes to the center of the chemotaxis plates away using the corner of a kimwipe.
5. Once the dispensing fluid is removed allow the nematodes to respond to the compounds for 1 hour at 20°C.
6. After 1 hour, transfer the chemotaxis plates to 4°C prior to imaging.
7. Image the plates within 5 days, when imaging ensure that the edges of the plate are visible so that ct can setup the plate regions correctly.
8. Prior to analyzing images with `ct`, open one image in [imageJ](https://imagej.nih.gov/ij/index.html) and manually measure the diameter of a plate in pixels.
    1. Select the straight line tool
    2. Draw a line across the diameter of the plate then click `command+m` to record a Length measurement. This is the diameter of the plate in pixels, divide by 2 to get the plate radius.
9. Run `ct` on your images and provide your plate radius `--radius <your radius>` as an argument.
10. Process the results
