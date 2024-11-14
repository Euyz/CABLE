## Code dependencies.  
Code dependencies are in 'requirement.txt'.  
```sh
pip install -r requirements.txt
```
The code uses GPU to accelerate the computation, if using CPU, please change cupy to numpy.

## Example data and usage.  
### Example data
* The example data 'CJ4ROI.ims' can be accessed via [http://cable.bigconnectome.org](http://cable.bigconnectome.org).
### OS Requirements
* It is recommended to run this program under Linux.
### Usage
* We use 'Mrtrix3' for estimating response function(or use response.nii in this repository directly) and constrained spherical deconvolution(CSD), which can be accessed via 'https://www.mrtrix.org/download/'.  

* After installing 'Mrtrix3' and downloading 'CJ4ROI.ims', execute the python file:

```sh
python main.py CJ4TEST.h5
```

* If the code in 'main.py' encounters difficulties in calling 'MRtrix3'(e.g. executing on Windows) please execute it in command line manually.
* The program will generate Fiber Orientation Distribution (FOD) and tractography (.tck) files which can be viewed with 'mrview' in 'Mrtrix3', use 'ODF Display' and 'Tractography' in 'Tools' toolbar to view them.


![image ](https://github.com/Euyz/CABLE/assets/33593212/e1d11bad-6171-4077-97b4-680b15ebdd21)
![image](https://github.com/Euyz/CABLE/assets/33593212/76fca208-a825-4109-bf2c-1382c2fbb889)

