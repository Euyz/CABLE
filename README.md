## Code dependencies  
Code dependencies are in requirement.txt.  
```sh
pip install -r requirements.txt
```

## Example data and usage.  
* The example data 'CJ4TEST.h5' can be accessed via XXX.

* We use 'Mrtrix3' for estimating response function(or use response.nii in this repository directly) and constrained spherical deconvolution(CSD), which can be accessed via 'https://www.mrtrix.org/download/'.  

* After installing 'Mrtrix3', execute the python file:

```sh
python main.py CJ4TEST.h5
```

* If the code in 'main.py' encounters difficulties in calling 'Mrtrix3'(e.g. executing on Windows) please execute it manually.  
* The program will generate a FOD file which can be viewed with 'mrview' in 'Mrtrix3', open the FOD file and use 'ODF Display' in 'Tools' to view the FOD file.  
![image](https://github.com/Euyz/CABLE/assets/33593212/fc2df123-4574-40ee-b5c6-5b3f2e782cdf)
