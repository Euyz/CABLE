## Code dependencies  
Code dependencies are in requirement.txt.  
## Example data and usage.  
The example data 'CJ4TEST.h5' can be accessed via XXX.

```sh
python main.py CJ4TEST.h5
```

We use 'Mrtrix3' for estimating response function(or use response1.nii in this repository directly) and constrained spherical deconvolution(CSD), which can be accessed via 'https://www.mrtrix.org/download/', if the code in 'main.py' encounters difficulties in calling 'Mrtrix3'(e.g. executing on Windows) please execute it manually.  

The program will generate a FOD file which can be viewed with 'mrview' in 'Mrtrix3', please open the FOD file and use 'ODF Display' in 'Tools' to view the FOD file.  
