Code dependencies are in requirement.txt.  
Please modify the variable ‘address’ in the 'mainPY' file to the address of the example data(can be accessed via XXX).  
We use 'Mrtrix3' for estimating response function(or use response1.nii in this repository directly) and constrained spherical deconvolution(CSD), which can be accessed via 'https://www.mrtrix.org/download/', if the code in 'mainPY' encounters difficulties in calling mrtrix3 (e.g. executing on Windows) please execute it manually.  
The program will generate a FOD file which can be viewed with 'mrview' in 'Mrtrix3', open the FOD file and use 'ODF Display' in 'Tools' to view the FOD file.  
