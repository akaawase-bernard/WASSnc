# Creates NetCDF from WASS data
Generates a netcdf file from processed Wave Acquisation Stereo System [WASS](https://github.com/fbergama/wass)
 point cloud. 

# How to Use
The main script for this job is `generateWASSnc.py` while all the functions needed to process the NetCDF file are available in `wavespec.py`

You can run the script via terminal or any IDE, before running it you may want to edit the inputs through a editor of your choosing (e.g [VScode](https://code.visualstudio.com/)). Things to edit include, number of frames you wish to process, path where the `yourfilename.nc` should be saved. To run the script use:  

```shell
python generateWASSnc.py
```
You can also just copy the content of the script and run it in jupyter notebook.
