# *ic*OS lab UV-visible spectroscopy toolbox

## Installation

These scripts require the preliminary installation of python 3.__. It uses several packages, only one of which is not a prebuilt and has to be manually installed :

- seaborn (color palette).

The following  packages are used but should be normally shipped with all versions of python.

- Tkinter
- pandas
- matplotlib
- numpy
- os
- scipy
- statistics
- math

## GUI instructions

The GUI implementation uses the same packages as the command line version. It can be executing with the following line :

```powershell
cd path_to_script
python.exe icOS_toolbox_GUI.py
```

Execution of the script should open the following window : 

![image](https://user-images.githubusercontent.com/77961780/212883804-4a6ca7f3-1744-4458-b8c0-c847358cf9e6.png)

Spectra files can then be opened with the first button. However, once the scripts are loaded, they cannot be discarded and if you wish to remove them, you will need to close the window and execute the script again. 
Once opened, the spectra will be plotted in the area under the last button. 

If the scrollbar does not immediately appear, resize the window or click the scroll button. 

The script offers two correction options :

### Constant baseline correction

Using the flat part of the spectrum in the near IR, the script calculates a constant baseline. Each spectrum is then scaled based on the maximum of the given peak (+/- 20 nm) and plotted. 

![image](https://user-images.githubusercontent.com/77961780/212883865-31ffb874-dd30-4079-bdcb-0b630e8ae7e7.png)

### Scattering baseline correction

This correction aims to take out the scattering induced component of the baseline. This component scales inversely proportional to the baseline as scat(λ)=1/λ^4 and is influenced by the shape, size of a protein crystal. In order to accurately compare protein crystal UV-vis absorption spectra, especially in the context of shifts of absorbance peaks, this correction is extremely handy.

In order to compute the correction, the script uses

- boundaries of the flat part of your spectrum (near IR) to fit the constant baseline (red in the Fig. 1 (a) curve).
- One additional peak-less area, optimally on the UV side of your absorbance peak(s) of interest(green in the Fig. 1 (a) curve). This one is used for the scattering correction fit  (orange in the Fig. 1 (b) curve).
- The maximum of absorbance your peak of interest (used to scale all of your spectra)
- A leftmost segment that is being automatically selected (non-specific)

It then plots a diagnostic plot for each of the spectra. This diagnostic plot contains the original spectrum in blue, the fitted baseline in orange and the selected segments in purple (flat baseline), green (peakless segment) and red (leftmost segment)

![image](https://user-images.githubusercontent.com/77961780/212883919-2ea9be22-a5fc-4347-8fd9-fe651be46955.png)

## Command line crystal spectroscopic corrections (*ic*OS data)

icOS_specorr.py is the script you want to use for command-line baseline correction. It is meant to be executed in the same folder as your icOS data files (.txt)

Upon execution it asks for:

- boundaries of the flat part of your spectrum (near IR) to fit the constant baseline (red in the Fig. 1 (a) curve).
- One additional peak-less area, optimally on the UV side of your absorbance peak(s) of interest(green in the Fig. 1 (a) curve). This one is used for the scattering correction fit  (orange in the Fig. 1 (b) curve).
- The maximum of absorbance your peak of interest (used to scale all of your spectra)
- A leftmost segment that is being automatically selected (non-specific)

![image](https://user-images.githubusercontent.com/77961780/212883991-d2d07202-29e6-48ce-b685-0b3badcc1ffe.png)  


![image](https://user-images.githubusercontent.com/77961780/212884032-3528d09e-7e91-49f7-bda0-c69bf53c4ffe.png)

It produces several outputs : 

- raw spectra figures and .txt files
- constant baseline corrected and scaled figures and .txt files
- scattering corrected figures and .txt files

You can use the .txt files to make final figures with whatever plotting software you prefer if the produced one do not suit you

in case of question/error, please contact me at nicolas.caramello@esrf.fr
