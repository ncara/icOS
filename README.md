# *ic*OS lab UV-visible spectroscopy toolbox

## Installation

These scripts require the preliminary installation of python 3.__. It uses several packages, only one of which is not a prebuilt and has to be manually installed:

- seaborn (color palette).
- wxPython (Graphical User Interface library)

For those not able to use wxPython, a plceholder clunkier version using tkinter is available (icOS_toolbox_GUI-nowxPython.py)

The following  packages are used but should be normally shipped with all versions of python.

- pandas
- matplotlib
- numpy
- os
- scipy
- statistics
- math

## GUI instructions

The GUI implementation uses the same packages as the command line version. It can be executing with the following line:

```powershell
cd path_to_script
python.exe icOS_toolbox_GUI.py
```

Execution of the script should open the following window: 

![image](https://user-images.githubusercontent.com/77961780/213933960-94098bd7-90ea-4555-a0a1-3e029dfa6de4.png)

Spectra files can then be opened with the first button. However, once the scripts are loaded, they cannot be discarded and if you wish to remove them, you will need to close the window and execute the script again. 
Once opened, the spectra will be plotted in the right panel. 

![image](https://user-images.githubusercontent.com/77961780/213933967-aa25f9e9-9c2b-4e2f-9e8f-26730a8bed79.png)

The toolbox at the bottom of the plot can be used to move in the plot and zoom in on the parts of interest.

![image](https://user-images.githubusercontent.com/77961780/213933970-ca54ae7e-0b16-4caf-a6ad-9d9fa9d88311.png)

The script offers two correction options:

### Constant baseline correction

You can input boundaries for a flat part of the spectrum in the second and third fields. Ideally choose a segment near IR, the script calculates a constant baseline based on the average absorbance in this segment and subtracts it to the raw data. Each spectrum is then scaled based on the maximum of the absorbance peak of interest (+/- 10 nm) inputed in the first field. Finally, each spectrum is smoothed through a Savitzky-Golay filter and then plotted. 

![image](https://user-images.githubusercontent.com/77961780/213933980-e91b954c-8c77-46b3-a453-48f86d742cf4.png)

### Scattering baseline correction

This correction aims to take out the scattering induced component of the baseline. This component scales inversely proportional to the baseline as scat(λ)=1/λ^4 and is influenced by the shape, size of a protein crystal. In order to accurately compare protein crystal UV-vis absorption spectra, especially in the context of shifts of absorbance peaks, this correction is extremely handy.

In order to compute the correction, the script uses

- boundaries of the flat part of your spectrum (near IR) to fit the constant baseline (red in following curve).
- One additional peak-less area derived from boundaries inputed in the fourth and fifth fields. Ideally, choose a segment on the UV side of your absorbance peak(s) of interest (green in the following curve). This one is used for the scattering correction fit  orange in the following plot.
- The maximum of absorbance your peak of interest (used to scale all of your spectra)
- A leftmost segment that is being automatically selected (non-specific, magenta in the following curve

A baseline function in 1/λ^4 is then fitted on the segment, and a diagnostic plot is produced for each of the spectra. 

![image](https://user-images.githubusercontent.com/77961780/213933993-1d2d30f1-b72f-4d2a-b7eb-c0bf41ea6f2a.png) ![image](https://user-images.githubusercontent.com/77961780/213934011-d391d334-9038-4fd5-8025-2a5881bf8cea.png)

The corrected, scaled and smoothed spectra are plotted within the right panel.

![image](https://user-images.githubusercontent.com/77961780/213934020-5f5841f3-207a-4483-8f17-12b9720832e1.png)

## Command line crystal spectroscopic corrections (*ic*OS data)

icOS_specorr.py is the script you want to use for command-line baseline correction. It is meant to be executed in the same folder as your icOS data files (.txt)

Upon execution it asks for:

- boundaries of the flat part of your spectrum (near IR) to fit the constant baseline (red in the Fig. 1 (a) curve).
- One additional peak-less area, optimally on the UV side of your absorbance peak(s) of interest(green in the Fig. 1 (a) curve). This one is used for the scattering correction fit  (orange in the Fig. 1 (b) curve).
- The maximum of absorbance your peak of interest (used to scale all of your spectra)
- A leftmost segment that is being automatically selected (non-specific)

![image](https://user-images.githubusercontent.com/77961780/213933993-1d2d30f1-b72f-4d2a-b7eb-c0bf41ea6f2a.png) ![image](https://user-images.githubusercontent.com/77961780/213934011-d391d334-9038-4fd5-8025-2a5881bf8cea.png)

It produces several outputs: 

- raw spectra figures and .txt files
- constant baseline corrected and scaled figures and .txt files
- scattering corrected figures and .txt files

You can use the .txt files to make final figures with whatever plotting software you prefer if the produced one do not suit you

in case of question/error, please contact me at nicolas.caramello@esrf.fr
