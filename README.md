# *ic*OS lab UV-visible spectroscopy toolbox

These scripts require the preliminary installation of several packages:

- seaborn (color palette), the only one which is not pre-packaged with python
- pandas
- matplotlib
- numpy
- os
- scipy
- statistics
- math

## Single crystal spectroscopic corrections (*ic*OS data)

icOS_specorr.py is the script you want to use for baseline correction. It is meant ot be executed in the same folder as your icOS data files (.txt)

Upon execution it asks for:

- boundaries of the flat part of your spectrum (near IR) to fit the constant baseline (red in the first curve below).
- One additional peak-less area, optimally on the UV side of your absorbance peak(s) of interest(green in the first curve below). This one is used for the scattering correction fit  (orange in the second curve below).
- The maximum of absorbance your peak of interest (used to scale all of your spectra)

![image](https://user-images.githubusercontent.com/77961780/192751464-90320cf1-f04c-4958-a3c4-9aff9a339236.png)

![image](https://user-images.githubusercontent.com/77961780/192751395-121150ac-7ad3-498f-b07f-ec238118a736.png)



The script produces several outputs: 

- raw spectra figures and .txt files
- constant baseline corrected and scaled figures and .txt files
- scattering corrected figures and .txt files

You can use the .txt files to make final figures with whatever plotting software you prefer if the produced one do not suit you

in case of question/error, please contact me at nicolas.caramello@esrf.fr
