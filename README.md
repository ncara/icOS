# The _ic_ OS toolbox

The *in crystallo* Optical Spectroscopy (*ic*OS) toolbox is a suite of tools encased in a graphical interface, designed to make the recording and processing of absorbance and fluorescence data (both in solution and *in crystallo*) more straightforward. It was originally developped at and for the *ic*OS lab setups, but is meant to be usable by external users. Currently, it supports text based format inputs from Ocean Optics, JASCO and Avantes (table of available format is at the bottom of this README file). 

I intend to make the toolbox as broadly available as possible. If your files are not correctly loaded in the toolbox, please contact me at [nicolas.caramello@esrf.fr](mailto:nicolas.caramello@esrf.fr) with a copy of your files. 

It consists in three tabs: (a) the main tab, where spectra can be corrected and plotted ; (b) the kinetic tab, where kinetic curve, difference spectra, and singular value decomposition can be calculated, and lastly the expert tab, where the smoothing, modelling and correction settings can be finely adjusted. Finally, by right clicking a plot and selecting the ‘configuration’ option, a plot-customisation window can be accessed. 

# Installation

The *ic*OS toolbox runs on python. For package availability, python version 3.12.6 is preferred. Following is a quick guide for the installation of python and needed dependencies. 

![image](https://github.com/user-attachments/assets/b2354214-57a7-4188-b724-c72692526e87)


### Installing python:

For machines running windows, the recommended version of python is available here: https://www.python.org/downloads/windows/. For Linux / MacOS machines, installing python through the package manager of your OS (homebrew for MacOS, apt for Ubuntu/Debian, *ect*) is preferred, and if possible please use python 3.11 as this the version that I used for testing. 

### Installing packages:

Running the icOS toolbox script as follows will result in the script attempting to install the necessary packages using pip: 

```bash
python icOS_toolbox.py
```

The first run of the script might produce an error right after having installed the package wxPython, if so a second run of the script using the same command will succeed in installing all dependencies. 

### Creating a conda environnement for the toolbox.

If you wish to encase the toolbox into a conda environnement, all dependancies can be installed through conda with the following command

```bash
conda create -n icOS-toolbox python conda-forge::wxmplot seaborn numpy scipy pandas 
```

Once the environment is created, the script can be run as for the previous section. 

# Spectra correction

A series of physical phenomena, which affect differently different crystals or orientations, complicate the direct comparison of *ic*AS data to in solution AS data. Spectra coming from different crystals should be baseline adjusted and eventually scaled. 

### Constant baseline adjustment

Various phenomena previously described contribute to flatly raising the baseline of an *ic*AS spectra. This is easily corrected, provided the spectrum features a region devoid of absorption (this is usually the case in the red to near-IR region). The average of absorption in this band is subtracted from each spectrum to bring them onto a common baseline, as is visible in the plots below (a) raw to (b) corrected. In the app, this function is called “constant-baseline correction”. 

![image 1](https://github.com/user-attachments/assets/74a5dd1b-89b5-43b4-a35d-b1c2a85fb234)


Spectra from different crystals or orientations should therefore be scaled based on a conserved absorption peak. The choice of a peak can be inferred from prior knowledge in solution data. 

### Scattering baseline subtraction

This adjustment might not be enough in some cases, as for instance for the light-green coloured spectrum in the plots above. The contribution of Rayleigh scattering as well as that of remaining focal spot displacement and reflection must be modelled to allow *ic*AS data recorded on different crystals to be compared.   

For the estimation of the contribution of scattering, and other phenomena, the baseline model is fitted against three non-absorbing (supposed baseline) segments of the spectra via the least-square minimization method (lime, magenta and dark green segments in the plots below). Ideally, two of these segments are on either side of the recorded range (dark green and lime segments in the plots below), where Rayleigh scattering is respectively strongest. Because the UV segment is sometimes unreliable (loss of signal through the optics), a third segment, between the UV range and the absorption peak of interest (magenta in the plot below), is used to fit the baseline model (red in the plot below). Additionally, a divergence factor (always positive, 1 by default) can be supplied to decrease the weight of each segment in the fit of the scattering baseline. This divergence factor should be inversely proportional to the length of its segment and increased if the segment is less reliable. The segments on the left and right side of the region of interest are user inputted (blue-side peakless and red-side peakless in the app). In case absorbance does not go back to the baseline between the absorption peak of interest and the UV-range, a percentage of the maximum absorbance peak can be supplied to create a constant offset between the fit and the absorbance (Fig. S1). A diagnostic plot is generated for each spectrum. In this diagnostic plot, segments used in the fit are coloured (lime, magenta and dark-green), and the fit baseline is overlaid (red) for assessment of the background correction quality (an example of the diagnostic plot is given below). The range and divergence factor of each segment should be adjusted so that the fit baseline is superposed to the segments. Finally, the modelled contribution of both phenomena as well as the flat baseline can be subtracted from the raw spectrum, effectively bringing the baseline to 0 (right panel of the plot below). 

![image 2](https://github.com/user-attachments/assets/1f4a2bd0-8eb6-4eb7-b581-5686ad64ff10)


### Laser dent removal

Because of the duration of integration of the spectrophotometer in the TR*-ic*OS setup, the tail of the nanosecond laser used to initiate the reaction in crystals can also contribute to the absorption spectrum, in the form of a negative dip, or dent in the absorption spectrum (plot below). The second derivative of the absorption spectra is calculated, and its local minima identify the absorption dips, as well as their edges. The largest absoption dip (in amplitude) marks the contribution of the nanosecond laser to the spectrum, while all other dips are marked by red dots. The data points corresponding to the contribution of the laser are removed. In the following spectra, only the main dent in the area of the previously detected laser dent is marked, and the corresponding points are also removed.

![image 3](https://github.com/user-attachments/assets/7f023b2a-3ee8-493c-b5cc-384f4cb80e91)

### Quality score

Absorbance is calculated by comparison to a reference signal as I0, but the emission spectrum of the polychromatic light sources used for *ic*OS exhibit low photon count regions. These regions typically exhibit noisy peak-like features which might appear meaningful to the untrained eye. In order to easily assess the validity of a feature, we implemented a confidence-score based on the photon count in the reference signal for each measurement. This confidence score is visualised as a colour scale from blue (trustworthy) to red (untrustworthy) via the ‘expert features’ panel (plot below). When raw photon counts are available, a confidence score is calculated to identify features of a spectrum that might originate from a low count in both the I0 blank signal and the I signal. 

![image 4](https://github.com/user-attachments/assets/67ee81a6-8761-44ae-8749-a9fa6eeca4b2)

# Kinetic analysis

The icOS toolbox allows to plot series of spectra, and calculate difference spectra (spectra(t) - spectra(0)), and plot absorbance over time (examples shown below). All of these features are available in the ‘kinetic’ tab. 

The icOS toolbox currently allows the fit of a mono-exponential decay or rise, as well as the Hill Equation. The produced reaction rate constant provides an estimation of an intermediate-state lifetime and can be used to plan a TR-MX experiment. 

![image 5](https://github.com/user-attachments/assets/4dabc7e7-7f5a-4824-b0f4-137c008bdad1)


In case of issue during the installation or use of the app, please contact me at nicolas.caramello@esrf.fr
