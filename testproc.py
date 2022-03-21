# -*- coding: utf-8 -*-
"""
Created on Sun Nov 21 17:27:40 2021

@author: NCARAMEL
"""

import pandas as pd #This is the DataFrame package, DataFrames are like excell tables with neat properties
import matplotlib.pyplot as plt #this is the plot package
import numpy as np   #Numerical operations such as exp, log ect
import os as os      # Architecture package, to handle directories ect
import scipy as sp   #Some useful tools, especially for processing spectra in scipy.signal
from scipy.optimize import curve_fit # I don't know why but that one function refuses to come with the rest of the package, it gets its own import
import scipy.signal 
import seaborn as sns    #Visually distinct color palette 
import numpy as np
from sklearn.linear_model import LinearRegression
import scipy.sparse as sparse
from statistics import mean
#import scipy.spsolve as spsolve

os.chdir('./') # pour definir ton dossier de travail (ou les sorties du programme seront sauvegardées)

def closest(lst, K): 
     lst = np.asarray(lst) 
     idx = (np.abs(lst - K)).argmin() 
     return(lst[idx])

# def fct_baseline(x, a, b):
#     return(np.log(1/(1-a/np.power(x,4))))

def fct_baseline(x, a, b):
    return(a/np.power(x,4)+b)

# def fct_baseline1(x, a, b):
#     return(ax+b)

def rescale_corrected(df, wlmin, wlmax):
    a=df.copy()
    # offset=a.A[a.wl.between(wlmax-10,wlmax-1)].min()
    # a.A=a.A.copy()-offset        
    fact=1/a.A[a.wl.between(wlmin,wlmax,inclusive=True)].max() #1/a.A[a.wl.between(425,440)].max()
    a.A=a.A.copy()*fact      
    return(a.copy())

def rescale_raw(df,rawdata, wlmin, wlmax):
    a=df.copy()
    raw=rawdata.copy()
    # offset=a.A[a.wl.between(wlmax-10,wlmax-1)].min()
    # a.A=a.A.copy()-offset        
    fact=1/a.A[a.wl.between(250,400,inclusive=True)].max() #1/a.A[a.wl.between(425,440)].max()
    raw.A=raw.A.copy()*fact      
    return(raw.copy())

def baseline_als(y, lam, p, niter=10):
    L = len(y)
    D = sparse.diags([1,-2,1],[0,-1,-2], shape=(L,L-2))
    D = lam * D.dot(D.transpose()) # Precompute this term since it does not depend on `w`
    w = np.ones(L)
    W = sparse.spdiags(w, 0, L, L)
    for i in range(niter):
        W.setdiag(w) # Do not create a new matrix, just update diagonal values
        Z = W + D
        z = sparse.linalg.spsolve(Z, w*y)
        w = p * (y > z) + (1-p) * (y < z)
    return z

def baselinefitcorr_3seg_smooth(df,  segment1, segment2, segmentend, sigmaby3segment):
    #segmentend=df.wl.between(600,800)
    segment=segment1+segment2+segmentend
    #min1 = closest(df.wl,minrange1)
    #min2 = closest(df.wl,minrange2)
    x=df.wl[segment].copy()
    y=df.A[segment].copy()
    initialParameters = np.array([1e9, 0])
    # sigma=[1,0.01,1,0.01]
    n=len(tmp.A[segment1])
    sigma=n*[sigmaby3segment[0]]
    n=len(tmp.A[segment2])
    sigma=sigma + n*[sigmaby3segment[1]]
    n=len(tmp.A[segmentend])
    sigma=sigma + n*[sigmaby3segment[2]]
    print(sigma)
    para, pcov = sp.optimize.curve_fit(f=fct_baseline, xdata=x, ydata=y, p0=initialParameters, sigma=sigma)
    baseline=df.copy()
    baseline.A=fct_baseline(baseline.wl.copy(), *para)
    #baseline.A=baseline_als(np.array(y), 10^5, 0.01)
    # plt.plot(df.wl,df.A)
    # plt.plot(baseline.wl, baseline.A)
    # plt.show()
    corrected=df.copy()
    corrected.A=df.A.copy()-baseline.A
    co={'wl' : x,
        'Sigma' : sigma}
    correction=pd.DataFrame(co)
    return(corrected, correction)


def baselineconst(df,segmentend):
    a=df.copy()
    baseline=df.copy()
    baseline.A=mean(a.A[segmentend])
#    plt.plot(a.wl,a.A)
#    plt.plot(baseline.wl, baseline.A)
#    plt.show()
    corrected=a.copy()
    corrected.A=a.A.copy()-baseline.A
    return(corrected)






#defining size of in-gui figures, /2.54 because inches are for heretics
plt.rcParams["figure.figsize"] = (40/2.54,30/2.54)

####Parsing through the dir to find spectra and importing/treating them#### 
directory = "./"  #Because of the genius who decided to put nearly all of our files in "Work Folder" and not "Work_Folder" we actually have to double backslash every path
raw_spec={}                #This is a Dictionary, it's one of python's list-like objects, it's main properties are : not ordered, can be changed ==> We're using it to store our spectra and index them by the name of the file
ready_spec={}               #Last one was for the brute files, this one is for the scaled ones
ready_spec_nolam4={}
for entry in os.scandir(directory):   #The equivalent of "For f in ./*" in bash
    if entry.path.endswith(".txt") and entry.is_file() :   #We're only interested in spectra, hopefully they are .csv, if not, you can still change the extension
        a=entry.path[2:-4]                               #This is to keep only the name, in my case, I have 59 characters of path before the name of your file, and 4 char which make up the extension
        print(a)
        tmp=pd.read_csv(filepath_or_buffer= entry.path,   #pandas has a neat function to read char-separated tables
                        sep= "\t",                         #Columns delimited by semicolon
                        decimal=".",                      #decimals delimited by colon
                        skiprows=17,                      #There is 18 rows of header
                        skip_blank_lines=True,            #There is a blank line at the end
                        skipfooter=2,                     #2 lines of footer, not counting the blank line
                        names=['wl','A'],                 #The whole scripts relies on column names being 'wl' for Wavelength and 'A' for Absorbance
                        engine="python")                  #The python engine is slower but supports header and footers
        tmp.index=tmp.wl                                  #We want rows to be indexed on wl
        tmp.origin=len(tmp.wl)*[0]
        raw_spec[a]=tmp.copy()                                  #We're storing the table in the dictionnary
        tmp.A=sp.signal.savgol_filter(x=tmp.A.copy(),     #This is the smoothing function, it takes in imput the y-axis data directly and fits a polynom on each section of the data at a time
                                       window_length=21,  #This defines the section, longer sections means smoother data but also bigger imprecision
                                       polyorder=3)       #The order of the polynom, more degree = less smooth, more precise (and more ressource expensive)
#        plt.plot(tmp.wl,tmp.A)
#        plt.plot(tmp.wl, tmp.origin)
#        plt.show()
        maxwl=250 #tmp.wl.min()
        # minrange1=tmp.A[tmp.wl.between(maxwl,280)].idxmin()
        # print(minrange1)
        # minrange2=tmp.A[tmp.wl.between(280,300)].idxmin()
        # print(minrange2)
        # minrange3=closest(tmp.wl,400)
        
        rightborn=tmp.A[tmp.wl.between(200,250)].idxmax()-10
        leftborn=rightborn-5
        # rightborn=tmp.A[tmp.wl.between(leftborn,280)].idxmin()
        # segment1=tmp.wl.between(235,245, inclusive='both')
        segment1=tmp.wl.between(leftborn,rightborn, inclusive='both')
        leftborn=tmp.A[tmp.wl.between(290,320)].idxmin()-5
        rightborn=tmp.A[tmp.wl.between(290,320)].idxmin()
        segment2=tmp.wl.between(leftborn,rightborn, inclusive='both')
        leftborn=closest(tmp.wl,505) #tmp.A[tmp.wl.between(500,rightborn)].idxmax()
        rightborn=closest(tmp.wl,605) #tmp.A[tmp.wl.between(500,900)].idxmin()        
        segmentend=tmp.wl.between(leftborn,rightborn, inclusive='both')
        # plt.plot(tmp.wl,tmp.A)
        # plt.plot(tmp.wl[segment1], tmp.A[segment1])
        # plt.plot(tmp.wl[segment2], tmp.A[segment2])
        # plt.plot(tmp.wl[segmentend], tmp.A[segmentend])
        # plt.plot(tmp.wl, tmp.origin)
        # plt.show()
        sigmafor3segment=[1,0.1,0.1]
        #sigmafor2segment=[1,1]
        tmp2, correction=baselinefitcorr_3seg_smooth(tmp,  segment1, segment2, segmentend, sigmafor3segment) # tmp.A[tmp.wl.between(280,432)].idxmin()
        tmp2_nolam4=baselineconst(tmp,segmentend)
        
        # plt.plot(tmp2.wl,tmp2.A)
        # plt.plot(tmp.wl, tmp.origin)
        # plt.show()
        
        # plt.plot(tmp2_nolam4.wl,tmp2_nolam4.A)
        # plt.plot(tmp.wl, tmp.origin)
        # plt.show()
        
        tmp3=rescale_corrected(tmp2, 400,500)
        # plt.plot(tmp3.wl,tmp3.A)
        # plt.plot(tmp.wl, tmp.origin)
        # plt.show()    
        
        tmp3_nolam4=rescale_corrected(tmp2_nolam4, 400,500)
        # plt.plot(tmp3_nolam4.wl,tmp3_nolam4.A)
        # plt.plot(tmp.wl, tmp.origin)
        # plt.show()  
        
        ready_spec[a]=tmp3
        ready_spec_nolam4[a]=tmp3_nolam4
        
        
        
        
fig, ax = plt.subplots()     #First let's create our figure, subplots ensures we can plot several curves on the same graph
ax.set_xlabel('Wavelength [nm]', fontsize=10)  #x axis 
ax.xaxis.set_label_coords(x=0.5, y=-0.08)      #This determines where the x-axis is on the figure 
ax.set_ylabel('Absorbance [AU]', fontsize=10)               #Label of the y axis
ax.yaxis.set_label_coords(x=-0.1, y=0.5)       #position of the y axis 

# ax = ax1.twinx()                                #Guillaume did that weird OD + Abs y-axis, don't know why, we can probably remove that and replace ax and #  ax1 by simply "ax"
# ax.set_ylabel('Absorbance [AU]', fontsize=10)  #label of our secondary y-axis 
# ax.yaxis.set_label_coords(x=1.09, y=0.5)        #Position of our secondary y-axis
palette=sns.color_palette(palette='bright', n_colors=len(ready_spec))   #This creates a palette with distinct colors in function of the number of sample, check it at https://seaborn.pydata.org/tutorial/color_palettes.html, in our case we might want to cherry-pick our colors, that's easy: palette are only lists of rgb triplets. Seaborn has a "desat" var, it modulates intensity of the color we can probably use that for emission/excitation plots 

n=0                                            #this is just a counter for the palette, it's ugly as hell but hey, it works 
for i in ready_spec :                          #We can then parse over our dictionary to plot our data
    ax.plot(ready_spec[i].wl,                  #x-axis is wavelength
              ready_spec[i].A ,                   #y-axis is abs, or emission, or else
              linewidth=1,                    #0.5 : pretty thin, 2 : probably what Hadrien used 
              label=i,                        #Label is currently the name of our file, we could replace that by a list of names
              color=palette[n])               #This determines the color of the curves, you can create a custom list of colors such a c['blue','red'] ect
    n=n+1
ax.set_title('scattering corrected in crystallo absorbance spectra', fontsize=10, fontweight='bold')  #This sets the title of the plot
# ax.set_xlim([260, 600])                        
# ax.set_ylim([-0.1, 1.1])
ax.set_xlim([250, 800]) 
ax.set_ylim([-0.3, 1.1])

#  ax1.tick_params(labelsize=10)   #This sets the font, we can probably use stuff like Arial Narrow
ax.tick_params(labelsize=10)
# #  ax1.yaxis.set_ticks(np.arange(0, 1.1, 0.5), pad=30)
ax.yaxis.set_ticks(np.arange(-0.3, 1.1, 0.1))  #This modulates the frequency of the x label (1, 50 ,100 ect)

legend = plt.legend(loc='lower right', shadow=True, prop={'size':7})    #Where the legend rectangle is 

# plt.show()    #Use this to check your figure before the output 

# ################################################

figfilename = "scattering_corrected_spec.pdf"             #Filename, for the PDF output, don't forget to close the last one before executing again, otherwise python can't write over it
plt.savefig(figfilename, dpi=1000, transparent=True,bbox_inches='tight')   #Transparent setting removes the background grid which is the standard for pyplot. 
figfilename = "scattering_corrected_spec.png"
plt.savefig(figfilename, dpi=1000, transparent=True,bbox_inches='tight')   #Filename, for the png output
plt.close()


fig, ax = plt.subplots()     #First let's create our figure, subplots ensures we can plot several curves on the same graph
ax.set_xlabel('Wavelength [nm]', fontsize=10)  #x axis 
ax.xaxis.set_label_coords(x=0.5, y=-0.08)      #This determines where the x-axis is on the figure 
ax.set_ylabel('Absorbance [AU]', fontsize=10)               #Label of the y axis
ax.yaxis.set_label_coords(x=-0.1, y=0.5)       #position of the y axis 

# ax = ax1.twinx()                                #Guillaume did that weird OD + Abs y-axis, don't know why, we can probably remove that and replace ax and #  ax1 by simply "ax"
# ax.set_ylabel('Absorbance [AU]', fontsize=10)  #label of our secondary y-axis 
# ax.yaxis.set_label_coords(x=1.09, y=0.5)        #Position of our secondary y-axis
palette=sns.color_palette(palette='bright', n_colors=len(ready_spec))   #This creates a palette with distinct colors in function of the number of sample, check it at https://seaborn.pydata.org/tutorial/color_palettes.html, in our case we might want to cherry-pick our colors, that's easy: palette are only lists of rgb triplets. Seaborn has a "desat" var, it modulates intensity of the color we can probably use that for emission/excitation plots 

n=0                                            #this is just a counter for the palette, it's ugly as hell but hey, it works 
for i in ready_spec :                          #We can then parse over our dictionary to plot our data
    ax.plot(ready_spec_nolam4[i].wl,                  #x-axis is wavelength
            ready_spec_nolam4[i].A ,                   #y-axis is abs, or emission, or else
            linewidth=1,                    #0.5 : pretty thin, 2 : probably what Hadrien used 
            label=i,                        #Label is currently the name of our file, we could replace that by a list of names
            color=palette[n])               #This determines the color of the curves, you can create a custom list of colors such a c['blue','red'] ect
    n=n+1
ax.set_title('only scaled in crystallo absorbance spectra (no scattering correction)', fontsize=10, fontweight='bold')  #This sets the title of the plot
# ax.set_xlim([260, 600])                        
# ax.set_ylim([-0.1, 1.1])
ax.set_xlim([250, 800]) 
ax.set_ylim([-0.1, 1.5])

#  ax1.tick_params(labelsize=10)   #This sets the font, we can probably use stuff like Arial Narrow
ax.tick_params(labelsize=10)
# #  ax1.yaxis.set_ticks(np.arange(0, 1.1, 0.5), pad=30)
ax.yaxis.set_ticks(np.arange(-0.1, 1.5, 0.1))  #This modulates the frequency of the x label (1, 50 ,100 ect)

legend = plt.legend(loc='upper right', shadow=True, prop={'size':7})    #Where the legend rectangle is 

# plt.show()    #Use this to check your figure before the output 


# ################################################

figfilename = "constant-baseline_corrected_spec.pdf"             #Filename, for the PDF output, don't forget to close the last one before executing again, otherwise python can't write over it
plt.savefig(figfilename, dpi=1000, transparent=True,bbox_inches='tight')   #Transparent setting removes the background grid which is the standard for pyplot. 
figfilename = "constant-baseline_corrected_spec.png"
plt.savefig(figfilename, dpi=1000, transparent=True,bbox_inches='tight')   #Filename, for the png output
plt.close()





# raw_spectra=pd.DataFrame()
# raw_spectra.index=raw_spec.wl
# raw_spectra['myname']=smooth_spec['M48-1_crystal_newspec_n2'].A


towrite_raw_spectra=tmp.drop(columns=['wl','A']) #structure for the written table
for spec in raw_spec:
    towrite_raw_spectra[spec]=raw_spec[spec].A

towrite_raw_spectra.to_csv("raw_spectra.csv", index=True)


towrite_ready_spec_nolam4tra=tmp.drop(columns=['wl','A']) #structure for the written table
for spec in ready_spec_nolam4:
    towrite_ready_spec_nolam4tra[spec]=ready_spec_nolam4[spec].A
towrite_ready_spec_nolam4tra.to_csv("constant-baseline_corrected_spec.csv", index=True)



towrite_ready_spectra=tmp.drop(columns=['wl','A']) #structure for the written table
for spec in ready_spec:
    towrite_ready_spectra[spec]=ready_spec[spec].A
towrite_ready_spectra.to_csv("scattering_corrected_spec.csv", index=True)



