#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 16:36:40 2021

@author: Caramello
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
import math as mth
#import scipy.spsolve as spsolve

os.chdir('./') 

def closest(lst, K): 
     lst = np.asarray(lst) 
     idx = (np.abs(lst - K)).argmin() 
     return(lst[idx])


def fct_baseline(x, a, b):
    return(a/np.power(x,4)+b)

def fct_relaxation_monoexp(x,a,b,tau):
    return(a-b*np.exp(-x/tau))

#def fct_relaxation_bi_exp(x,a,b1,tau1,b2,tau2):
#    return(a-b1*np.exp(-x/tau1)-b2*np.exp(-x/tau2))

def rescale_corrected(df, timestamp, wlmin, wlmax): #scales with a factor corresponding to 
    a=df.copy() #df[df.wl.between(wlmin,wlmax)].copy()
    a.wl=df.wl
    # offset=a.A[a.wl.between(wlmax-10,wlmax-1)].min()
    # a.A=a.A.copy()-offset        
#    scale=df[df.wl.between(wlmin,wlmax)].mean()
    fact=1/a[timestamp][a.wl.between(wlmin,wlmax,inclusive="both")].max() #1/a.A[a.wl.between(425,440)].max()
    a[timestamp]=a[timestamp].copy()*fact      
    return(a.copy())

def rescale_raw(df,rawdata, wlmin, wlmax):
    a=df[df.wl.between(wlmin,wlmax)].copy()
    raw=rawdata[rawdata.wl.between(wlmin,wlmax)].copy()
    # offset=a.A[a.wl.between(wlmax-10,wlmax-1)].min()
    # a.A=a.A.copy()-offset        
    fact=1/a.A[a.wl.between(wlmin,wlmax,inclusive="both")].max() #1/a.A[a.wl.between(425,440)].max()
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
    n=len(df.A[segment1])
    sigma=n*[sigmaby3segment[0]]
    n=len(df.A[segment2])
    sigma=sigma + n*[sigmaby3segment[1]]
    n=len(df.A[segmentend])
    sigma=sigma + n*[sigmaby3segment[2]]
    #print(sigma)
    para, pcov = sp.optimize.curve_fit(f=fct_baseline, xdata=x, ydata=y, p0=initialParameters, sigma=sigma)
    baseline=df.copy()
    baseline.A=fct_baseline(baseline.wl.copy(), *para)
    #baseline.A=baseline_als(np.array(y), 10^5, 0.01)
    #plt.plot(df.wl,df.A)
    #plt.plot(baseline.wl, baseline.A)
    #plt.show()
    corrected=df.copy()
    corrected.A=df.A.copy()-baseline.A
    return(corrected)


def baselinefit_cst_smooth(df,timestamp):
    base=df[timestamp][df.wl.between(600,700)].mean() #sum(df.A[df.A.between(600,700)])/len(df.A[df.A.between(600,700)]) #df.A[df.A.between(600,700)].mean()
    print(base)
    corrected=df.copy()
    corrected[timestamp]=df[timestamp].copy()-base
    return(corrected)

def absorbance(lightref, ourdata):
    ourabs=ourdata.copy()
    for timestamp in ourdata:
        tmp=ourdata[timestamp].copy()
        for wavelength in tmp.index:
            tmp[wavelength]=-np.log(tmp[wavelength]/lightref.counts[wavelength])
        ourabs[timestamp]=tmp
    print(ourabs)
    return(ourabs)
            
limit_plot=1000
limbas=8000

####timescale!!!!!!!!!!!!!!!!!!!!! eeeennnnnn secooooooondes
timescale=0.05

#defining size of in-gui figures, /2.54 because inches are for heretics
plt.rcParams["figure.figsize"] = (40/2.54,30/2.54)

####Parsing through the dir to find spectra and importing/treating them#### 
directory = './'  #Because of the genius who decided to put nearly all of our files in "Work Folder" and not "Work_Folder" we actually have to double backslash every path



listspec=[]
numspec=[]
for entry in os.scandir(directory):   #The equivalent of "For f in ./*" in bash
    if entry.path.endswith(".txt") and entry.is_file() :   #We're only interested in spectra, hopefully they are .csv, if not, you can still change the extension
        listspec.append(entry.path)
#        numspec.append(float(entry.path[-8:-4]))
print(listspec)


# spec=pd.DataFrame(data=listspec,index=numspec,columns=["paths"])

print(listspec[0])
#lam_390=pd.DataFrame(columns=["t","l_390"], index=listspec)

#lam_300=pd.DataFrame(columns=["t","l_300"], index=listspec)
# count=0
# numspec.sort()

#first laser activation 
# ini_laser=0

reference=pd.read_csv(filepath_or_buffer= 'light.csv',   #pandas has a neat function to read char-separated tables
                    sep= "[\t]",             #Columns delimited by semicolon
                    decimal=".",              #decimals delimited by colon
                    skip_blank_lines=True,        #There is a blank line at the end
                    skipfooter=1,             #2 lines of footer, not counting the blank line
                    header=0,
                    engine="python")          #The python engine is slower but supports header and footers
reference.index=reference.wl


raw_lamp=pd.read_csv(filepath_or_buffer= listspec[0],   #pandas has a neat function to read char-separated tables
                    sep= "[\t]",             #Columns delimited by semicolon
                    decimal=".",              #decimals delimited by colon
                    skip_blank_lines=True,        #There is a blank line at the end
                    skipfooter=1,             #2 lines of footer, not counting the blank line
                    header=0,
                    index_col=False,
                    engine="python")          #The python engine is slower but supports header and footers
raw_lamp.index=raw_lamp.wl

tmp=raw_lamp.drop(columns='wl')
raw_spec=absorbance(reference, tmp)
raw_spec=raw_spec.fillna(method='ffill')
raw_spec.wl=raw_lamp.wl

lam_492=pd.DataFrame(columns=["t","l_492"], index=raw_spec.columns)
#raw_spec.wl=raw_spec.index
#raw_spec.origin=len(raw_spec.wl)*[0]
ready_spec=raw_spec.copy()              #Last one was for the brute files, this one is for the scaled ones
ready_spec.wl=raw_lamp.wl
raw_ready_spec=raw_spec.copy()
raw_ready_spec.wl=raw_lamp.wl
smoothed_spec=raw_spec.copy()
smoothed_spec.wl=raw_lamp.wl
corr_spec=raw_spec.copy()
corr_spec.wl=raw_lamp.wl

for timestamp in raw_spec:
    print(timestamp)
    smoothed_spec[timestamp]=sp.signal.savgol_filter(x=raw_spec[timestamp].copy(),
                 window_length=21,
                 polyorder=3)       #The order of the polynom, more degree = less smooth, more precise (and more ressource expensive)
    maxwl=250 #raw_spec.index.min()
#    leftborn=smoothed_spec[timestamp][raw_spec.wl.between(220,255)].idxmax()
#    rightborn=smoothed_spec[timestamp][raw_spec.index.between(leftborn,270)].idxmin()
#    segment1=raw_spec.index.between(leftborn,rightborn, inclusive='both')
#    center_creux=smoothed_spec[timestamp][raw_spec.index.between(290,360)].idxmin()
#    segment2=raw_spec.index.between(center_creux-3,center_creux+3, inclusive='both')    
#    segmentend=raw_spec.index.between(550,900, inclusive='both')
#    sigmafor3segment=[1,1,1]
    tmp2=baselinefit_cst_smooth(smoothed_spec,timestamp)[timestamp] # smoothed_spec[timestamp][raw_spec.index.between(280,432)].idxmin()
    corr_spec[timestamp]=tmp2.copy()
    tmp3=rescale_corrected(corr_spec,timestamp, 200,300)[timestamp]
    ready_spec[timestamp]=tmp3
    #lam_390.t[timestamp]=num*timescale
    #lam_390.l_390[timestamp]=corr_spec[timestamp].timestamp[closest(corr_spec[timestamp].raw_spec.index, 390)]
    lam_492.t[timestamp]=float(timestamp)
    lam_492.l_492[timestamp]=corr_spec[timestamp][closest(corr_spec.wl, 492)]
    #lam_300.t[timestamp]=num*timescale
    #lam_300.l_300[timestamp]=corr_spec[timestamp].timestamp[closest(corr_spec[timestamp].raw_spec.index, 300)]


lam_492=lam_492.fillna(method='ffill')
lam_492.index=lam_492.t
    
#lam_492=lam_492.sort_values("t")  

#initialParameters = np.array([0.4, 0.4, 1.0])
#sigma=len(lam_492.t[lam_492.t.between(limit_plot,limbas, inclusive="both")])*[1]
#x=np.array(lam_492.t[lam_492.t.between(limit_plot,limbas, inclusive="both")])
#y=np.array(lam_492.l_492[lam_492.t.between(limit_plot,limbas, inclusive="both")])
#para, pcov = sp.optimize.curve_fit(f=fct_relaxation_monoexp, xdata=x, ydata=y, p0=initialParameters, sigma=sigma)
#
#pd.DataFrame(data=[para,pcov], 
#             index=["parameters","covariance"], 
#             columns = ["a","b1","tau1","b2","tau2"]).to_csv(path_or_buf="492_model_para.tsv", sep = "\t")
#
#for num in range(0,len(listspec),1):   #The equivalent of "For f in ./*" in bash
#    a=spec.paths[num]
#    model_a=fct_relaxation_monoexp(lam_492.t[a],para[0],para[1],para[2],para[3],para[4])
#    print(a, model_a)
#    lam_492.loc[a,'model']=model_a

#plt.plot(lam_492.t,lam_492.l_492)
#plt.plot(lam_492.t,lam_492.model)
#plt.show()



####Plotting the data####

fig, ax = plt.subplots()     #First let's create our figure, subplots ensures we can plot several curves on the same graph
ax.set_xlabel('Time since blue light extinction [s]', fontsize=10)  #x axis 
ax.xaxis.set_label_coords(x=0.5, y=-0.08)      #This determines where the x-axis is on the figure 
ax.set_ylabel('Absorbance at 492nm [AU]', fontsize=10)               #Label of the y axis
ax.yaxis.set_label_coords(x=-0.1, y=0.5)       #position of the y axis 
palette=sns.color_palette(palette="bright", n_colors=2)   #This creates a palette with distinct colors in function of the number of sample, check it at https://seaborn.pydata.org/tutorial/color_palettes.html, in our case we might want to cherry-pick our colors, that's easy: palette are only lists of rgb triplets. Seaborn has a "desat" var, it modulates intensity of the color we can probably use that for emission/excitation plots 

n=0                          #this is just a counter for the palette, it's ugly as hell but hey, it works 
                  #We can then parse over our dictionary to plot our data
ax.plot(lam_492.t[lam_492.t.between(limit_plot,limbas, inclusive='both')], 
        lam_492.l_492[lam_492.t.between(limit_plot,limbas, inclusive='both')] ,
        'bo')

#ax.plot(lam_492.t,
#        lam_492.model,             #y-axis is abs, or emission, or else
#        linewidth=1,              #0.5 : pretty thin, 2 : probably what Hadrien used 
#        label="modelled relaxation curve with tau1="+format(para[2], '.2f')+" tau2="+format(para[4], '.2f')+" a="+format(para[0], '.2f')+" b1="+format(para[1], '.2f')+" b2="+format(para[3], '.2f'), #Label is currently the name of our file, we could replace that by a list of names
#        color=palette[1])
#

ax.set_title('absorbance at 492nm over time during and after blue light extinction', fontsize=10, fontweight='bold')  #This sets the title of the plot
ax.set_xlim([limit_plot,limbas]) 
ax.set_ylim([lam_492.l_492[lam_492.t.between(limit_plot,limbas, inclusive='both')].min(), lam_492.l_492[lam_492.t.between(limit_plot,limbas,inclusive="both")].max()+0.05])
ax.tick_params(labelsize=10)
ax.yaxis.set_ticks(np.arange(lam_492.l_492[lam_492.t.between(limit_plot,limbas, inclusive='both')].min(), lam_492.l_492[lam_492.t.between(limit_plot,limbas,inclusive="both")].max()+0.05, 0.05))  #This modulates the frequency of the x label (1, 50 ,40 ect)

legend = plt.legend(loc='lower right', shadow=True, prop={'size':7})

#plt.show()
# ################################################

figfilename = "l492.pdf"             #Filename, for the PDF output, don't forget to close the last one before executing again, otherwise python can't write over it
plt.savefig(figfilename, dpi=3000, transparent=True,bbox_inches='tight')
plt.close()






#I'll let you figure out how to set visible axis and make them black instead of grey to match Hadrien's pattern

tmp=corr_spec.copy()


for timestamp in raw_spec:
    if float(timestamp)<limit_plot or float(timestamp)>limbas:
        tmp=tmp.drop(columns=timestamp)
len(tmp.columns)

#processed_plot
listmax=[]
listmin=[]
tmp.wl=raw_spec.wl
for i in tmp :
    a=tmp[i][tmp.wl.between(200,300)].max()
    if not (mth.isinf(a) | mth.isnan(a)):
        listmax.append(a)
    a=tmp[i][tmp.wl.between(750,800)].min()
    if not (mth.isinf(a) | mth.isnan(a)):
        listmin.append(a)
globmax=max(listmax)
globmin=min(listmin)




fig, ax = plt.subplots()     #First let's create our figure, subplots ensures we can plot several curves on the same graph
ax.set_xlabel('Wavelength [nm]', fontsize=10)  #x axis 
ax.xaxis.set_label_coords(x=0.5, y=-0.08)      #This determines where the x-axis is on the figure 
ax.set_ylabel('Absorbance [AU]', fontsize=10)               #Label of the y axis
ax.yaxis.set_label_coords(x=-0.1, y=0.5)       #position of the y axis 
palette=sns.color_palette(palette='Spectral', n_colors=len(tmp.columns))   #This creates a palette with distinct colors in function of the number of sample, check it at https://seaborn.pydata.org/tutorial/color_palettes.html, in our case we might want to cherry-pick our colors, that's easy: palette are only lists of rgb triplets. Seaborn has a "desat" var, it modulates intensity of the color we can probably use that for emission/excitation plots 

n=0                                            #this is just a counter for the palette, it's ugly as hell but hey, it works 
for i in tmp :                          #We can then parse over our dictionary to plot our data
    ax.plot(tmp.index,                  #x-axis is wavelength
            tmp[i] ,                   #y-axis is abs, or emission, or else
            linewidth=1,                    #0.5 : pretty thin, 2 : probably what Hadrien used 
            label=i,                        #Label is currently the name of our file, we could replace that by a list of names
            color=palette[n])               #This determines the color of the curves, you can create a custom list of colors such a c['blue','red'] ect
    n=n+1
ax.set_title('corrected absorbance in crystallo absorbance spectra', fontsize=10, fontweight='bold')  #This sets the title of the plot
ax.set_xlim([200,900]) 
ax.set_ylim([globmin,globmax])
ax.tick_params(labelsize=10)
ax.yaxis.set_ticks(np.arange(globmin, globmax, 0.1))  #This modulates the frequency of the x label (1, 50 ,100 ect)

legend = plt.legend(loc='upper right', shadow=True, prop={'size':7})


# plt.show()

# ################################################

figfilename = "constant_baseline_corrected_spectra.pdf"             #Filename, for the PDF output, don't forget to close the last one before executing again, otherwise python can't write over it
plt.savefig(figfilename, dpi=3000, transparent=True,bbox_inches='tight') 
plt.close()

 
#### smoothedplots

#processed_plot
listmax=[]
listmin=[]

tmp=smoothed_spec.copy()
for timestamp in raw_spec:
    if float(timestamp)<limit_plot or float(timestamp)>limbas:
        tmp=tmp.drop(columns=timestamp)
len(tmp.columns)

#processed_plot
listmax=[]
listmin=[]
tmp.wl=raw_spec.wl
for i in tmp :
    a=tmp[i][tmp.wl.between(200,300)].max()
    if not (mth.isinf(a) | mth.isnan(a)):
        listmax.append(a)
    a=tmp[i][tmp.wl.between(750,800)].min()
    if not (mth.isinf(a) | mth.isnan(a)):
        listmin.append(a)
globmax=max(listmax)
globmin=min(listmin)



fig, ax = plt.subplots()     #First let's create our figure, subplots ensures we can plot several curves on the same graph
ax.set_xlabel('Wavelength [nm]', fontsize=10)  #x axis 
ax.xaxis.set_label_coords(x=0.5, y=-0.08)      #This determines where the x-axis is on the figure 
ax.set_ylabel('Absorbance [AU]', fontsize=10)               #Label of the y axis
ax.yaxis.set_label_coords(x=-0.1, y=0.5)       #position of the y axis 
palette=sns.color_palette(palette='Spectral', n_colors=len(tmp.columns))   #This creates a palette with distinct colors in function of the number of sample, check it at https://seaborn.pydata.org/tutorial/color_palettes.html, in our case we might want to cherry-pick our colors, that's easy: palette are only lists of rgb triplets. Seaborn has a "desat" var, it modulates intensity of the color we can probably use that for emission/excitation plots 

n=0                                            #this is just a counter for the palette, it's ugly as hell but hey, it works 
for i in tmp :                          #We can then parse over our dictionary to plot our data
    ax.plot(tmp.index,                  #x-axis is wavelength
            tmp[i] ,                   #y-axis is abs, or emission, or else
            linewidth=1,                    #0.5 : pretty thin, 2 : probably what Hadrien used 
            label=i,                        #Label is currently the name of our file, we could replace that by a list of names
            color=palette[n])               #This determines the color of the curves, you can create a custom list of colors such a c['blue','red'] ect
    n=n+1
ax.set_title('smoothed in crystallo absorbance spectra', fontsize=10, fontweight='bold')  #This sets the title of the plot
ax.set_xlim([200,900]) 
ax.set_ylim([globmin,globmax])
ax.tick_params(labelsize=10)
ax.yaxis.set_ticks(np.arange(globmin, globmax, 0.1))  #This modulates the frequency of the x label (1, 50 ,100 ect)

legend = plt.legend(loc='upper right', shadow=True, prop={'size':7})
################################################

#plt.show()

figfilename = "spectra_smoothed.pdf"             #Filename, for the PDF output, don't forget to close the last one before executing again, otherwise python can't write over it
plt.savefig(figfilename, dpi=3000, transparent=True,bbox_inches='tight')
plt.close()


 
#### rawplots
listmax=[]
listmin=[]


tmp=raw_spec.copy()
for timestamp in raw_spec:
    if float(timestamp)<limit_plot or float(timestamp)>limbas:
        tmp=tmp.drop(columns=timestamp)
len(tmp.columns)

#processed_plot
listmax=[]
listmin=[]
tmp.wl=raw_spec.wl
for i in tmp :
    a=tmp[i][tmp.wl.between(200,300)].max()
    if not (mth.isinf(a) | mth.isnan(a)):
        listmax.append(a)
    a=tmp[i][tmp.wl.between(750,800)].min()
    if not (mth.isinf(a) | mth.isnan(a)):
        listmin.append(a)
globmax=max(listmax)
globmin=min(listmin)




fig, ax = plt.subplots()     #First let's create our figure, subplots ensures we can plot several curves on the same graph
ax.set_xlabel('Wavelength [nm]', fontsize=10)  #x axis 
ax.xaxis.set_label_coords(x=0.5, y=-0.08)      #This determines where the x-axis is on the figure 
ax.set_ylabel('Absorbance [AU]', fontsize=10)               #Label of the y axis
ax.yaxis.set_label_coords(x=-0.1, y=0.5)       #position of the y axis 
palette=sns.color_palette(palette='Spectral', n_colors=len(tmp.columns))   #This creates a palette with distinct colors in function of the number of sample, check it at https://seaborn.pydata.org/tutorial/color_palettes.html, in our case we might want to cherry-pick our colors, that's easy: palette are only lists of rgb triplets. Seaborn has a "desat" var, it modulates intensity of the color we can probably use that for emission/excitation plots 

n=0                                            #this is just a counter for the palette, it's ugly as hell but hey, it works 
for i in tmp :                          #We can then parse over our dictionary to plot our data
    ax.plot(tmp.index,                  #x-axis is wavelength
            tmp[i] ,                   #y-axis is abs, or emission, or else
            linewidth=1,                    #0.5 : pretty thin, 2 : probably what Hadrien used 
            label=i,                        #Label is currently the name of our file, we could replace that by a list of names
            color=palette[n])               #This determines the color of the curves, you can create a custom list of colors such a c['blue','red'] ect
    n=n+1
ax.set_title('Raw in crystallo absorbance spectra', fontsize=10, fontweight='bold')  #This sets the title of the plot
ax.set_xlim([200,900]) 
ax.set_ylim([globmin,globmax])
ax.tick_params(labelsize=10)
ax.yaxis.set_ticks(np.arange(globmin, globmax, 0.1))  #This modulates the frequency of the x label (1, 50 ,100 ect)

legend = plt.legend(loc='upper right', shadow=True, prop={'size':7})
################################################

#plt.show()

#temporary=plt.plot(raw_spec)

figfilename = "spectra_raw.pdf"             #Filename, for the PDF output, don't forget to close the last one before executing again, otherwise python can't write over it
plt.savefig(figfilename, dpi=3000, transparent=True,bbox_inches='tight')
plt.close()



raw_spec.to_csv("raw_spectra.csv", index=True)
smoothed_spec.to_csv("smoothed_spectra.csv", index=True)
corr_spec.to_csv("constant_baseline_corrected_spectra.csv", index=True)


lam_492.drop(columns='t').to_csv("Abs_392nm.csv", index=True)