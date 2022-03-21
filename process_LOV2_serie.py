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
#import scipy.spsolve as spsolve

os.chdir('./') 

def closest(lst, K): 
     lst = np.asarray(lst) 
     idx = (np.abs(lst - K)).argmin() 
     return(lst[idx])


def fct_baseline(x, a, b):
    return(a/np.power(x,4)+b)

def fct_relaxation390(x,a,b,tau):
    return(a-b*np.exp(-x/tau))

def fct_relaxation475(x,a,b1,tau1,b2,tau2):
    return(a-b1*np.exp(-x/tau1)-b2*np.exp(-x/tau2))

def rescale_corrected(df, wlmin, wlmax): #scales with a factor corresponding to 
    a=df.copy() #df[df.wl.between(wlmin,wlmax)].copy()
    # offset=a.A[a.wl.between(wlmax-10,wlmax-1)].min()
    # a.A=a.A.copy()-offset        
    scale=df.A[df.wl.between(wlmin,wlmax)].mean()
    fact=1/a.A[a.wl.between(wlmin,wlmax,inclusive="both")].max() #1/a.A[a.wl.between(425,440)].max()
    a.A=a.A.copy()*fact      
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


def baselinefit_cst_smooth(df,  segment1, segment2, segmentend, sigmaby3segment):
    base=df.A[df.wl.between(600,700)].mean() #sum(df.A[df.A.between(600,700)])/len(df.A[df.A.between(600,700)]) #df.A[df.A.between(600,700)].mean()
    print(base)
    corrected=df.copy()
    corrected.A=df.A.copy()-base
    return(corrected)


####timescale!!!!!!!!!!!!!!!!!!!!! eeeennnnnn secooooooondes
timescale=0.5

#defining size of in-gui figures, /2.54 because inches are for heretics
plt.rcParams["figure.figsize"] = (20/2.54,15/2.54)

####Parsing through the dir to find spectra and importing/treating them#### 
directory = './'  #Because of the genius who decided to put nearly all of our files in "Work Folder" and not "Work_Folder" we actually have to double backslash every path
raw_spec={}                #This is a Dictionary, it's one of python's list-like objects, it's main properties are : not ordered, can be changed ==> We're using it to store our spectra and index them by the name of the file
ready_spec={}               #Last one was for the brute files, this one is for the scaled ones
raw_ready_spec={}
smoothed_spec={}
corr_spec={}


listspec=[]
numspec=[]
for entry in os.scandir(directory):   #The equivalent of "For f in ./*" in bash
    if entry.path.endswith(".txt") and entry.is_file() :   #We're only interested in spectra, hopefully they are .csv, if not, you can still change the extension
        listspec.append(entry.path)
        numspec.append(float(entry.path[-7:-4]))
print(listspec)
print(numspec)


spec=pd.DataFrame(data=listspec,index=numspec,columns=["paths"])

#print(listspec)
lam_390=pd.DataFrame(columns=["t","l_390"], index=listspec)
lam_475=pd.DataFrame(columns=["t","l_475"], index=listspec)
lam_300=pd.DataFrame(columns=["t","l_300"], index=listspec)
count=0
numspec.sort()
for num in numspec: # range(0,100,1): #range(0,len(listspec),1):  #The equivalent of "For f in ./*" in bash
    a=spec.paths[num]
    print(a)
    tmp=pd.read_csv(filepath_or_buffer= a,   #pandas has a neat function to read char-separated tables
                    sep= "[\t]",             #Columns delimited by semicolon
                    decimal=".",              #decimals delimited by colon
                    skiprows=17,              #There is 18 rows of header
                    skip_blank_lines=True,        #There is a blank line at the end
                    skipfooter=1,             #2 lines of footer, not counting the blank line
                    names=['wl','A'],         #The whole scripts relies on column names being 'wl' for Wavelength and 'A' for Absorbance
                    engine="python")          #The python engine is slower but supports header and footers
    tmp.index=tmp.wl                  #We want rows to be indexed on wl
    tmp.origin=len(tmp.wl)*[0]
    raw_spec[a]=tmp.copy()                  #We're storing the table in the dictionnary
    tmp.A=sp.signal.savgol_filter(x=tmp.A.copy(),     #This is the smoothing function, it takes in imput the y-axis data directly and fits a polynom on each section of the data at a time
                                  window_length=21,  #This defines the section, longer sections means smoother data but also bigger imprecision
                                  polyorder=3)       #The order of the polynom, more degree = less smooth, more precise (and more ressource expensive)
    smoothed_spec[a]=tmp.copy()
    maxwl=250 #tmp.wl.min()
    leftborn=tmp.A[tmp.wl.between(220,255)].idxmax()
    rightborn=tmp.A[tmp.wl.between(leftborn,270)].idxmin()
    segment1=tmp.wl.between(leftborn,rightborn, inclusive='both')
    center_creux=tmp.A[tmp.wl.between(290,360)].idxmin()
    segment2=tmp.wl.between(center_creux-3,center_creux+3, inclusive='both')    
    segmentend=tmp.wl.between(550,900, inclusive='both')
    sigmafor3segment=[1,1,1]
    tmp2=baselinefit_cst_smooth(tmp,  segment1, segment2, segmentend, sigmafor3segment) # tmp.A[tmp.wl.between(280,432)].idxmin()
    corr_spec[a]=tmp2.copy()
    tmp3=rescale_corrected(tmp2, 200,300)
    ready_spec[a]=tmp3
    lam_390.t[a]=num*timescale
    lam_390.l_390[a]=corr_spec[a].A[closest(corr_spec[a].wl, 390)]
    lam_475.t[a]=num*timescale
    lam_475.l_475[a]=corr_spec[a].A[closest(corr_spec[a].wl, 475)]
    lam_300.t[a]=num*timescale
    lam_300.l_300[a]=corr_spec[a].A[closest(corr_spec[a].wl, 300)]






    
#lam_300=lam_300.sort_values("t")  
#
#initialParameters = np.array([0.4, 0.4, 20])
#sigma=len(lam_300.t[lam_300.t.between(0,100, inclusive="both")])*[1]
#x=np.array(lam_300.t[lam_300.t.between(0,100, inclusive="both")])
#y=np.array(lam_300.l_300[lam_300.t.between(0,100, inclusive="both")])
#para, pcov = sp.optimize.curve_fit(f=fct_relaxation390, xdata=x, ydata=y, p0=initialParameters, sigma=sigma)
#
#pd.DataFrame(data=[para,pcov], 
#             index=["parameters","covariance"], 
#             columns = ["a","b","tau"]).to_csv(path_or_buf="300_model_para.tsv", sep = "\t")
#
#for num in range(0,len(listspec),1):   #The equivalent of "For f in ./*" in bash
#    a=spec.paths[num]
#    model_a=fct_relaxation390(lam_300.t[a],para[0],para[1],para[2])
#    print(a, model_a)
#    lam_300.loc[a,'model']=model_a
#
##plt.plot(lam_300.t,lam_300.l_300)
##plt.plot(lam_300.t,lam_300.model)
##plt.show()
#
#
#
#
#
#####Plotting the data####
#
#fig, ax = plt.subplots()     #First let's create our figure, subplots ensures we can plot several curves on the same graph
#ax.set_xlabel('Time since blue light extinction [s]', fontsize=10)  #x axis 
#ax.xaxis.set_label_coords(x=0.5, y=-0.08)      #This determines where the x-axis is on the figure 
#ax.set_ylabel('Absorbance at 300nm [AU]', fontsize=10)               #Label of the y axis
#ax.yaxis.set_label_coords(x=-0.1, y=0.5)       #position of the y axis 
#palette=sns.color_palette(palette="bright", n_colors=2)   #This creates a palette with distinct colors in function of the number of sample, check it at https://seaborn.pydata.org/tutorial/color_palettes.html, in our case we might want to cherry-pick our colors, that's easy: palette are only lists of rgb triplets. Seaborn has a "desat" var, it modulates intensity of the color we can probably use that for emission/excitation plots 
#
#n=0                          #this is just a counter for the palette, it's ugly as hell but hey, it works 
#                  #We can then parse over our dictionary to plot our data
#ax.plot(lam_300.t, 
#        lam_300.l_300 ,
#        'bo')
#
#ax.plot(lam_300.t,
#        lam_300.model,             #y-axis is abs, or emission, or else
#        linewidth=1,              #0.5 : pretty thin, 2 : probably what Hadrien used 
#        label="modelled relaxation curve with tau="+format(para[2], '.2f')+"a="+format(para[0], '.2f')+"b="+format(para[1], '.2f'),                  #Label is currently the name of our file, we could replace that by a list of names
#        color=palette[1])
#
#
#ax.set_title('absorbance at 300nm over time after blue light extinction', fontsize=10, fontweight='bold')  #This sets the title of the plot
#ax.set_xlim([0, closest(lam_300.t,100)]) 
#ax.set_ylim([lam_300.l_300.min(), lam_300.l_300[lam_300.t.between(0,100,inclusive="both")].max()+0.05])
#ax.tick_params(labelsize=10)
#ax.yaxis.set_ticks(np.arange(lam_300.l_300.min(), lam_300.l_300[lam_300.t.between(0,100,inclusive="both")].max()+0.05, 0.05))  #This modulates the frequency of the x label (1, 50 ,40 ect)
#
#legend = plt.legend(loc='lower right', shadow=True, prop={'size':7})
#
##plt.show()
## ################################################
#
#figfilename = "l300.pdf"             #Filename, for the PDF output, don't forget to close the last one before executing again, otherwise python can't write over it
#plt.savefig(figfilename, dpi=3000, transparent=True)
#plt.close()
#
#
#
#
#    
#lam_390=lam_390.sort_values("t")  
#
#initialParameters = np.array([0.4, 0.4, 20])
#sigma=len(lam_390.t[lam_390.t.between(0,100, inclusive="both")])*[1]
#x=np.array(lam_390.t[lam_390.t.between(0,100, inclusive="both")])
#y=np.array(lam_390.l_390[lam_390.t.between(0,100, inclusive="both")])
#para, pcov = sp.optimize.curve_fit(f=fct_relaxation390, xdata=x, ydata=y, p0=initialParameters, sigma=sigma)
#
#pd.DataFrame(data=[para,pcov], 
#             index=["parameters","covariance"], 
#             columns = ["a","b","tau"]).to_csv(path_or_buf="390_model_para.tsv", sep = "\t")
#
#for num in range(0,len(listspec),1):   #The equivalent of "For f in ./*" in bash
#    a=spec.paths[num]
#    model_a=fct_relaxation390(lam_390.t[a],para[0],para[1],para[2])
#    print(a, model_a)
#    lam_390.loc[a,'model']=model_a
#
##plt.plot(lam_390.t,lam_390.l_390)
##plt.plot(lam_390.t,lam_390.model)
##plt.show()
#
#
#
#
#
#####Plotting the data####
#
#fig, ax = plt.subplots()     #First let's create our figure, subplots ensures we can plot several curves on the same graph
#ax.set_xlabel('Time since blue light extinction [s]', fontsize=10)  #x axis 
#ax.xaxis.set_label_coords(x=0.5, y=-0.08)      #This determines where the x-axis is on the figure 
#ax.set_ylabel('Absorbance at 390nm [AU]', fontsize=10)               #Label of the y axis
#ax.yaxis.set_label_coords(x=-0.1, y=0.5)       #position of the y axis 
#palette=sns.color_palette(palette="bright", n_colors=2)   #This creates a palette with distinct colors in function of the number of sample, check it at https://seaborn.pydata.org/tutorial/color_palettes.html, in our case we might want to cherry-pick our colors, that's easy: palette are only lists of rgb triplets. Seaborn has a "desat" var, it modulates intensity of the color we can probably use that for emission/excitation plots 
#
#n=0                          #this is just a counter for the palette, it's ugly as hell but hey, it works 
#                  #We can then parse over our dictionary to plot our data
#ax.plot(lam_390.t, 
#        lam_390.l_390 ,
#        'bo')
#
#ax.plot(lam_390.t,
#        lam_390.model,             #y-axis is abs, or emission, or else
#        linewidth=1,              #0.5 : pretty thin, 2 : probably what Hadrien used 
#        label="modelled relaxation curve with tau="+format(para[2], '.2f')+"a="+format(para[0], '.2f')+"b="+format(para[1], '.2f'),                  #Label is currently the name of our file, we could replace that by a list of names
#        color=palette[1])
#
#
#ax.set_title('absorbance at 390nm over time after blue light extinction', fontsize=10, fontweight='bold')  #This sets the title of the plot
#ax.set_xlim([0, closest(lam_390.t,100)]) 
#ax.set_ylim([lam_390.l_390.min(), lam_390.l_390[lam_390.t.between(0,100,inclusive="both")].max()+0.05])
#ax.tick_params(labelsize=10)
#ax.yaxis.set_ticks(np.arange(lam_390.l_390.min(), lam_390.l_390[lam_390.t.between(0,100,inclusive="both")].max()+0.05, 0.05))  #This modulates the frequency of the x label (1, 50 ,40 ect)
#
#legend = plt.legend(loc='lower right', shadow=True, prop={'size':7})
#
##plt.show()
## ################################################
#
#figfilename = "l390.pdf"             #Filename, for the PDF output, don't forget to close the last one before executing again, otherwise python can't write over it
#plt.savefig(figfilename, dpi=3000, transparent=True)
#plt.close()
#
#
#    
#lam_475=lam_475.sort_values("t")  
#
#initialParameters = np.array([0.4, 0.4, 20,0.4,20])
#sigma=len(lam_475.t[lam_475.t.between(0,100, inclusive="both")])*[1]
#x=np.array(lam_475.t[lam_475.t.between(0,100, inclusive="both")])
#y=np.array(lam_475.l_475[lam_475.t.between(0,100, inclusive="both")])
#para, pcov = sp.optimize.curve_fit(f=fct_relaxation475, xdata=x, ydata=y, p0=initialParameters, sigma=sigma)
#
#pd.DataFrame(data=[para,pcov], 
#             index=["parameters","covariance"], 
#             columns = ["a","b1","tau1","b2","tau2"]).to_csv(path_or_buf="475_model_para.tsv", sep = "\t")
#
#for num in range(0,len(listspec),1):   #The equivalent of "For f in ./*" in bash
#    a=spec.paths[num]
#    model_a=fct_relaxation475(lam_475.t[a],para[0],para[1],para[2],para[3],para[4])
#    print(a, model_a)
#    lam_475.loc[a,'model']=model_a
#
##plt.plot(lam_475.t,lam_475.l_475)
##plt.plot(lam_475.t,lam_475.model)
##plt.show()
#
#
#
#
#
#####Plotting the data####
#
#fig, ax = plt.subplots()     #First let's create our figure, subplots ensures we can plot several curves on the same graph
#ax.set_xlabel('Time since blue light extinction [s]', fontsize=10)  #x axis 
#ax.xaxis.set_label_coords(x=0.5, y=-0.08)      #This determines where the x-axis is on the figure 
#ax.set_ylabel('Absorbance at 475nm [AU]', fontsize=10)               #Label of the y axis
#ax.yaxis.set_label_coords(x=-0.1, y=0.5)       #position of the y axis 
#palette=sns.color_palette(palette="bright", n_colors=2)   #This creates a palette with distinct colors in function of the number of sample, check it at https://seaborn.pydata.org/tutorial/color_palettes.html, in our case we might want to cherry-pick our colors, that's easy: palette are only lists of rgb triplets. Seaborn has a "desat" var, it modulates intensity of the color we can probably use that for emission/excitation plots 
#
#n=0                          #this is just a counter for the palette, it's ugly as hell but hey, it works 
#                  #We can then parse over our dictionary to plot our data
#ax.plot(lam_475.t, 
#        lam_475.l_475 ,
#        'bo')
#
#ax.plot(lam_475.t,
#        lam_475.model,             #y-axis is abs, or emission, or else
#        linewidth=1,              #0.5 : pretty thin, 2 : probably what Hadrien used 
#        label="modelled relaxation curve with tau1="+format(para[2], '.2f')+" tau2="+format(para[4], '.2f')+" a="+format(para[0], '.2f')+" b1="+format(para[1], '.2f')+" b2="+format(para[3], '.2f'), #Label is currently the name of our file, we could replace that by a list of names
#        color=palette[1])
#
#
#ax.set_title('absorbance at 475nm over time after blue light extinction', fontsize=10, fontweight='bold')  #This sets the title of the plot
#ax.set_xlim([0, closest(lam_475.t,100)]) 
#ax.set_ylim([lam_475.l_475.min(), lam_475.l_475[lam_475.t.between(0,100,inclusive="both")].max()+0.05])
#ax.tick_params(labelsize=10)
#ax.yaxis.set_ticks(np.arange(lam_475.l_475.min(), lam_475.l_475[lam_475.t.between(0,100,inclusive="both")].max()+0.05, 0.05))  #This modulates the frequency of the x label (1, 50 ,40 ect)
#
#legend = plt.legend(loc='lower right', shadow=True, prop={'size':7})
#
##plt.show()
## ################################################
#
#figfilename = "l475.pdf"             #Filename, for the PDF output, don't forget to close the last one before executing again, otherwise python can't write over it
#plt.savefig(figfilename, dpi=3000, transparent=True)
#plt.close()
#
#
#



#I'll let you figure out how to set visible axis and make them black instead of grey to match Hadrien's pattern

#processed_plot
listmax=[]
listmin=[]

for i in corr_spec :
    listmax.append(corr_spec[i].A[corr_spec[i].wl.between(250,251)].max())
    listmin.append(corr_spec[i].A[corr_spec[i].wl.between(795,800)].min())
globmax=max(listmax)
globmin=min(listmin)

fig, ax = plt.subplots()     #First let's create our figure, subplots ensures we can plot several curves on the same graph
ax.set_xlabel('Wavelength [nm]', fontsize=10)  #x axis 
ax.xaxis.set_label_coords(x=0.5, y=-0.08)      #This determines where the x-axis is on the figure 
ax.set_ylabel('Absorbance [AU]', fontsize=10)               #Label of the y axis
ax.yaxis.set_label_coords(x=-0.1, y=0.5)       #position of the y axis 
palette=sns.color_palette(palette="Spectral", n_colors=len(corr_spec))   #This creates a palette with distinct colors in function of the number of sample, check it at https://seaborn.pydata.org/tutorial/color_palettes.html, in our case we might want to cherry-pick our colors, that's easy: palette are only lists of rgb triplets. Seaborn has a "desat" var, it modulates intensity of the color we can probably use that for emission/excitation plots 

n=0                                            #this is just a counter for the palette, it's ugly as hell but hey, it works 
for i in corr_spec :                          #We can then parse over our dictionary to plot our data
    ax.plot(corr_spec[i].wl,                  #x-axis is wavelength
              corr_spec[i].A ,                   #y-axis is abs, or emission, or else
              linewidth=1,                        #Label is currently the name of our file, we could replace that by a list of names
              color=palette[n])               #This determines the color of the curves, you can create a custom list of colors such a c['blue','red'] ect
    n=n+1
ax.set_title('baseline corrected in crystallo absorbance spectra', fontsize=10, fontweight='bold')  #This sets the title of the plot
ax.set_xlim([280,525]) 
ax.set_ylim([globmin, globmax])
ax.tick_params(labelsize=10)
ax.yaxis.set_ticks(np.arange(globmin, globmax, 0.1))  #This modulates the frequency of the x label (1, 50 ,100 ect)

legend = plt.legend(loc='upper right', shadow=True, prop={'size':7})    #Where the legend rectangle is 


#plt.show()

# ################################################

figfilename = "spectra_processed.pdf"             #Filename, for the PDF output, don't forget to close the last one before executing again, otherwise python can't write over it
plt.savefig(figfilename, dpi=3000, transparent=True) 
plt.close()

 
#### smoothedplots

listmax=[]
listmin=[]

for i in smoothed_spec :
    listmax.append(smoothed_spec[i].A[smoothed_spec[i].wl.between(250,251)].max())
    listmin.append(smoothed_spec[i].A[smoothed_spec[i].wl.between(795,800)].min())
globmax=max(listmax)
globmin=min(listmin)

fig, ax = plt.subplots()     #First let's create our figure, subplots ensures we can plot several curves on the same graph
ax.set_xlabel('Wavelength [nm]', fontsize=10)  #x axis 
ax.xaxis.set_label_coords(x=0.5, y=-0.08)      #This determines where the x-axis is on the figure 
ax.set_ylabel('Absorbance [AU]', fontsize=10)               #Label of the y axis
ax.yaxis.set_label_coords(x=-0.1, y=0.5)       #position of the y axis 
palette=sns.color_palette(palette='Spectral', n_colors=len(ready_spec))   #This creates a palette with distinct colors in function of the number of sample, check it at https://seaborn.pydata.org/tutorial/color_palettes.html, in our case we might want to cherry-pick our colors, that's easy: palette are only lists of rgb triplets. Seaborn has a "desat" var, it modulates intensity of the color we can probably use that for emission/excitation plots 

n=0                                            #this is just a counter for the palette, it's ugly as hell but hey, it works 
for i in smoothed_spec :                          #We can then parse over our dictionary to plot our data
    ax.plot(smoothed_spec[i].wl,                  #x-axis is wavelength
             smoothed_spec[i].A ,                   #y-axis is abs, or emission, or else
             linewidth=1,                    #0.5 : pretty thin, 2 : probably what Hadrien used 
             label=i,                        #Label is currently the name of our file, we could replace that by a list of names
             color=palette[n])               #This determines the color of the curves, you can create a custom list of colors such a c['blue','red'] ect
    n=n+1
ax.set_title('Raw in crystallo absorbance spectra', fontsize=10, fontweight='bold')  #This sets the title of the plot
ax.set_xlim([200,900]) 
ax.set_ylim([globmin, globmax])
ax.tick_params(labelsize=10)
ax.yaxis.set_ticks(np.arange(globmin, globmax, 0.1))  #This modulates the frequency of the x label (1, 50 ,100 ect)

legend = plt.legend(loc='upper right', shadow=True, prop={'size':7})
################################################

figfilename = "spectra_smoothed.pdf"             #Filename, for the PDF output, don't forget to close the last one before executing again, otherwise python can't write over it
plt.savefig(figfilename, dpi=3000, transparent=True)
plt.close()


 
#### rawplots

listmax=[]
listmin=[]

for i in raw_spec :
    listmax.append(raw_spec[i].A[raw_spec[i].wl.between(250,251)].max())
    listmin.append(raw_spec[i].A[raw_spec[i].wl.between(795,800)].min())
globmax=max(listmax)
globmin=min(listmin)

fig, ax = plt.subplots()     #First let's create our figure, subplots ensures we can plot several curves on the same graph
ax.set_xlabel('Wavelength [nm]', fontsize=10)  #x axis 
ax.xaxis.set_label_coords(x=0.5, y=-0.08)      #This determines where the x-axis is on the figure 
ax.set_ylabel('Absorbance [AU]', fontsize=10)               #Label of the y axis
ax.yaxis.set_label_coords(x=-0.1, y=0.5)       #position of the y axis 
palette=sns.color_palette(palette='Spectral', n_colors=len(ready_spec))   #This creates a palette with distinct colors in function of the number of sample, check it at https://seaborn.pydata.org/tutorial/color_palettes.html, in our case we might want to cherry-pick our colors, that's easy: palette are only lists of rgb triplets. Seaborn has a "desat" var, it modulates intensity of the color we can probably use that for emission/excitation plots 

n=0                                            #this is just a counter for the palette, it's ugly as hell but hey, it works 
for i in raw_spec :                          #We can then parse over our dictionary to plot our data
    ax.plot(raw_spec[i].wl,                  #x-axis is wavelength
             raw_spec[i].A ,                   #y-axis is abs, or emission, or else
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

figfilename = "spectra_raw.pdf"             #Filename, for the PDF output, don't forget to close the last one before executing again, otherwise python can't write over it
plt.savefig(figfilename, dpi=3000, transparent=True)
plt.close()



