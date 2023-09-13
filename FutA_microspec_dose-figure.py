# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 09:38:01 2022

@author: NCARAMEL
"""

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
import math as mth
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

def fct_baseline2(x, a, b):
    return(a/np.exp(np.power(x,4))+b)


# def fct_baseline1(x, a, b):
#     return(ax+b)
def fct_relaxation_monoexp(x,a,b,tau):
    return(a+b*np.exp(-x/tau))


def rescale_corrected(df, wlmin, wlmax):
    a=df.copy()
    # offset=a.A[a.wl.between(wlmax-10,wlmax-1)].min()
    # a.A=a.A.copy()-offset        
    fact=1/a.A[a.wl.between(wlmin,wlmax,inclusive='both')].max() #1/a.A[a.wl.between(425,440)].max()
    a.A=a.A.copy()*fact      
    return(a.copy())

def rescale_raw(df,rawdata, wlmin, wlmax):
    a=df.copy()
    raw=rawdata.copy()
    # offset=a.A[a.wl.between(wlmax-10,wlmax-1)].min()
    # a.A=a.A.copy()-offset        
    fact=1/a.A[a.wl.between(250,400,inclusive='both')].max() #1/a.A[a.wl.between(425,440)].max()
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
    x=np.array(df.wl[segment].copy())
    y=np.array(df.A[segment].copy())
    initialParameters = np.array([1, 1])
    # sigma=[1,0.01,1,0.01]
    n=len(tmp.A[segment1])
    sigma=n*[sigmaby3segment[0]]
    n=len(tmp.A[segment2])
    sigma=sigma + n*[sigmaby3segment[1]]
    n=len(tmp.A[segmentend])
    sigma=sigma + n*[sigmaby3segment[2]]
    #print(sigma)
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

def baselinefitcorr_2seg_smooth(df,  segment1, segmentend, sigmaby2segment):
    #segmentend=df.wl.between(600,800)
    segment=segment1+segmentend
    #min1 = closest(df.wl,minrange1)
    #min2 = closest(df.wl,minrange2)
    x=df.wl[segment].copy()
    y=df.A[segment].copy()
    initialParameters = np.array([1e9, 0])
    # sigma=[1,0.01,1,0.01]
    n=len(tmp.A[segment1])
    sigma=n*[sigmaby2segment[0]]
    n=len(tmp.A[segmentend])
    sigma=sigma + n*[sigmaby2segment[1]]
    #print(sigma)
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





#%%% read_proc

####Parsing through the dir to find spectra and importing/treating them#### 
directory = "./"  #Because of the genius who decided to put nearly all of our files in "Work Folder" and not "Work_Folder" we actually have to double backslash every path
raw_spec={}                #This is a Dictionary, it's one of python's list-like objects, it's main properties are : not ordered, can be changed ==> We're using it to store our spectra and index them by the name of the file
ready_spec={}               #Last one was for the brute files, this one is for the scaled ones
ready_spec_nolam4={}
corr={}

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
lam_438=pd.DataFrame(columns=["t","l_438","dose","dose_plot","p_abs","p_model"], index=listspec)
lam_550=pd.DataFrame(columns=["t","l_550","dose","dose_plot","p_abs","p_model"], index=listspec)
lam_464=pd.DataFrame(columns=["t","l_464","dose","dose_plot","p_abs","p_model"], index=listspec)
lam_620=pd.DataFrame(columns=["t","l_620","dose","dose_plot","p_abs","p_model"], index=listspec)
scat_464=pd.DataFrame(columns=["t","l_464","dose","dose_plot","p_abs","p_model"], index=listspec)


count=0
numspec.sort()
timescale=2.5 # time in between spectral acquisitions
dose_s=1.898 # dose per seconds, in kGy 
dose_step=dose_s*timescale # dose in between spectral acquisition
shutter_oppenned=20

#defining size of in-gui figures, /2.54 because inches are for heretics
plt.rcParams["figure.figsize"] = (40/2.54,30/2.54)
plt.rcParams.update({'font.size': 22})

for num in numspec: # range(0,100,1): #range(0,len(listspec),1):  #The equivalent of "For f in ./*" in bash
    a=spec.paths[num]
    print(a)
    tmp=pd.read_csv(filepath_or_buffer= a,   #pandas has a neat function to read char-separated tables
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
    # leftborn=closest(tmp.wl,290) 
    # rightborn=closest(tmp.wl,310)
    # print(leftborn,rightborn)
    segment1=tmp.wl.between(227,247, inclusive='both')
    
    segment2=tmp.wl.between(290,310, inclusive='both')
    
    leftborn=closest(tmp.wl,800)
    rightborn=closest(tmp.wl,880)      
    segmentend=tmp.wl.between(leftborn,rightborn, inclusive='both')
    # plt.plot(tmp.wl,tmp.A)
    # plt.plot(tmp.wl[segment1], tmp.A[segment1])
    # plt.plot(tmp.wl[segment2], tmp.A[segment2])
    # plt.plot(tmp.wl[segmentend], tmp.A[segmentend])
    # plt.plot(tmp.wl, tmp.origin)
    # plt.show()
    # sigmafor3segment=[1,1,1]
    sigmafor2segment=[1,1]
    tmp2, correction=baselinefitcorr_2seg_smooth(tmp,  segment1, segmentend, sigmafor2segment) # tmp.A[tmp.wl.between(280,432)].idxmin()
    tmp2_nolam4=baselineconst(tmp,segmentend)
    
#    plt.plot(tmp2.wl,tmp2.A)
#    plt.plot(tmp.wl, tmp.origin)
#    plt.show()
    
    # plt.plot(tmp2_nolam4.wl,tmp2_nolam4.A)
    # plt.plot(tmp.wl, tmp.origin)
    # plt.show()
    
    tmp3=rescale_corrected(tmp2, 290,300)
    # plt.plot(tmp3.wl,tmp3.A)
    # plt.plot(tmp.wl, tmp.origin)
    # plt.show()    
    
    tmp3_nolam4=rescale_corrected(tmp2_nolam4, 290,300)
    # plt.plot(tmp3_nolam4.wl,tmp3_nolam4.A)
    # plt.plot(tmp.wl, tmp.origin)
    # plt.show()  
    
    if(num>shutter_oppenned): 
        dosetmp=(num-shutter_oppenned)*dose_step
    else :
        dosetmp=0
    
    ready_spec[a]=tmp3
    ready_spec_nolam4[a]=tmp2_nolam4
    lam_438.t[a]=num*timescale
    lam_438.dose[a]=dosetmp
    lam_438.dose_plot[a]=(num-shutter_oppenned)*dose_step
    lam_438.l_438[a]=ready_spec_nolam4[a].A[closest(ready_spec_nolam4[a].wl, 438)]
    lam_550.t[a]=num*timescale
    lam_550.dose[a]=dosetmp
    lam_550.dose_plot[a]=(num-shutter_oppenned)*dose_step
    lam_550.l_550[a]=ready_spec_nolam4[a].A[closest(ready_spec_nolam4[a].wl, 550)]
    lam_464.t[a]=num*timescale
    lam_464.dose[a]=dosetmp
    lam_464.dose_plot[a]=(num-shutter_oppenned)*dose_step
    lam_464.l_464[a]=ready_spec_nolam4[a].A[closest(ready_spec_nolam4[a].wl, 464)]
    scat_464.t[a]=num*timescale
    scat_464.dose[a]=dosetmp
    scat_464.dose_plot[a]=(num-shutter_oppenned)*dose_step
    scat_464.l_464[a]=ready_spec[a].A[closest(ready_spec[a].wl, 464)]
    lam_620.t[a]=num*timescale
    lam_620.dose[a]=dosetmp
    lam_620.dose_plot[a]=(num-shutter_oppenned)*dose_step
    lam_620.l_620[a]=ready_spec_nolam4[a].A[closest(ready_spec_nolam4[a].wl, 620)]

#%% from the start


#%%% lam 464 plot

# Option 1

# 


startexpo=shutter_oppenned*timescale

finexpo=lam_464.t.max()

# plt.plot(lam_464.t,lam_464.l_464)

lam_464=lam_464.sort_values("t")  

initialpara464meters = np.array([0.15, 0.70, 40])
sigma=np.array(len(lam_464.t[lam_464.t.between(startexpo,finexpo, inclusive="both")])*[1])
x=np.array(lam_464.dose[lam_464.t.between(startexpo,finexpo, inclusive="both")])
y=np.array(lam_464.l_464[lam_464.t.between(startexpo,finexpo, inclusive="both")])
para464, pcov = sp.optimize.curve_fit(f=fct_relaxation_monoexp, xdata=x, ydata=y, p0=initialpara464meters, sigma=sigma)

pd.DataFrame(data=[para464,pcov], 
              index=["para464meters","covariance"], 
              columns = ["a","b","tau"]).to_csv(path_or_buf="464_model_para464.tsv", sep = "\t")

for num in range(0,len(listspec),1):   #The equivalent of "For f in ./*" in bash
    a=spec.paths[num]
    model_a=fct_relaxation_monoexp(lam_464.dose[a],para464[0],para464[1],para464[2])
    print(a, model_a)
    lam_464.loc[a,'model']=model_a

lam_464['t_shutter']=lam_464.t-shutter_oppenned*timescale

delta=float(lam_464.l_464[lam_464.t==(shutter_oppenned*timescale)]-lam_464.l_464.min())

globmax=lam_464.l_464.max()
globmin=lam_464.l_464.min()

demidose_464=float(lam_464.dose[lam_464.l_464==closest(lam_464.l_464, delta/2+globmin)])
dose80_464=float(lam_464.dose[lam_464.l_464==closest(lam_464.l_464, delta/5+globmin)])


# plt.plot(lam_464.dose_plot,lam_464.l_464)
# plt.axvline(demidose_464, color='blue', ls='-.')
# plt.axvline(dose80_464, color='red', ls='-.')
# plt.show()



for i in lam_464.index:
    lam_464.loc[i,'p_abs']=(lam_464.l_464[i]-globmin)/delta

for i in lam_464.index:
    lam_464.loc[i,'p_model']=(lam_464.model[i]-globmin)/delta

####Plotting the data####


def format_func(value, tick_number):
    # renames tick values 
    N = value
    print(N)
    dose_eq=lam_464.loc[value/timescale,'dose']
    return dose_eq
    

plt.rcParams["figure.figsize"] = (20/2.54,15/2.54)
fig, ax = plt.subplots()     #First let's create our figure, subplots ensures we can plot several curves on the same graph
ax.set_xlabel('Absorbed dose [kGy]', fontsize=15)  #x axis 
ax.xaxis.set_label_coords(x=0.5, y=-0.05)      #This determines where the x-axis is on the figure 
ax.set_ylabel('Absorbance', fontsize=15)               #Label of the y axis
ax.yaxis.set_label_coords(x=-0, y=0.5)       #position of the y axis 
palette=sns.color_palette(palette="bright", n_colors=2)   #This creates a palette with distinct colors in function of the number of sample, check it at https://seaborn.pydata.org/tutorial/color_palettes.html, in our case we might want to cherry-pick our colors, that's easy: palette are only lists of rgb triplets. Seaborn has a "desat" var, it modulates intensity of the color we can probably use that for emission/excitation plots 

n=0                          #this is just a counter for the palette, it's ugly as hell but hey, it works 
                  #We can then parse over our dictionary to plot our data
ax.plot(lam_464.dose_plot,
        lam_464.p_abs , 
        'bo',
        markersize=3,
        # label="calculated half dose is "+format(demidose_464, '.3f')
        )

ax.plot(lam_464.dose_plot,
        lam_464.p_model,             #y-axis is abs, or emission, or else
        linewidth=1,              #0.5 : pretty thin, 2 : probably what Hadrien used 
        label="demidose is: "+format(demidose_464, '.3f'),                  #Label is currently the name of our file, we could replace that by a list of names
        color=palette[1])

plt.plot((demidose_464, demidose_464), (0, lam_464.p_abs[lam_464.dose_plot==demidose_464]), 'green')
plt.plot((0, demidose_464), (lam_464.p_abs[lam_464.dose_plot==demidose_464], lam_464.p_abs[lam_464.dose_plot==demidose_464]), 'green')

plt.plot((dose80_464, dose80_464), (0, lam_464.p_abs[lam_464.dose_plot==dose80_464]), 'red')
plt.plot((0, dose80_464), (lam_464.p_abs[lam_464.dose_plot==dose80_464], lam_464.p_abs[lam_464.dose_plot==dose80_464]), 'red')

# timeandspace=[100,200,300,400,500,600,700,800,90-100,500]*dose_step
# timeandspace = [x*dose_step for x in [100,200,300,400,500,600,700,800,90-100,500]]

# # labels = [ax.xaxis.get_text() for item in ax.get_xticklabels()]
# Move left y-axis and bottim x-axis to centre, passing through (0,0)

ax.spines['left'].set_position('zero' )
ax.spines['bottom'].set_position('zero' )

# Eliminate upper and right axes
# ax.spines['right'].set_color('none')
# ax.spines['top'].set_color('none')

# ax.xaxis.set_major_formatter(plt.FuncFormatter(format_func))
# ax.set_title('Absorbance at 464nm over dose after xray exposure', fontsize=15, fontweight='bold')  #This sets the title of the plot
ax.set_xlim([lam_464.dose_plot.min(), lam_464.dose_plot.max()]) 
# ax.set_ylim([-10,110])
ax.tick_params(labelsize=10)
ax.xaxis.set_ticks([100,200,300,400,500,600,700,800,90-100,500])
# ax.yaxis.set_ticks(np.arange(0,110,10))  #This modulates the frequency of the x label (1, 50 ,40 ect)

legend = plt.legend(loc='upper right', shadow=True, prop={'size':8})

# # plt.show()
# # ################################################
# figfilename = "xtal4_l464.svg"             #Filename, for the PDF output, don't forget to close the last one before executing again, otherwise python can't write over it
# plt.savefig(figfilename, dpi=3000, transparent=True)
# figfilename = "xtal4_l464.png"             #Filename, for the PDF output, don't forget to close the last one before executing again, otherwise python can't write over it
# plt.savefig(figfilename, dpi=3000, transparent=True)
# figfilename = "xtal4_l464.pdf"             #Filename, for the PDF output, don't forget to close the last one before executing again, otherwise python can't write over it
# plt.savefig(figfilename, dpi=3000, transparent=True)
# plt.close()






#%%% lam 550 plot

# Option 1

startexpo=shutter_oppenned*timescale

finexpo=lam_550.t.max()

# plt.plot(lam_550.t,lam_550.l_550)

lam_550=lam_550.sort_values("t")  

initialpara550meters = np.array([0.15, 0.70, 40])
sigma=np.array(len(lam_550.t[lam_550.t.between(startexpo,finexpo, inclusive="both")])*[1])
x=np.array(lam_550.dose[lam_550.t.between(startexpo,finexpo, inclusive="both")])
y=np.array(lam_550.l_550[lam_550.t.between(startexpo,finexpo, inclusive="both")])
para550, pcov = sp.optimize.curve_fit(f=fct_relaxation_monoexp, xdata=x, ydata=y, p0=initialpara550meters, sigma=sigma)

pd.DataFrame(data=[para550,pcov], 
              index=["para550meters","covariance"], 
              columns = ["a","b","tau"]).to_csv(path_or_buf="550_model_para550.tsv", sep = "\t")

for num in range(0,len(listspec),1):   #The equivalent of "For f in ./*" in bash
    a=spec.paths[num]
    model_a=fct_relaxation_monoexp(lam_550.dose[a],para550[0],para550[1],para550[2])
    print(a, model_a)
    lam_550.loc[a,'model']=model_a

lam_550['t_shutter']=lam_550.t-shutter_oppenned*timescale

delta=float(lam_550.l_550[lam_550.t==(shutter_oppenned*timescale)]-lam_550.l_550.min())

globmax=lam_550.l_550.max()
globmin=lam_550.l_550.min()

demidose_550=float(lam_550.dose[lam_550.l_550==closest(lam_550.l_550, delta/2+globmin)])
dose80_550=float(lam_550.dose[lam_550.l_550==closest(lam_550.l_550, delta/5+globmin)])

for i in lam_550.index:
    lam_550.loc[i,'p_abs']=(lam_550.l_550[i]-globmin)/delta

for i in lam_550.index:
    lam_550.loc[i,'p_model']=(lam_550.model[i]-globmin)/delta

# ####Plotting the data####


# def format_func(value, tick_number):
#     # renames tick values 
#     N = value
#     print(N)
#     dose_eq=lam_550.loc[value/timescale,'dose']
#     return dose_eq
    

# plt.rcParams["figure.figsize"] = (20/2.54,15/2.54)
# fig, ax = plt.subplots()     #First let's create our figure, subplots ensures we can plot several curves on the same graph
# ax.set_xlabel('Absorbed dose [kGy]', fontsize=15)  #x axis 
# ax.xaxis.set_label_coords(x=0.5, y=-0.05)      #This determines where the x-axis is on the figure 
# ax.set_ylabel('Absorbance', fontsize=15)               #Label of the y axis
# ax.yaxis.set_label_coords(x=-0, y=0.5)       #position of the y axis 
# palette=sns.color_palette(palette="bright", n_colors=2)   #This creates a palette with distinct colors in function of the number of sample, check it at https://seaborn.pydata.org/tutorial/color_palettes.html, in our case we might want to cherry-pick our colors, that's easy: palette are only lists of rgb triplets. Seaborn has a "desat" var, it modulates intensity of the color we can probably use that for emission/excitation plots 

# n=0                          #this is just a counter for the palette, it's ugly as hell but hey, it works 
#                   #We can then parse over our dictionary to plot our data
# ax.plot(lam_550.dose_plot, 
#         lam_550.p_abs , 
#         'bo',
#         markersize=3,
#         # label="calculated half dose is "+format(demidose_550, '.3f')◘
#         )

# ax.plot(lam_550.dose_plot,
#         lam_550.p_model,             #y-axis is abs, or emission, or else
#         linewidth=1,              #0.5 : pretty thin, 2 : probably what Hadrien used 
#         label="demidose is: "+format(demidose_550, '.3f'),                  #Label is currently the name of our file, we could replace that by a list of names
#         color=palette[1])

# plt.plot((demidose_550, demidose_550), (0, lam_550.p_abs[lam_550.dose_plot==demidose_550]), 'green')
# plt.plot((0, demidose_550), (lam_550.p_abs[lam_550.dose_plot==demidose_550], lam_550.p_abs[lam_550.dose_plot==demidose_550]), 'green')

# plt.plot((dose80_550, dose80_550), (0, lam_550.p_abs[lam_550.dose_plot==dose80_550]), 'red')
# plt.plot((0, dose80_550), (lam_550.p_abs[lam_550.dose_plot==dose80_550], lam_550.p_abs[lam_550.dose_plot==dose80_550]), 'red')

# # timeandspace=[100,200,300,400,500,600,700,800,90-100,500]*dose_step
# # timeandspace = [x*dose_step for x in [100,200,300,400,500,600,700,800,90-100,500]]

# # # labels = [ax.xaxis.get_text() for item in ax.get_xticklabels()]
# # Move left y-axis and bottim x-axis to centre, passing through (0,0)

# ax.spines['left'].set_position('zero' )
# ax.spines['bottom'].set_position('zero' )

# # Eliminate upper and right axes
# # ax.spines['right'].set_color('none')
# # ax.spines['top'].set_color('none')

# # ax.xaxis.set_major_formatter(plt.FuncFormatter(format_func))
# # ax.set_title('Absorbance at 550nm over dose after xray exposure', fontsize=15, fontweight='bold')  #This sets the title of the plot
# ax.set_xlim([lam_550.dose_plot.min(), lam_550.dose_plot.max()]) 
# # ax.set_ylim([-10,110])
# ax.tick_params(labelsize=10)
# ax.xaxis.set_ticks([100,200,300,400,500,600,700,800,90-100,500])
# # ax.yaxis.set_ticks(np.arange(0,110,10))  #This modulates the frequency of the x label (1, 50 ,40 ect)

# legend = plt.legend(loc='upper right', shadow=True, prop={'size':8})

# # plt.show()
# # ################################################
# figfilename = "xtal4_l550.svg"             #Filename, for the PDF output, don't forget to close the last one before executing again, otherwise python can't write over it
# plt.savefig(figfilename, dpi=3000, transparent=True)
# figfilename = "xtal4_l550.png"             #Filename, for the PDF output, don't forget to close the last one before executing again, otherwise python can't write over it
# plt.savefig(figfilename, dpi=3000, transparent=True)
# figfilename = "xtal4_l550.pdf"             #Filename, for the PDF output, don't forget to close the last one before executing again, otherwise python can't write over it
# plt.savefig(figfilename, dpi=3000, transparent=True)
# plt.close()


#%%% lam 438 plot

# Option 1

# plt.plot(lam_438.t,lam_438.l_438)


startexpo=shutter_oppenned*timescale

finexpo=lam_438.t.max()

# plt.plot(lam_438.t,lam_438.l_438)

lam_438=lam_438.sort_values("t")  

initialpara438meters = np.array([0.15, 0.70, 40])
sigma=np.array(len(lam_438.t[lam_438.t.between(startexpo,finexpo, inclusive="both")])*[1])
x=np.array(lam_438.dose[lam_438.t.between(startexpo,finexpo, inclusive="both")])
y=np.array(lam_438.l_438[lam_438.t.between(startexpo,finexpo, inclusive="both")])
para438, pcov = sp.optimize.curve_fit(f=fct_relaxation_monoexp, xdata=x, ydata=y, p0=initialpara438meters, sigma=sigma)

pd.DataFrame(data=[para438,pcov], 
              index=["para438meters","covariance"], 
              columns = ["a","b","tau"]).to_csv(path_or_buf="438_model_para438.tsv", sep = "\t")

for num in range(0,len(listspec),1):   #The equivalent of "For f in ./*" in bash
    a=spec.paths[num]
    model_a=fct_relaxation_monoexp(lam_438.dose[a],para438[0],para438[1],para438[2])
    print(a, model_a)
    lam_438.loc[a,'model']=model_a

lam_438['t_shutter']=lam_438.t-shutter_oppenned*timescale

delta=float(lam_438.l_438[lam_438.t==(shutter_oppenned*timescale)]-lam_438.l_438.min())

globmax=lam_438.l_438.max()
globmin=lam_438.l_438.min()

demidose_438=float(lam_438.dose[lam_438.l_438==closest(lam_438.l_438, delta/2+globmin)])
dose80_438=float(lam_438.dose[lam_438.l_438==closest(lam_438.l_438, delta/5+globmin)])

for i in lam_438.index:
    lam_438.loc[i,'p_abs']=(lam_438.l_438[i]-globmin)/delta

for i in lam_438.index:
    lam_438.loc[i,'p_model']=(lam_438.model[i]-globmin)/delta

# ####Plotting the data####


# def format_func(value, tick_number):
#     # renames tick values 
#     N = value
#     print(N)
#     dose_eq=lam_438.loc[value/timescale,'dose']
#     return dose_eq
    

# plt.rcParams["figure.figsize"] = (20/2.54,15/2.54)
# fig, ax = plt.subplots()     #First let's create our figure, subplots ensures we can plot several curves on the same graph
# ax.set_xlabel('Absorbed dose [kGy]', fontsize=15)  #x axis 
# ax.xaxis.set_label_coords(x=0.5, y=-0.05)      #This determines where the x-axis is on the figure 
# ax.set_ylabel('Absorbance', fontsize=15)               #Label of the y axis
# ax.yaxis.set_label_coords(x=-0, y=0.5)       #position of the y axis 
# palette=sns.color_palette(palette="bright", n_colors=2)   #This creates a palette with distinct colors in function of the number of sample, check it at https://seaborn.pydata.org/tutorial/color_palettes.html, in our case we might want to cherry-pick our colors, that's easy: palette are only lists of rgb triplets. Seaborn has a "desat" var, it modulates intensity of the color we can probably use that for emission/excitation plots 

# n=0                          #this is just a counter for the palette, it's ugly as hell but hey, it works 
#                   #We can then parse over our dictionary to plot our data
# ax.plot(lam_438.dose_plot, 
#         lam_438.p_abs , 
#         'bo',
#         markersize=3,
#         # label="calculated half dose is "+format(demidose_438, '.3f')◘
#         )

# ax.plot(lam_438.dose_plot,
#         lam_438.p_model,             #y-axis is abs, or emission, or else
#         linewidth=1,              #0.5 : pretty thin, 2 : probably what Hadrien used 
#         label="demidose is: "+format(demidose_438, '.3f'),                  #Label is currently the name of our file, we could replace that by a list of names
#         color=palette[1])

# plt.plot((demidose_438, demidose_438), (0, lam_438.p_abs[lam_438.dose_plot==demidose_438]), 'green')
# plt.plot((0, demidose_438), (lam_438.p_abs[lam_438.dose_plot==demidose_438], lam_438.p_abs[lam_438.dose_plot==demidose_438]), 'green')

# plt.plot((dose80_438, dose80_438), (0, lam_438.p_abs[lam_438.dose_plot==dose80_438]), 'red')
# plt.plot((0, dose80_438), (lam_438.p_abs[lam_438.dose_plot==dose80_438], lam_438.p_abs[lam_438.dose_plot==dose80_438]), 'red')

# # timeandspace=[100,200,300,400,500,600,700,800,90-100,500]*dose_step
# # timeandspace = [x*dose_step for x in [100,200,300,400,500,600,700,800,90-100,500]]

# # # labels = [ax.xaxis.get_text() for item in ax.get_xticklabels()]
# # Move left y-axis and bottim x-axis to centre, passing through (0,0)

# ax.spines['left'].set_position('zero' )
# ax.spines['bottom'].set_position('zero' )

# # Eliminate upper and right axes
# # ax.spines['right'].set_color('none')
# # ax.spines['top'].set_color('none')

# # ax.xaxis.set_major_formatter(plt.FuncFormatter(format_func))
# # ax.set_title('Absorbance at 438nm over dose after xray exposure', fontsize=15, fontweight='bold')  #This sets the title of the plot
# ax.set_xlim([lam_438.dose_plot.min(), lam_438.dose_plot.max()]) 
# # ax.set_ylim([-10,110])
# ax.tick_params(labelsize=10)
# ax.xaxis.set_ticks([100,200,300,400,500,600,700,800,90-100,500])
# # ax.yaxis.set_ticks(np.arange(0,110,10))  #This modulates the frequency of the x label (1, 50 ,40 ect)

# legend = plt.legend(loc='upper right', shadow=True, prop={'size':8})

# # plt.show()
# # ################################################
# figfilename = "xtal4_l438.svg"             #Filename, for the PDF output, don't forget to close the last one before executing again, otherwise python can't write over it
# plt.savefig(figfilename, dpi=3000, transparent=True)
# figfilename = "xtal4_l438.png"             #Filename, for the PDF output, don't forget to close the last one before executing again, otherwise python can't write over it
# plt.savefig(figfilename, dpi=3000, transparent=True)
# figfilename = "xtal4_l438.pdf"             #Filename, for the PDF output, don't forget to close the last one before executing again, otherwise python can't write over it
# plt.savefig(figfilename, dpi=3000, transparent=True)
# plt.close()


 
#%%% lam 620 plot

# Option 1

# plt.plot(lam_620.dose_plot[lam_620.t.between((shutter_oppenned-1)*timescale,finexpo,inclusive='both')],lam_620.l_620[lam_620.t.between((shutter_oppenned-1)*timescale,finexpo,inclusive='both')])






startexpo=shutter_oppenned*timescale

finexpo=lam_620.t.max()

# plt.plot(lam_620.t,lam_620.l_620)

lam_620=lam_620.sort_values("t")  

initialpara620meters = np.array([0.15, 0.70, 40])
sigma=np.array(len(lam_620.t[lam_620.t.between(startexpo,finexpo, inclusive="both")])*[1])
x=np.array(lam_620.dose[lam_620.t.between(startexpo,finexpo, inclusive="both")])
y=np.array(lam_620.l_620[lam_620.t.between(startexpo,finexpo, inclusive="both")])
para620, pcov = sp.optimize.curve_fit(f=fct_relaxation_monoexp, xdata=x, ydata=y, p0=initialpara620meters, sigma=sigma)

pd.DataFrame(data=[para620,pcov], 
              index=["para620meters","covariance"], 
              columns = ["a","b","tau"]).to_csv(path_or_buf="620_model_para620.tsv", sep = "\t")

for num in range(0,len(listspec),1):   #The equivalent of "For f in ./*" in bash
    a=spec.paths[num]
    model_a=fct_relaxation_monoexp(lam_620.dose[a],para620[0],para620[1],para620[2])
    print(a, model_a)
    lam_620.loc[a,'model']=model_a

lam_620['t_shutter']=lam_620.t-shutter_oppenned*timescale

delta=float(lam_620.l_620[lam_620.t==(shutter_oppenned*timescale)]-lam_620.l_620.min())

globmax=lam_620.l_620.max()
globmin=lam_620.l_620.min()

demidose_620=float(lam_620.dose[lam_620.l_620==closest(lam_620.l_620, delta/2+globmin)])
dose80_620=float(lam_620.dose[lam_620.l_620==closest(lam_620.l_620, delta/5+globmin)])

for i in lam_620.index:
    lam_620.loc[i,'p_abs']=(lam_620.l_620[i]-globmin)/delta

for i in lam_620.index:
    lam_620.loc[i,'p_model']=(lam_620.model[i]-globmin)/delta

# ####Plotting the data####


# def format_func(value, tick_number):
#     # renames tick values 
#     N = value
#     print(N)
#     dose_eq=lam_620.loc[value/timescale,'dose']
#     return dose_eq
    

# plt.rcParams["figure.figsize"] = (20/2.54,15/2.54)
# fig, ax = plt.subplots()     #First let's create our figure, subplots ensures we can plot several curves on the same graph
# ax.set_xlabel('Absorbed dose [kGy]', fontsize=15)  #x axis 
# ax.xaxis.set_label_coords(x=0.5, y=-0.05)      #This determines where the x-axis is on the figure 
# ax.set_ylabel('Absorbance', fontsize=15)               #Label of the y axis
# ax.yaxis.set_label_coords(x=-0, y=0.5)       #position of the y axis 
# palette=sns.color_palette(palette="bright", n_colors=2)   #This creates a palette with distinct colors in function of the number of sample, check it at https://seaborn.pydata.org/tutorial/color_palettes.html, in our case we might want to cherry-pick our colors, that's easy: palette are only lists of rgb triplets. Seaborn has a "desat" var, it modulates intensity of the color we can probably use that for emission/excitation plots 

# n=0                          #this is just a counter for the palette, it's ugly as hell but hey, it works 
#                   #We can then parse over our dictionary to plot our data
# ax.plot(lam_620.dose_plot, 
#         lam_620.p_abs , 
#         'bo',
#         markersize=3,
#         # label="calculated half dose is "+format(demidose_620, '.3f')◘
#         )

# ax.plot(lam_620.dose_plot,
#         lam_620.p_model,             #y-axis is abs, or emission, or else
#         linewidth=1,              #0.5 : pretty thin, 2 : probably what Hadrien used 
#         label="demidose is: "+format(demidose_620, '.3f'),                  #Label is currently the name of our file, we could replace that by a list of names
#         color=palette[1])

# plt.plot((demidose_620, demidose_620), (0, lam_620.p_abs[lam_620.dose_plot==demidose_620]), 'green')
# plt.plot((0, demidose_620), (lam_620.p_abs[lam_620.dose_plot==demidose_620], lam_620.p_abs[lam_620.dose_plot==demidose_620]), 'green')

# plt.plot((dose80_620, dose80_620), (0, lam_620.p_abs[lam_620.dose_plot==dose80_620]), 'red')
# plt.plot((0, dose80_620), (lam_620.p_abs[lam_620.dose_plot==dose80_620], lam_620.p_abs[lam_620.dose_plot==dose80_620]), 'red')

# # timeandspace=[100,200,300,400,500,600,700,800,90-100,500]*dose_step
# # timeandspace = [x*dose_step for x in [100,200,300,400,500,600,700,800,90-100,500]]

# # # labels = [ax.xaxis.get_text() for item in ax.get_xticklabels()]
# # Move left y-axis and bottim x-axis to centre, passing through (0,0)

# ax.spines['left'].set_position('zero' )
# ax.spines['bottom'].set_position('zero' )

# # Eliminate upper and right axes
# # ax.spines['right'].set_color('none')
# # ax.spines['top'].set_color('none')

# # ax.xaxis.set_major_formatter(plt.FuncFormatter(format_func))
# # ax.set_title('Absorbance at 620nm over dose after xray exposure', fontsize=15, fontweight='bold')  #This sets the title of the plot
# ax.set_xlim([lam_620.dose_plot.min(), lam_620.dose_plot.max()]) 
# # ax.set_ylim([-10,110])
# ax.tick_params(labelsize=10)
# ax.xaxis.set_ticks([100,200,300,400,500,600,700,800,90-100,500])
# # ax.yaxis.set_ticks(np.arange(0,110,10))  #This modulates the frequency of the x label (1, 50 ,40 ect)

# legend = plt.legend(loc='upper right', shadow=True, prop={'size':8})

# # plt.show()
# # ################################################
# figfilename = "xtal4_l620.svg"             #Filename, for the PDF output, don't forget to close the last one before executing again, otherwise python can't write over it
# plt.savefig(figfilename, dpi=3000, transparent=True)
# figfilename = "xtal4_l620.png"             #Filename, for the PDF output, don't forget to close the last one before executing again, otherwise python can't write over it
# plt.savefig(figfilename, dpi=3000, transparent=True)
# figfilename = "xtal4_l620.pdf"             #Filename, for the PDF output, don't forget to close the last one before executing again, otherwise python can't write over it
# plt.savefig(figfilename, dpi=3000, transparent=True)
# plt.close()
    





# pd.DataFrame(data=[float(para438[2]),float(para464[2]),float(parascat464[2]),float(para550[2]),float(para620[2])], 
#               index=["l438","l464","scat464","l550","l620"], 
#               columns = ["xtal4"]).transpose().to_csv(path_or_buf="allpar.tsv", sep = "\t")


pd.DataFrame(data=[dose_s,demidose_438,demidose_464,demidose_550,demidose_620,dose80_438,dose80_464,dose80_550,dose80_620], 
              index=["dose_per_seconds","dose50_l438","dose50_l464","dose50_l550","dose50_l620","dose80_l438","dose80_l464","dose80_l580","dose80_l620"], 
              columns = ["xtal4"]).transpose().to_csv(path_or_buf="dosepar.tsv", sep = "\t")



#%% fig for article 
 
#%%% lam 620 article plot

# Option 1

# plt.plot(lam_620.dose_plot[lam_620.t.between((shutter_oppenned-1)*timescale,finexpo,inclusive='both')],lam_620.l_620[lam_620.t.between((shutter_oppenned-1)*timescale,finexpo,inclusive='both')])





delta=float(lam_620.l_620[lam_620.t==(shutter_oppenned*timescale)]-lam_620.l_620.min())

globmax=lam_620.l_620.max()
globmin=lam_620.l_620.min()

demidose_620=float(lam_620.dose[lam_620.l_620==closest(lam_620.l_620, delta/2+globmin)])
dose80_620=float(lam_620.dose[lam_620.l_620==closest(lam_620.l_620, delta/5+globmin)])

for i in lam_620.index:
    lam_620.loc[i,'p_abs']=100*(lam_620.l_620[i]-globmin)/delta

####Plotting the data####


def format_func(value, tick_number):
    # renames tick values 
    N = value
    print(N)
    dose_eq=lam_620.loc[value/timescale,'dose']
    return dose_eq
    

plt.rcParams["figure.figsize"] = (20/2.54,15/2.54)
fig, ax = plt.subplots()     #First let's create our figure, subplots ensures we can article plot several curves on the same graph
ax.set_xlabel('Absorbed dose [kGy]', fontsize=15)  #x axis 
ax.xaxis.set_label_coords(x=0.5, y=-0.05)      #This determines where the x-axis is on the figure 
ax.set_ylabel('% of initial Absorbance', fontsize=15)               #Label of the y axis
ax.yaxis.set_label_coords(x=-0, y=0.5)       #position of the y axis 
palette=sns.color_palette(palette="bright", n_colors=2)   #This creates a palette with distinct colors in function of the number of sample, check it at https://seaborn.pydata.org/tutorial/color_palettes.html, in our case we might want to cherry-pick our colors, that's easy: palette are only lists of rgb triplets. Seaborn has a "desat" var, it modulates intensity of the color we can probably use that for emission/excitation article plots 

n=0                          #this is just a counter for the palette, it's ugly as hell but hey, it works 
                  #We can then parse over our dictionary to article plot our data
ax.plot(lam_620.dose_plot[lam_620.dose_plot.between(-100,500)], 
        lam_620.p_abs[lam_620.dose_plot.between(-100,500)] ,
        'bo',
        markersize=3,
        # label="calculated half dose is "+format(demidose_620, '.3f')◘
        )

print("The end of exponential is :" + lam_620.index[lam_620.dose_plot.between(-100,500)][-1])
# plt.plot((demidose_620, demidose_620), (0, lam_620.p_abs[lam_620.dose_plot==demidose_620]), 'green')
# plt.plot((0, demidose_620), (lam_620.p_abs[lam_620.dose_plot==demidose_620], lam_620.p_abs[lam_620.dose_plot==demidose_620]), 'green')

plt.plot((dose80_620, dose80_620), (0, lam_620.p_abs[lam_620.dose_plot==dose80_620]), 'red')
plt.plot((0, dose80_620), (lam_620.p_abs[lam_620.dose_plot==dose80_620], lam_620.p_abs[lam_620.dose_plot==dose80_620]), 'red')

# timeandspace=[100,200,300,400,500,600,700,800,90-100,500]*dose_step
# timeandspace = [x*dose_step for x in [100,200,300,400,500,600,700,800,90-100,500]]

# # labels = [ax.xaxis.get_text() for item in ax.get_xticklabels()]
# Move left y-axis and bottim x-axis to centre, passing through (0,0)

ax.spines['left'].set_position('zero' )
ax.spines['bottom'].set_position('zero' )

# Eliminate upper and right axes
# ax.spines['right'].set_color('none')
# ax.spines['top'].set_color('none')

# ax.xaxis.set_major_formatter(plt.FuncFormatter(format_func))
# ax.set_title('Absorbance at 620nm over dose after xray exposure', fontsize=15, fontweight='bold')  #This sets the title of the article plot
ax.set_xlim([-100,500]) 
# ax.set_xlim([-10,920])
ax.tick_params(labelsize=10)
ax.xaxis.set_ticks([0,100,200,300,400,500])
# ax.yaxis.set_ticks(np.arange(0,110,10))  #This modulates the frequency of the x label (1, 50 ,40 ect)

# legend = plt.legend(loc='upper right', shadow=True, prop={'size':8})

# plt.show()
# ################################################
figfilename = "xtal4_article_l620.svg"             #Filename, for the PDF output, don't forget to close the last one before executing again, otherwise python can't write over it
plt.savefig(figfilename, dpi=3000, transparent=True)
figfilename = "xtal4_article_l620.png"             #Filename, for the PDF output, don't forget to close the last one before executing again, otherwise python can't write over it
plt.savefig(figfilename, dpi=3000, transparent=True)
figfilename = "xtal4_article_l620.pdf"             #Filename, for the PDF output, don't forget to close the last one before executing again, otherwise python can't write over it
plt.savefig(figfilename, dpi=3000, transparent=True)
plt.close()



# #%% general


# #%%% scattering corrected

# listmax=[]
# listmin=[]
# for i in ready_spec :
#     a=ready_spec[i].A[ready_spec[i].wl.between(280,700)].max()
#     if not (mth.isinf(a) | mth.isnan(a)):
#         listmax.append(a)
#     a=ready_spec[i].A[ready_spec[i].wl.between(700,800)].min()
#     if not (mth.isinf(a) | mth.isnan(a)):
#         listmin.append(a)
# globmax=max(listmax)
# globmin=min(listmin)        


# fig, ax = plt.subplots()     #First let's create our figure, subplots ensures we can plot several curves on the same graph
# ax.set_xlabel('Wavelength [nm]', fontsize=10)  #x axis 
# ax.xaxis.set_label_coords(x=0.5, y=-0.08)      #This determines where the x-axis is on the figure 
# ax.set_ylabel('Absorbance [AU]', fontsize=10)               #Label of the y axis
# ax.yaxis.set_label_coords(x=-0.1, y=0.5)       #position of the y axis 

# # ax = ax1.twinx()                                #Guillaume did that weird OD + Abs y-axis, don't know why, we can probably remove that and replace ax and #  ax1 by simply "ax"
# # ax.set_ylabel('Absorbance [AU]', fontsize=10)  #label of our secondary y-axis 
# # ax.yaxis.set_label_coords(x=1.09, y=0.5)        #Position of our secondary y-axis
# palette=sns.color_palette(palette='Spectral', n_colors=len(ready_spec))   #This creates a palette with distinct colors in function of the number of sample, check it at https://seaborn.pydata.org/tutorial/color_palettes.html, in our case we might want to cherry-pick our colors, that's easy: palette are only lists of rgb triplets. Seaborn has a "desat" var, it modulates intensity of the color we can probably use that for emission/excitation plots 

# n=0                                            #this is just a counter for the palette, it's ugly as hell but hey, it works 
# for i in ready_spec :                          #We can then parse over our dictionary to plot our data
#     ax.plot(ready_spec[i].wl,                  #x-axis is wavelength
#               ready_spec[i].A ,                   #y-axis is abs, or emission, or else
#               linewidth=1,                    #0.5 : pretty thin, 2 : probably what Hadrien used 
#               label=i,                        #Label is currently the name of our file, we could replace that by a list of names
#               color=palette[n])               #This determines the color of the curves, you can create a custom list of colors such a c['blue','red'] ect
#     n=n+1
# ax.set_title('scattering corrected in crystallo absorbance spectra', fontsize=10, fontweight='bold')  #This sets the title of the plot
# # ax.set_xlim([260, 600])                        
# # ax.set_ylim([-0.1, 1.1])
# ax.set_xlim([250, 900]) 
# ax.set_ylim([globmin-0.1, globmax+0.1])

# #  ax1.tick_params(labelsize=10)   #This sets the font, we can probably use stuff like Arial Narrow
# ax.tick_params(labelsize=10)
# #  ax1.yaxis.set_ticks(np.arange(0, 1.1, 0.5), pad=30)
# ax.yaxis.set_ticks(np.arange(int(10*globmin-1)/10, int(10*globmax+1)/10, 0.1))  #This modulates the frequency of the x label (1, 50 ,100 ect)

# # legend = plt.legend(loc='upper right', shadow=True, prop={'size':7})    #Where the legend rectangle is 

# # plt.show()    #Use this to check your figure before the output 

# # ################################################

# figfilename = "scattering_corrected_spec.pdf"             #Filename, for the PDF output, don't forget to close the last one before executing again, otherwise python can't write over it
# plt.savefig(figfilename, dpi=1000, transparent=True,bbox_inches='tight')   #Transparent setting removes the background grid which is the standard for pyplot. 
# figfilename = "scattering_corrected_spec.png"
# plt.savefig(figfilename, dpi=1000, transparent=True,bbox_inches='tight')   #Filename, for the png output
# plt.close()


#%%% cst corrected plot 
listmax=[]
listmin=[]
for i in ready_spec_nolam4 :
    a=ready_spec_nolam4[i].A[ready_spec_nolam4[i].wl.between(280,800)].max()
    if not (mth.isinf(a) | mth.isnan(a)):
        listmax.append(a)
    a=ready_spec_nolam4[i].A[ready_spec_nolam4[i].wl.between(750,800)].min()
    if not (mth.isinf(a) | mth.isnan(a)):
        listmin.append(a)
globmax=max(listmax)
globmin=min(listmin)

plt.rcParams["figure.figsize"] = (20/2.54,15/2.54)
fig, ax = plt.subplots()     #First let's create our figure, subplots ensures we can plot several curves on the same graph
ax.set_xlabel('Wavelength [nm]', fontsize=16)  #x axis 
ax.xaxis.set_label_coords(x=0.5, y=-0.08)      #This determines where the x-axis is on the figure 
ax.set_ylabel('Absorbance', fontsize=16)               #Label of the y axis
ax.yaxis.set_label_coords(x=-0.07, y=0.5)       #position of the y axis 

# ax = ax1.twinx()                                #Guillaume did that weird OD + Abs y-axis, don't know why, we can probably remove that and replace ax and #  ax1 by simply "ax"
# ax.set_ylabel('Absorbance [AU]', fontsize=10)  #label of our secondary y-axis 
# ax.yaxis.set_label_coords(x=1.09, y=0.5)        #Position of our secondary y-axis
palette=sns.color_palette(palette='Spectral', n_colors=len(list(lam_620.index[lam_620.dose_plot.between(0,500)])))   #This creates a palette with distinct colors in function of the number of sample, check it at https://seaborn.pydata.org/tutorial/color_palettes.html, in our case we might want to cherry-pick our colors, that's easy: palette are only lists of rgb triplets. Seaborn has a "desat" var, it modulates intensity of the color we can probably use that for emission/excitation plots 

n=0                                            #this is just a counter for the palette, it's ugly as hell but hey, it works 
for a in list(lam_620.index[lam_620.dose_plot.between(0,500)]) :#range(0,126):#i in ready_spec :                          #We can then parse over our dictionary to plot our data
    # i=list(ready_spec.keys())[a]    
    ax.plot(ready_spec_nolam4[a].wl,                  #x-axis is wavelength
            ready_spec_nolam4[a].A ,                   #y-axis is abs, or emission, or else
            linewidth=1,                    #0.5 : pretty thin, 2 : probably what Hadrien used 
            # label=i,                        #Label is currently the name of our file, we could replace that by a list of names
            color=palette[n])               #This determines the color of the curves, you can create a custom list of colors such a c['blue','red'] ect
    n=n+1
# ax.set_title('only scaled in crystallo absorbance spectra (no scattering correction)', fontsize=10, fontweight='bold')  #This sets the title of the plot
# ax.set_xlim([260, 600])                        
# ax.set_ylim([-0.1, 1.1])
ax.set_xlim([250, 800]) 
ax.set_ylim([globmin, globmax+0.1])
# ax.spines['left'].set_position('zero' )
ax.spines['bottom'].set_position('zero' )
#  ax1.tick_params(labelsize=10)   #This sets the font, we can probably use stuff like Arial Narrow
ax.tick_params(labelsize=11)
# #  ax1.yaxis.set_ticks(np.arange(0, 1.1, 0.5), pad=30)
ax.yaxis.set_ticks(np.arange(int(10*globmin-1)/10, int(10*globmax+1)/10, 0.5))  #This modulates the frequency of the x label (1, 50 ,100 ect)

# legend = plt.legend(loc='upper right', shadow=True, prop={'size':7})    #Where the legend rectangle is 

# plt.show()    #Use this to check your figure before the output 


# ################################################

figfilename = "constant-baseline_corrected_spec.pdf"             #Filename, for the PDF output, don't forget to close the last one before executing again, otherwise python can't write over it
plt.savefig(figfilename, dpi=3000, transparent=True,bbox_inches='tight')   #Transparent setting removes the background grid which is the standard for pyplot. 
figfilename = "constant-baseline_corrected_spec.png"
plt.savefig(figfilename, dpi=3000, transparent=True,bbox_inches='tight')   #Filename, for the png output
figfilename = "constant-baseline_corrected_spec.svg"
plt.savefig(figfilename, dpi=3000, transparent=True,bbox_inches='tight')   #Filename, for the png output
plt.close()

    
#raw_plot
listmax=[]
listmin=[]
for i in raw_spec :
    a=raw_spec[i].A[raw_spec[i].wl.between(200,800)].max()
    if not (mth.isinf(a) | mth.isnan(a)):
        listmax.append(a)
    a=raw_spec[i].A[raw_spec[i].wl.between(750,800)].min()
    if not (mth.isinf(a) | mth.isnan(a)):
        listmin.append(a)
globmax=max(listmax)
globmin=min(listmin)

    
#%%% raw spectra 
fig, ax = plt.subplots()     #First let's create our figure, subplots ensures we can plot several curves on the same graph
ax.set_xlabel('Wavelength [nm]', fontsize=10)  #x axis 
ax.xaxis.set_label_coords(x=0.5, y=-0.08)      #This determines where the x-axis is on the figure 
ax.set_ylabel('Absorbance [AU]', fontsize=10)               #Label of the y axis
ax.yaxis.set_label_coords(x=-0.1, y=0.5)       #position of the y axis 

# ax = ax1.twinx()                                #Guillaume did that weird OD + Abs y-axis, don't know why, we can probably remove that and replace ax and #  ax1 by simply "ax"
# ax.set_ylabel('Absorbance [AU]', fontsize=10)  #label of our secondary y-axis 
# ax.yaxis.set_label_coords(x=1.09, y=0.5)        #Position of our secondary y-axis
palette=sns.color_palette(palette='Spectral', n_colors=len(raw_spec))   #This creates a palette with distinct colors in function of the number of sample, check it at https://seaborn.pydata.org/tutorial/color_palettes.html, in our case we might want to cherry-pick our colors, that's easy: palette are only lists of rgb triplets. Seaborn has a "desat" var, it modulates intensity of the color we can probably use that for emission/excitation plots 

n=0                                            #this is just a counter for the palette, it's ugly as hell but hey, it works 
for i in raw_spec :                          #We can then parse over our dictionary to plot our data
    ax.plot(raw_spec[i].wl,                  #x-axis is wavelength
            raw_spec[i].A ,                   #y-axis is abs, or emission, or else
            linewidth=1,                    #0.5 : pretty thin, 2 : probably what Hadrien used 
            label=i,                        #Label is currently the name of our file, we could replace that by a list of names
            color=palette[n])               #This determines the color of the curves, you can create a custom list of colors such a c['blue','red'] ect
    n=n+1
ax.set_title('only scaled in crystallo absorbance spectra (no scattering correction)', fontsize=10, fontweight='bold')  #This sets the title of the plot
# ax.set_xlim([260, 600])                        
# ax.set_ylim([-0.1, 1.1])
ax.set_xlim([250, 900]) 
ax.set_ylim([globmin-0.1, globmax+0.1])

#  ax1.tick_params(labelsize=10)   #This sets the font, we can probably use stuff like Arial Narrow
ax.tick_params(labelsize=10)
# #  ax1.yaxis.set_ticks(np.arange(0, 1.1, 0.5), pad=30)
ax.yaxis.set_ticks(np.arange(int(10*globmin-1)/10, int(10*globmax+1)/10, 0.1))  #This modulates the frequency of the x label (1, 50 ,100 ect)

legend = plt.legend(loc='upper right', shadow=True, prop={'size':7})    #Where the legend rectangle is 

# plt.show()    #Use this to check your figure before the output 


# # ################################################

# figfilename = "raw_spectra.pdf"             #Filename, for the PDF output, don't forget to close the last one before executing again, otherwise python can't write over it
# plt.savefig(figfilename, dpi=1000, transparent=True,bbox_inches='tight')   #Transparent setting removes the background grid which is the standard for pyplot. 
# figfilename = "raw_spectra.png"
# plt.savefig(figfilename, dpi=1000, transparent=True,bbox_inches='tight')   #Filename, for the png output
# plt.close()






# towrite_raw_spectra=tmp.drop(columns=['wl','A']) #structure for the written table
# for spec in raw_spec:
#     towrite_raw_spectra[spec]=raw_spec[spec].A

# towrite_raw_spectra.to_csv("raw_spectra.csv", index=True)


# towrite_ready_spec_nolam4tra=tmp.drop(columns=['wl','A']) #structure for the written table
# for spec in ready_spec_nolam4:
#     towrite_ready_spec_nolam4tra[spec]=ready_spec_nolam4[spec].A
# towrite_ready_spec_nolam4tra.to_csv("constant-baseline_corrected_spec.csv", index=True)



# towrite_ready_spectra=tmp.drop(columns=['wl','A']) #structure for the written table
# for spec in ready_spec:
#     towrite_ready_spectra[spec]=ready_spec[spec].A
# towrite_ready_spectra.to_csv("scattering_corrected_spec.csv", index=True)
