# -*- coding: utf-8 -*-
"""
Created on Sun Dec 25 17:27:40 2022
@author: Nicolas Caramello
"""
from tkinter import *
from tkinter import ttk
import pandas as pd
import matplotlib.pyplot as plt
from tkinter import filedialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
# from matplotlib.figure import Figure
import os as osys
import math as mth
import seaborn as sns
import numpy as np
import scipy as sp
from statistics import mean
plt.rcParams["figure.figsize"] = (20/2.54,15/2.54)
# dev_nUV=1
# dev_nopeak=0.1
# dev_nIR=0.5
#specorr fct

def set_scrollbar():
    globcanvas.configure(scrollregion = globcanvas.bbox("all"))

def baselineconst(df,segmentend):
    """
    This function removes a cosntant baseline taken from 600 to 800 nm from the given spectrum.

    Parameters:
    - df (pandas.DataFrame): The dataframe to be processed.
    - segmentend (int): The index of the last point of the segment used to determine the baseline.

    Returns:
    - pandas.DataFrame: The corrected dataframe with the baseline removed.
    """
    a=df.copy()
    baseline=df.copy()
    baseline.A=mean(a.A[segmentend])
    corrected=a.copy()
    corrected.A=a.A.copy()-baseline.A
    return corrected
def rescale_corrected(df, wlmin, wlmax):
    """
    This function rescales the given spectrum to a maximum of 1.

    Parameters:
    - df (pandas.DataFrame): The dataframe to be rescaled.
    - wlmin (int): The minimum wavelength to consider when determining the maximum value.
    - wlmax (int): The maximum wavelength to consider when determining the maximum value.

    Returns:
    - pandas.DataFrame: The rescaled dataframe.
    """
    a=df.copy()
    fact=1/a.A[a.wl.between(wlmin,wlmax,inclusive='both')].max() #1/a.A[a.wl.between(425,440)].max()
    a.A=a.A.copy()*fact
    return(a.copy())

def fct_baseline(x, a, b):
    """
    A function that takes x values, a, and b and returns the value of a/x^4+b

    Parameters:
    x (numpy array): x values
    a (float): The parameter a for the function
    b (float): The parameter b for the function

    Returns:
    numpy array: The values of the function evaluated at the given x values
    """
    return a/np.power(x,4)+b
def linbase(x,a,b):
    """
    A function that takes x values, a, and b and returns the value of a*x+b

    Parameters:
    x (numpy array): x values
    a (float): The parameter a for the function
    b (float): The parameter b for the function

    Returns:
    numpy array: The values of the function evaluated at the given x values
    """
    return a*x+b

def baselinefitcorr_3seg_smooth(df,  segment1, segment2, segmentend, sigmaby3segment):
    """
    Fit and subtract a baseline from the absorbance data in a pandas DataFrame. The baseline is fit using a non-linear function with three segments, and the fit is weighted by the sigma values in the `sigmaby3segment` list. The resulting corrected data and the correction DataFrame are returned.

    Parameters
    ----------
    df : pandas DataFrame
        DataFrame containing the absorbance data. The DataFrame must have a 'wl' column for the wavelength and an 'A' column for the absorbance.
    segment1 : list
        List of indices for the first segment of the baseline fit.
    segment2 : list
        List of indices for the second segment of the baseline fit.
    segmentend : list
        List of indices for the third segment of the baseline fit.
    sigmaby3segment : list
        List of sigma values for the three segments of the baseline fit.

    Returns
    -------
    corrected : pandas DataFrame
        DataFrame containing the corrected absorbance data.
    correction : pandas DataFrame
        DataFrame containing the correction data.
    """
    x=df.wl[segmentend].copy()
    y=df.A[segmentend].copy()
    segment=segment1+segment2+segmentend
    tmp=df.copy()
    x=tmp.wl[segment].copy()
    y=tmp.A[segment].copy()
    initialParameters = np.array([1e9,1])
    n=len(tmp.A[segment1])
    sigma=n*[sigmaby3segment[0]]
    n=len(tmp.A[segment2])
    sigma=sigma + n*[sigmaby3segment[1]]
    n=len(tmp.A[segmentend])
    sigma=sigma + n*[sigmaby3segment[2]]
    para, pcov = sp.optimize.curve_fit(f=fct_baseline, xdata=x, ydata=y, p0=initialParameters, sigma=sigma)
    baseline=tmp.copy()
    baseline.A=fct_baseline(baseline.wl.copy(), *para)
    corrected=tmp.copy()
    corrected.A=tmp.A.copy()-baseline.A
    return(corrected, baseline)

#GUI fct

def open_file():
    """
    This function opens a file dialog to allow the user to select one or more files. It then reads the contents of the selected files into pandas DataFrames, plots the contents of the DataFrames using matplotlib, and displays the plot in a matplotlib canvas. 

    The function uses the following libraries: 
    - filedialog (from tkinter)
    - os.path
    - pandas
    - matplotlib
    - seaborn

    The function takes no arguments, but uses a global variable `m` to keep track of the number of plots displayed.
    """
    # m is a row counter for the plot frame
    global m
    # Prompt the user to select a file
    file_paths = filedialog.askopenfilenames()
    # Create a figure and axis object using subplots
    # Loop through the file paths
    for file_path in file_paths:
      # Read the contents of the file into a pandas DataFrame
      file_p1, file_ext = osys.path.splitext(file_path)
      file_p2, file_name = osys.path.split(file_p1)
      df = pd.read_csv(filepath_or_buffer= file_path,
                         sep= "\t",
                         decimal=".",
                         skiprows=17,
                         skip_blank_lines=True,
                         skipfooter=2,
                         names=['wl','A'],
                         engine="python")
      # Plot the contents of the DataFrame
      df.index=df.wl
      raw_spec[file_name]=df
    listmax=[]
    listmin=[]
    for i in raw_spec :
        a=raw_spec[i].A[raw_spec[i].wl.between(200,300)].max()
        if not mth.isinf(a) | mth.isnan(a):
            listmax.append(a)
        a=raw_spec[i].A[raw_spec[i].wl.between(750,875)].min()
        if not mth.isinf(a) | mth.isnan(a):
            listmin.append(a)
    globmax=max(listmax)
    globmin=min(listmin)
    # create the fig and axis objects
    fig, ax = plt.subplots()
    ax.set_xlabel('Wavelength [nm]', fontsize=10)
    ax.xaxis.set_label_coords(x=0.5, y=-0.08)
    ax.set_ylabel('Absorbance [AU]', fontsize=10)
    ax.yaxis.set_label_coords(x=-0.1, y=0.5)
    palette=sns.color_palette(palette='bright', n_colors=len(raw_spec))
    n=0
    for i in raw_spec :
        ax.plot(raw_spec[i].wl,
                raw_spec[i].A ,
                linewidth=1,
                label=i+
                      "Max abs peak ="+
                      format(raw_spec[i][raw_spec[i].wl.between(scaling_top-10,scaling_top+10)].A.idxmax(), '.2f'),
                color=palette[n])
        n=n+1
    ax.set_title('raw spectra', fontsize=10, fontweight='bold')
    ax.set_xlim([200, 875])
    ax.set_ylim([globmin-0.1, globmax+0.2])
    ax.tick_params(labelsize=10)
    ax.yaxis.set_ticks(np.arange(int(10*globmin-1)/10, int(10*globmax+1)/10, 0.1))
    plt.legend(loc='upper right', shadow=True, prop={'size':7})
    # Plot the contents of the DataFrame
    plt.plot(ax=ax)
    # Display the plot in a matplotlib canvas
    canvas = FigureCanvasTkAgg(plt.gcf(), master=secondframe)
    canvas.draw()
    canvas.get_tk_widget().grid(row=m, column=0)
    m+=1
    set_scrollbar()

def constant_baseline_correction():
    """
    This function applies a constant baseline correction to the spectra in the `raw_spec` dictionary. The user is prompted to enter a blue and red wavelength value to define the baseline segment, as well as a scaling_top value used to rescale the spectra after correction.
    The function uses the following libraries: 
    - math
    - numpy
    - matplotlib
    - seaborn
    The function takes no arguments, but uses global variables `m`, `raw_spec`, `constant_spec`, `input_field1`, `input_field2`, `input_field3`, `constant_label` 
    """
    # m is the max number of rows of the plotting frame
    global m
    # Get the value from the input field
    baseline_blue = float(input_field1.get())
    baseline_red = float(input_field2.get())
    scaling_top = float(input_field3.get())
      # Do something with the value
      # result = input_value * 2
      # Update the label to warn the user of the correction calculation
    constant_label.config(text="Correcting spectra for constant baseline from " + str(baseline_blue) + ' to ' + str(baseline_red) + ' nm')
    segmentend=raw_spec[next(iter(raw_spec))].wl.between(baseline_blue,baseline_red, inclusive='both')
    for i in raw_spec:
        tmp=baselineconst(raw_spec[i],segmentend)
        tmp.A=sp.signal.savgol_filter(x=tmp.A.copy(),
                                       window_length=21,
                                       polyorder=3)
        constant_spec[i]=rescale_corrected(tmp, scaling_top-10, scaling_top+10)
        raw_spec[i].index=raw_spec[i].wl
    listmax=[]
    listmin=[]
    for i in constant_spec :
        a=constant_spec[i].A[constant_spec[i].wl.between(300,850)].max()
        if not mth.isinf(a) | mth.isnan(a):
            listmax.append(a)
        a=constant_spec[i].A[constant_spec[i].wl.between(300,850)].min()
        if not mth.isinf(a) | mth.isnan(a):
            listmin.append(a)
    globmax=max(listmax)
    globmin=min(listmin)
    # create the fig and axis objects
    fig, ax = plt.subplots()
    ax.set_xlabel('Wavelength [nm]', fontsize=10)
    ax.xaxis.set_label_coords(x=0.5, y=-0.08)
    ax.set_ylabel('Absorbance [AU]', fontsize=10)
    ax.yaxis.set_label_coords(x=-0.1, y=0.5)
    palette=sns.color_palette(palette='bright', n_colors=len(constant_spec))
    n=0
    for i in constant_spec :
        ax.plot(constant_spec[i].wl,
                constant_spec[i].A ,
                linewidth=1,
                label=i+"Max abs peak ="+format(constant_spec[i][constant_spec[i].wl.between(scaling_top-10,scaling_top+10)].A.idxmax(), '.2f'),
                color=palette[n])
        n=n+1
    ax.set_title('only scaled in crystallo absorbance spectra (no scattering correction)', fontsize=10, fontweight='bold')
    ax.set_xlim([200, 875])
    ax.set_ylim([globmin-0.1, globmax+0.2])
    ax.tick_params(labelsize=10)
    ax.yaxis.set_ticks(np.arange(int(10*globmin-1)/10, int(10*globmax+1)/10, 0.1))
    ax.legend(loc='upper right', shadow=True, prop={'size':7})
    # Plot the contents of the DataFrame
    plt.plot(ax=ax)
    # Display the plot in a matplotlib canvas
    canvas = FigureCanvasTkAgg(plt.gcf(), master=secondframe)
    canvas.draw()
    canvas.get_tk_widget().grid(row=m, column=0)
    m+=1
    set_scrollbar()

def scattering_correction():
    """
    This function applies a scattering correction to the spectra in the `raw_spec` dictionary. The user is prompted to enter a blue and red wavelength value to define the baseline segment, a scaling_top value used to rescale the spectra after correction, a nopeak_blue and nopeak_red value used to define the peakless segment where the middle of the scattering baseline is fitted. The leftmost segment is always taken automatically.
    The function uses the following libraries: 
    - math
    - numpy
    - matplotlib
    - seaborn
    The function takes no arguments, but uses global variables `m`, `raw_spec`, `constant_spec`, `input_field1`, `input_field2`, `input_field3`, `input_field4`, `input_field5`
    """
    global m
    baseline_blue = float(input_field1.get())
    baseline_red = float(input_field2.get())
    scaling_top = float(input_field3.get())
    nopeak_blue = float(input_field4.get())
    nopeak_red = float(input_field5.get())
    # baseline segment
    segmentend = raw_spec[next(iter(raw_spec))].wl.between(baseline_blue,baseline_red, inclusive='both')
    # near UV segment
    rightborn=raw_spec[next(iter(raw_spec))].A[raw_spec[next(iter(raw_spec))].wl.between(200,250)].idxmax()+20
    leftborn=raw_spec[next(iter(raw_spec))].A[raw_spec[next(iter(raw_spec))].wl.between(200,250)].idxmax()
    segment1 = raw_spec[next(iter(raw_spec))].wl.between(nopeak_blue,nopeak_red, inclusive='both')
    segment2 = raw_spec[next(iter(raw_spec))].wl.between(leftborn,rightborn, inclusive='both')
    #peakless visible segment
    sigmafor3segment=[1,1,1]
    n=0
    # this plots each fitted baseline against the raw data, highlighting the chose segments
    for i in raw_spec :
        vars()['fig' + str(n)], vars()['ax' + str(n)] = plt.subplots()
        tmp=raw_spec[i].copy()
        tmp.A=sp.signal.savgol_filter(x=tmp.A.copy(),
                                       window_length=21,
                                       polyorder=3)
        tmp, baseline=baselinefitcorr_3seg_smooth(tmp,  segment1, segment2, segmentend, sigmafor3segment)
        constant_spec[i]=rescale_corrected(tmp, scaling_top-30, scaling_top+30)
        vars()['ax' + str(n)].plot(raw_spec[i].wl,raw_spec[i].A)
        vars()['ax' + str(n)].plot(baseline.wl,baseline.A)
        vars()['ax' + str(n)].plot(raw_spec[i].wl[segment1], raw_spec[i].A[segment1])
        vars()['ax' + str(n)].plot(raw_spec[i].wl[segment2], raw_spec[i].A[segment2])
        vars()['ax' + str(n)].plot(raw_spec[i].wl[segmentend], raw_spec[i].A[segmentend])
        # this defines a canvas for each and plots them on the same line using tk.grid
        plt.plot(ax=vars()['ax' + str(n)])
        vars()['canvas' + str(n)] = FigureCanvasTkAgg(plt.gcf(), master=secondframe)
        vars()['canvas' + str(n)].draw()
        vars()['canvas' + str(n)].get_tk_widget().grid(row=m, column=n)
        ready_spec[i]=tmp
        n+=1
    # This segment plots the corrected spectra together at the end of the current line
    listmax=[]
    listmin=[]
    for i in ready_spec :
        a=ready_spec[i].A[ready_spec[i].wl.between(300,850)].max()
        if not (mth.isinf(a) | mth.isnan(a)):
            listmax.append(a)
        a=ready_spec[i].A[ready_spec[i].wl.between(300,850)].min()
        if not (mth.isinf(a) | mth.isnan(a)):
            listmin.append(a)
    globmax=max(listmax)
    globmin=min(listmin)
    fig, ax = plt.subplots()     
    ax.set_xlabel('Wavelength [nm]', fontsize=10)  
    ax.xaxis.set_label_coords(x=0.5, y=-0.08)      
    ax.set_ylabel('Absorbance [AU]', fontsize=10)               
    ax.yaxis.set_label_coords(x=-0.1, y=0.5)       
    palette=sns.color_palette(palette='bright', n_colors=len(ready_spec))   
    n=0                                            
    for i in ready_spec :                          
        ax.plot(ready_spec[i].wl,                  
                  ready_spec[i].A ,                   
                  linewidth=1,                    
                   
                  label=i +" top = " +format(ready_spec[i][ready_spec[i].wl.between(320,850)].A.idxmax(), '.2f'), 
                  color=palette[n])               
        n=n+1
    ax.set_title('scattering corrected in crystallo absorbance spectra', fontsize=10, fontweight='bold')  
    ax.set_xlim([250, 800])
    ax.set_ylim([globmin-0.05, globmax+0.1])
    ax.tick_params(labelsize=8)
    ax.legend(loc='upper right', shadow=True, prop={'size':8})
    plt.plot(ax=ax)
    canvas = FigureCanvasTkAgg(plt.gcf(), master=secondframe)
    canvas.draw()
    canvas.get_tk_widget().grid(row=m,column = n)
    m+=1
    set_scrollbar()

def save_plot():
    """
    This function saves the current plot displayed in the matplotlib canvas in three different formats:
    svg, png and csv. The user can choose the location and the name of the file using the filedialog.asksaveasfilename function.
    The csv file will contain the raw spectra data with columns 'wl' and 'A' dropped.
    """
    image_name = filedialog.asksaveasfilename(defaultextension=".svg")
    plt.savefig(image_name, dpi=1000, transparent=True,bbox_inches='tight')
    image_namepng=image_name[0:-4]+'.png'
    plt.savefig(image_namepng, dpi=1000, transparent=True,bbox_inches='tight')
    towrite_raw_spectra=raw_spec[next(iter(raw_spec))].drop(columns=['wl','A'])
    for spec in raw_spec:
        towrite_raw_spectra[spec]=raw_spec[spec].A
    towrite_raw_spectra.to_csv(image_name[0:-4] + ".csv", index=True)
    if len(constant_spec)==len(raw_spec):
        towrite_constant_spectra=raw_spec[next(iter(constant_spec))].drop(columns=['wl','A'])
        for spec in constant_spec:
            towrite_constant_spectra[spec]=constant_spec[spec].A
        towrite_raw_spectra.to_csv(image_name[0:-4] + ".csv", index=True)
    if len(ready_spec)==len(raw_spec):
        towrite_ready_spectra=raw_spec[next(iter(ready_spec))].drop(columns=['wl','A'])
        for spec in raw_spec:
            towrite_ready_spectra[spec]=raw_spec[spec].A
        towrite_raw_spectra.to_csv(image_name[0:-4] + ".csv", index=True)

def close_plot():
    """
    This function closes the current plot displayed in the matplotlib canvas by destroying the last child (which should be the matplotlib canvas) of the main root.
    """
    # Get the list of all children of the main root
    children = secondframe.winfo_children()
    # Get the last child in the list (which should be the matplotlib canvas)
    last_child = children[-1]
    # Destroy the last child
    last_child.destroy()

# Create the main root
# Create a function that will be called when the "Open" button is clicked
global baseline_blue
global baseline_red
global scaling_top
global nopeak_blue
global nopeak_red
global raw_spec
global fig
global ax
global m
raw_spec={}
constant_spec={}
ready_spec={}
fig, ax = plt.subplots()
baseline_blue=600
baseline_red=800
nopeak_blue=300
nopeak_red=320
scaling_top=500
m = 0
root = Tk()
# root.iconbitmap('the scaled down icOS lab picture')
root.title("icOS toolbox")
# root.geometry("800x800")
#frame containing buttons and entry fields
butframe = Frame(root)
butframe.pack(side = TOP, fill=X)
#frame for plotting
plotframe = Frame(root)
plotframe.pack(side = BOTTOM, fill = BOTH, expand = 1)
# create a global canvas in the second frame to implement the scrollbar
globcanvas=Canvas(plotframe)
globcanvas.configure(scrollregion = globcanvas.bbox("all"))
globcanvas.pack(side=LEFT, fill=BOTH ,expand=1) #slide to 0 if it's a problem
# add scrollbar to canvas
scrollbar=ttk.Scrollbar(plotframe, orient=VERTICAL,command=globcanvas.yview)
scrollbar.pack(side=RIGHT,fill=Y)
h_scrollbar=ttk.Scrollbar(plotframe, orient=HORIZONTAL,command=globcanvas.xview)
h_scrollbar.pack(side=BOTTOM,fill=X,expand=1)
# configure the canvas
globcanvas.configure(yscrollcommand=scrollbar.set)
globcanvas.configure(xscrollcommand=h_scrollbar.set)
globcanvas.bind('<Configure>', lambda e: globcanvas.configure(scrollregion = globcanvas.bbox("all")))
globcanvas.bind('<Configure>', lambda e: globcanvas.configure(scrollregion = globcanvas.bbox("all")))
# create an other frame in the canvas
secondframe=Frame(globcanvas)#, fill=BOTH, expand = 1)
# add that frame to a window in the canvas with a slight offset
globcanvas.create_window((0,0), window=secondframe, anchor = 'nw')#, width=600,height=400)
# Create an "Open" button
open_button = Button(butframe,text="Open new file", command=open_file)
open_button.pack()
# Create a "Save" button
save_button = Button(butframe,text="Save last plot", command=save_plot)
save_button.pack()
# Create a "Close" button
save_button = Button(butframe,text="Close last plot", command=close_plot)
save_button.pack()
# Create a constant_corr_button to trigger the function
constant_corr_button = Button(butframe, text="Correct for constant baseline", command=constant_baseline_correction)
# Create labels and input fields for the constant baseline correction
input_label1 = Label(butframe, text="blue-side boundary to the baseline:")
input_field1 = Entry(butframe)
input_label2 = Label(butframe, text="red-side boundary to the baseline:")
input_field2 = Entry(butframe)
input_label3 = Label(butframe, text="top of the peak used for scaling:")
input_field3 = Entry(butframe)
# Create a label to display the result
constant_label = Label(butframe, text="Constant correction:")
# Create a scattering_corr_button to trigger the function
scatt_button = Button(butframe, text="Correct for scattering", command=scattering_correction)
# Create a label for the scattering correction
scatt_label = Label(butframe, text="Scattering correction correction:")
input_label4 = Label(butframe, text="blue-side of a peakless segment:")
input_field4 = Entry(butframe)
input_label5 = Label(butframe, text="red-side of a peakless segment:")
input_field5 = Entry(butframe)
# Create labels and input fields for the scattering baseline correction
# Place the widgets in the root
constant_label.pack()
constant_corr_button.pack()
input_label1.pack()
input_field1.pack()
input_label2.pack()
input_field2.pack()
input_label3.pack()
input_field3.pack()
scatt_label.pack()
scatt_button.pack()
input_label4.pack()
input_field4.pack()
input_label5.pack()
input_field5.pack()
#button to try and make the scrollbar appear
scrollbut = Button(butframe, text = 'scroll', command = set_scrollbar )
scrollbut.pack()
# Create a button to close the root
close_button = Button(butframe,text="Close main window", command=root.destroy)
close_button.pack()
# scrollbar.pack(side="right", fill="y")
# Run the main loop
root.mainloop()
