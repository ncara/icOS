# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 15:07:56 2023

@author: NCARAMEL
"""

import wx
import pandas as pd
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas
from statistics import mean
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from matplotlib.backends.backend_wx import NavigationToolbar2Wx
import matplotlib
matplotlib.use('wxAgg')


import math as mth
import seaborn as sns
import numpy as np
import scipy as sp
from scipy import signal 
if 'app' in vars():
    del app

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

class GenPanel(wx.Panel):
    raw_spec = {}
    const_spec = {}
    ready_spec = {}

class RightPanel(GenPanel):
    def __init__(self, parent):
        wx.Panel.__init__(self, parent, style = wx.FULL_REPAINT_ON_RESIZE | wx.SUNKEN_BORDER)
        self.figure = plt.figure()
        self.canvas = FigureCanvas(self, -1, self.figure)
        self.toolbar = NavigationToolbar2Wx(self.canvas)
        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(self.canvas, proportion=1, flag=wx.EXPAND)
        sizer.Add(self.toolbar, proportion=0, flag=wx.EXPAND)
        self.SetSizer(sizer)
        
    def plot_data(self,typlot,scaling_top):
        self.figure.clear()
        ax = self.figure.add_subplot()
        if typlot == 'raw':
            print('plotting raw data')
            listmax=[]
            listmin=[]
            for i in GenPanel.raw_spec :
                a=GenPanel.raw_spec[i].A[GenPanel.raw_spec[i].wl.between(270,850)].max()
                if not (mth.isinf(a) | mth.isnan(a)):
                    listmax.append(a)
                a=GenPanel.raw_spec[i].A[GenPanel.raw_spec[i].wl.between(270,850)].min()
                if not (mth.isinf(a) | mth.isnan(a)):
                    listmin.append(a)
            # for i in GenPanel.raw_spec :
            #     ax.plot(GenPanel.raw_spec[i].wl, GenPanel.raw_spec[i].A)
            globmax=max(listmax)
            globmin=min(listmin)
            ax.set_xlabel('Wavelength [nm]', fontsize=10)  
            ax.xaxis.set_label_coords(x=0.5, y=-0.08)      
            ax.set_ylabel('Absorbance [AU]', fontsize=10)               
            ax.yaxis.set_label_coords(x=-0.1, y=0.5)       
            palette=sns.color_palette(palette='bright', n_colors=len(GenPanel.raw_spec))   
            n=0                                            
            for i in GenPanel.raw_spec :                          
                ax.plot(GenPanel.raw_spec[i].wl,                  
                        GenPanel.raw_spec[i].A ,                   
                        linewidth=1,                    
                       
                        label=i +" top = " +format(GenPanel.raw_spec[i][GenPanel.raw_spec[i].wl.between(scaling_top-10,scaling_top+10)].A.idxmax(), '.3f'), 
                        color=palette[n])               
                n=n+1
            ax.set_title('raw in crystallo absorbance spectra', fontsize=10, fontweight='bold')  
            ax.set_xlim([250, 800])
            ax.set_ylim([globmin-0.05, globmax+0.1])
            ax.tick_params(labelsize=8)
            ax.legend(loc='upper right', shadow=True, prop={'size':8})
            self.canvas.draw()
        
        elif typlot == 'const':
            print('plotting constant corrected data')
            listmax=[]
            listmin=[]
            for i in GenPanel.const_spec :
                a=GenPanel.const_spec[i].A[GenPanel.const_spec[i].wl.between(200,300)].max()
                if not mth.isinf(a) | mth.isnan(a):
                    listmax.append(a)
                a=GenPanel.const_spec[i].A[GenPanel.const_spec[i].wl.between(750,875)].min()
                if not mth.isinf(a) | mth.isnan(a):
                    listmin.append(a)
            globmax=max(listmax)
            globmin=min(listmin)
            # create the fig and axis objects
            # fig, ax = plt.subplots()
            ax.set_xlabel('Wavelength [nm]', fontsize=10)
            ax.xaxis.set_label_coords(x=0.5, y=-0.08)
            ax.set_ylabel('Absorbance [AU]', fontsize=10)
            ax.yaxis.set_label_coords(x=-0.1, y=0.5)
            palette=sns.color_palette(palette='bright', n_colors=len(GenPanel.const_spec))
            n=0
            for i in GenPanel.const_spec :
                ax.plot(GenPanel.const_spec[i].wl,
                        GenPanel.const_spec[i].A ,
                        linewidth=1,
                        label=i+"Max abs peak ="+format(GenPanel.const_spec[i][GenPanel.const_spec[i].wl.between(scaling_top-10,scaling_top+10)].A.idxmax(), '.2f'),
                        color=palette[n])
                n=n+1
            ax.set_title('only scaled in crystallo absorbance spectra (no scattering correction)', fontsize=10, fontweight='bold')
            ax.set_xlim([200, 875])
            ax.set_ylim([globmin-0.1, globmax+0.2])
            ax.tick_params(labelsize=10)
            ax.yaxis.set_ticks(np.arange(int(10*globmin-1)/10, int(10*globmax+1)/10, 0.1))
            ax.legend(loc='upper right', shadow=True, prop={'size':7})
            self.canvas.draw()
        elif typlot == 'ready':
            print('plotting scattering corrected spectra')
            listmax=[]
            listmin=[]
            for i in GenPanel.ready_spec :
                a=GenPanel.ready_spec[i].A[GenPanel.ready_spec[i].wl.between(300,500)].max()
                if not (mth.isinf(a) | mth.isnan(a)):
                    listmax.append(a)
                a=GenPanel.ready_spec[i].A[GenPanel.ready_spec[i].wl.between(300,600)].min()
                if not (mth.isinf(a) | mth.isnan(a)):
                    listmin.append(a)
            globmax=max(listmax)
            globmin=min(listmin) 
            ax.set_xlabel('Wavelength [nm]', fontsize=10)  
            ax.xaxis.set_label_coords(x=0.5, y=-0.08)      
            ax.set_ylabel('Absorbance [AU]', fontsize=10)               
            ax.yaxis.set_label_coords(x=-0.1, y=0.5)       
            palette=sns.color_palette(palette='bright', n_colors=len(GenPanel.ready_spec))   
            n=0                                            
            for i in GenPanel.ready_spec :                          
                ax.plot(GenPanel.ready_spec[i].wl,                  
                          GenPanel.ready_spec[i].A ,                   
                          linewidth=1,                    
                           
                          label=i +" top = " +format(GenPanel.ready_spec[i][GenPanel.ready_spec[i].wl.between(scaling_top-10,scaling_top+10)].A.idxmax(), '.2f'), 
                          color=palette[n])               
                n=n+1
            ax.set_title('scattering corrected in crystallo absorbance spectra', fontsize=10, fontweight='bold')  
            ax.set_xlim([250, 800])
            ax.set_ylim([globmin-0.05, globmax+0.1])
            ax.tick_params(labelsize=8)
            ax.legend(loc='upper right', shadow=True, prop={'size':8})
            self.canvas.draw()
            
    
class LeftPanel(GenPanel):
    
    def __init__(self, parent):
        wx.Panel.__init__(self, parent, style = wx.SUNKEN_BORDER)
        self.button_openfile = wx.Button(self, label="Open File")
        self.button_openfile.Bind(wx.EVT_BUTTON, self.on_open_file)
        # constant baseline correction 
        self.StaticBox_const = wx.StaticBox(self, label = "Constant Baseline")
        constboxsizer = wx.StaticBoxSizer(self.StaticBox_const, wx.VERTICAL)
        self.label_topeak = wx.StaticText(self, label="wl of the peak of interest", style = wx.ALIGN_CENTER_HORIZONTAL)
        self.field_topeak = wx.TextCtrl(self, style = wx.TE_CENTER)
        self.button_constancorr = wx.Button(self, label="Correct for constant baseline")
        self.button_constancorr.Bind(wx.EVT_BUTTON, self.on_constant_corr)
        self.label_baseline_blue = wx.StaticText(self, label="Baseline blue-side Boundary", style = wx.ALIGN_CENTER_HORIZONTAL)
        self.field_baseline_blue = wx.TextCtrl(self, style = wx.TE_CENTER)
        self.label_baseline_red = wx.StaticText(self, label="Baseline red-side Boundary", style = wx.ALIGN_CENTER_HORIZONTAL)
        self.field_baseline_red = wx.TextCtrl(self, style = wx.TE_CENTER)
        constboxsizer.Add(self.field_topeak, 1, wx.ALIGN_CENTER | wx.ALL, border = 2)
        constboxsizer.Add(self.label_topeak, 1, wx.ALIGN_CENTER, border = 0)
        constboxsizer.Add(self.field_baseline_blue, 1, wx.ALIGN_CENTER | wx.ALL, border = 2)
        constboxsizer.Add(self.label_baseline_blue, 1, wx.ALIGN_CENTER, border = 0)
        constboxsizer.Add(self.field_baseline_red, 1, wx.ALIGN_CENTER | wx.ALL, border = 2)
        constboxsizer.Add(self.label_baseline_red, 1, wx.ALIGN_CENTER, border  = 0)
        constboxsizer.Add(self.button_constancorr, 1, wx.EXPAND | wx.ALL, border = 2)
        
        #Scattering correction 
        self.StaticBox_scat = wx.StaticBox(self, label = "Scattering Baseline")
        scatboxsizer = wx.StaticBoxSizer(self.StaticBox_scat, wx.VERTICAL)
        self.button_scattercor = wx.Button(self, label="Correct for Scattering")
        self.button_scattercor.Bind(wx.EVT_BUTTON, self.on_scat_corr)
        self.label_nopeak_blue = wx.StaticText(self, label="blue-side boundary of the peakless segment", style = wx.ALIGN_CENTER_HORIZONTAL)
        self.field_nopeak_blue = wx.TextCtrl(self, style = wx.TE_CENTER)
        self.label_nopeak_red = wx.StaticText(self, label="red-side boundary of the peakless segment", style = wx.ALIGN_CENTER_HORIZONTAL)
        self.field_nopeak_red = wx.TextCtrl(self, style = wx.TE_CENTER)
        scatboxsizer.Add(self.field_nopeak_blue, 1, wx.ALIGN_CENTER | wx.ALL, border = 2)
        scatboxsizer.Add(self.label_nopeak_blue, 1, wx.ALIGN_CENTER, border = 0)
        scatboxsizer.Add(self.field_nopeak_red, 1, wx.ALIGN_CENTER | wx.ALL, border = 2)
        scatboxsizer.Add(self.label_nopeak_red, 1, wx.ALIGN_CENTER, border = 0)
        scatboxsizer.Add(self.button_scattercor, 1, wx.EXPAND | wx.ALL, border = 2)
        
        self.button_save = wx.Button(self, label="Save figure and spectra")
        self.button_save.Bind(wx.EVT_BUTTON, self.on_save)
        
        # Add widgets to the right panel sizer
        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(self.button_openfile, 1, wx.EXPAND | wx.ALL, border = 2)

        sizer.Add(constboxsizer, 1, wx.EXPAND, border = 5)
        
        sizer.Add(scatboxsizer, 1, wx.EXPAND, border = 5)

        sizer.Add(self.button_save, 1, wx.EXPAND | wx.ALL, border = 2)
        self.SetSizer(sizer)
        # self.SetBackgroundColour('grey') 
        
    def on_open_file(self, event):
        wildcard = "TXT files (*.txt)|*.txt|All files (*.*)|*.*"
        dialog = wx.FileDialog(self, "Choose one or several files", wildcard=wildcard, style=wx.FD_OPEN | wx.FD_MULTIPLE)
        if dialog.ShowModal() == wx.ID_OK:
            file_paths = dialog.GetPaths()
            for file_path in file_paths:
                file_name = file_path.split('\\')[-1][0:-4]
                print(file_name)
                GenPanel.raw_spec[file_name] = pd.read_csv(filepath_or_buffer= file_path,
                              sep= "\t",
                              decimal=".",
                              skiprows=17,
                              skip_blank_lines=True,
                              skipfooter=2,
                              names=['wl','A'],
                              engine="python")
                GenPanel.raw_spec[file_name].index=GenPanel.raw_spec[file_name].wl
                print(f"File '{file_name}' added to dictionary with data: {GenPanel.raw_spec[file_name].A}")
            self.update_right_panel('raw')
        dialog.Destroy()
        # Plot the DataFrame
    def on_constant_corr(self, event):
        baseline_blue = float(self.field_baseline_blue.GetValue())
        baseline_red = float(self.field_baseline_red.GetValue())
        scaling_top = float(self.field_topeak.GetValue())
        segmentend=GenPanel.raw_spec[next(iter(GenPanel.raw_spec))].wl.between(baseline_blue,baseline_red, inclusive='both')
        for i in GenPanel.raw_spec:
            tmp=GenPanel.raw_spec[i].copy()
            tmp.A-=mean(GenPanel.raw_spec[i].A[segmentend])
            tmp.A*=1/tmp.A[tmp.wl.between(scaling_top-10,scaling_top+10,inclusive='both')].max()
            tmp.A=sp.signal.savgol_filter(x=tmp.A.copy(),     #This is the smoothing function, it takes in imput the y-axis data directly and fits a polynom on each section of the data at a time
                               window_length=21,  #This defines the section, longer sections means smoother data but also bigger imprecision
                               polyorder=3)       #The order of the polynom, more degree = less smooth, more precise (and more ressource expensive)
            GenPanel.const_spec[i]=tmp.copy()
            GenPanel.const_spec[i].index=GenPanel.raw_spec[i].wl
            print(f"Spectrum '{i}' corrected: {GenPanel.const_spec[i].A}")
        self.update_right_panel('const')
        
    def on_scat_corr(self, event):
        baseline_blue = float(self.field_baseline_blue.GetValue())
        baseline_red = float(self.field_baseline_red.GetValue())
        scaling_top = float(self.field_topeak.GetValue())
        nopeak_blue = float(self.field_nopeak_blue.GetValue())
        nopeak_red = float(self.field_nopeak_red.GetValue())
        rightborn=GenPanel.raw_spec[next(iter(GenPanel.raw_spec))].A[GenPanel.raw_spec[next(iter(GenPanel.raw_spec))].wl.between(200,250)].idxmax()+20
        leftborn=GenPanel.raw_spec[next(iter(GenPanel.raw_spec))].A[GenPanel.raw_spec[next(iter(GenPanel.raw_spec))].wl.between(200,250)].idxmax()
        segment1 = GenPanel.raw_spec[next(iter(GenPanel.raw_spec))].wl.between(nopeak_blue,nopeak_red, inclusive='both')
        segment2 = GenPanel.raw_spec[next(iter(GenPanel.raw_spec))].wl.between(leftborn,rightborn, inclusive='both')
        segmentend=GenPanel.raw_spec[next(iter(GenPanel.raw_spec))].wl.between(baseline_blue,baseline_red, inclusive='both')
        segment=segment1+segment2+segmentend
        #peakless visible segment
        sigmafor3segment=[1,1,1]
        n=0
        # this plots each fitted baseline against the raw data, highlighting the chose segments
        for i in GenPanel.raw_spec :
            tmp=GenPanel.raw_spec[i].copy()
            tmp.A=sp.signal.savgol_filter(x=tmp.A.copy(),
                                          window_length=21,
                                          polyorder=3)
            x=tmp.wl[segment].copy()
            y=tmp.A[segment].copy()
            initialParameters = np.array([1e9,1])
            n=len(tmp.A[segment1])
            sigma=n*[sigmafor3segment[0]]
            n=len(tmp.A[segment2])
            sigma=sigma + n*[sigmafor3segment[1]]
            n=len(tmp.A[segmentend])
            sigma=sigma + n*[sigmafor3segment[2]]
            para, pcov = sp.optimize.curve_fit(f=fct_baseline, xdata=x, ydata=y, p0=initialParameters, sigma=sigma)
            baseline=tmp.copy()
            baseline.A=fct_baseline(baseline.wl.copy(), *para)
            corrected=tmp.copy()
            corrected.A=tmp.A.copy()-baseline.A
            corrected.A*=1/corrected.A[corrected.wl.between(scaling_top-10,scaling_top+10,inclusive='both')].max()
            GenPanel.ready_spec[i]=corrected
            # tmp, baseline=baselinefitcorr_3seg_smooth(tmp,  segment1, segment2, segmentend, sigmafor3segment)
            vars()['fig' + str(n)], vars()['ax' + str(n)] = plt.subplots()
            vars()['ax' + str(n)].set_title(str(i))
            vars()['ax' + str(n)].plot(GenPanel.raw_spec[i].wl,GenPanel.raw_spec[i].A)
            vars()['ax' + str(n)].plot(baseline.wl,baseline.A)
            vars()['ax' + str(n)].plot(GenPanel.raw_spec[i].wl[segment1], GenPanel.raw_spec[i].A[segment1], color = 'lime')
            vars()['ax' + str(n)].plot(GenPanel.raw_spec[i].wl[segment2], GenPanel.raw_spec[i].A[segment2], color = 'magenta')
            vars()['ax' + str(n)].plot(GenPanel.raw_spec[i].wl[segmentend], GenPanel.raw_spec[i].A[segmentend], color = 'crimson') 
            vars()['fig' + str(n)].show()
            
            # this defines a canvas for each and plots them on the same line using tk.grid
            # plt.plot(ax=vars()['ax' + str(n)])
            # plt.show(vars()['ax' + str(n)])  # find a way to prevent the current plot from showing up with the rest 
            # vars()['canvas' + str(n)] = FigureCanvasTkAgg(plt.gcf(), master=secondframe)
            # vars()['canvas' + str(n)].draw()
            # vars()['canvas' + str(n)].get_tk_widget().grid(row=m, column=n)
            n+=1
        self.update_right_panel('ready')
        
        pass
    def on_save(self, event):
        wildcard = "CSV files (*.csv)|*.csv|All files (*.*)|*.*"
        dialog = wx.FileDialog(self, "Save File(s)", wildcard=wildcard, style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT)
        if dialog.ShowModal() == wx.ID_OK:
            totalpath = dialog.GetPath()
            # file_path2 = file_path.split('\\')[:-1]
            file_path=''
            for i in totalpath.split('\\')[:-1]:
                file_path+=i+'\\'
            print(file_path)
            file_name = totalpath.split('\\')[-1][0:-4]
                
        dialog.Destroy()
        towrite_raw_spectra=GenPanel.raw_spec[next(iter(GenPanel.raw_spec))].drop(columns=['wl','A'])
        for spec in GenPanel.raw_spec:
            towrite_raw_spectra[spec]=GenPanel.raw_spec[spec].A
            print("File" + file_path + f" '{spec}' saved in: raw_{file_name}.csv in column {spec}")
        towrite_raw_spectra.to_csv('raw_' +  file_name + ".csv", index=True)
        if len(GenPanel.const_spec)==len(GenPanel.raw_spec):
            towrite_constant_spectra=GenPanel.raw_spec[next(iter(GenPanel.const_spec))].drop(columns=['wl','A'])
            for spec in GenPanel.const_spec:
                towrite_constant_spectra[spec]=GenPanel.const_spec[spec].A
                print("File" + file_path + f" '{spec}' saved in: constant_{file_name}.csv in column {spec}")
            towrite_raw_spectra.to_csv('constant_' +  file_name + ".csv", index=True)
        if len(GenPanel.ready_spec)==len(GenPanel.raw_spec):
            towrite_ready_spectra=GenPanel.raw_spec[next(iter(GenPanel.ready_spec))].drop(columns=['wl','A'])
            for spec in GenPanel.raw_spec:
                towrite_ready_spectra[spec]=GenPanel.raw_spec[spec].A
                print("File" + file_path + f" '{spec}' saved in: ready_{file_name}.csv in column {spec}")
            towrite_raw_spectra.to_csv(file_path + 'ready_' +  file_name + ".csv", index=True)
        self.GetParent().right_panel.figure.savefig(file_path + file_name + ".svg", dpi=900 , transparent=True,bbox_inches='tight')
        self.GetParent().right_panel.figure.savefig(file_path + file_name + ".png", dpi=900, transparent=True,bbox_inches='tight')
        self.GetParent().right_panel.figure.savefig(file_path + file_name + ".pdf", dpi=900, transparent=True,bbox_inches='tight')
        print("Figure saved at: " + file_path + file_name + '.png')
        
        
    def update_right_panel(self, typlot):
        if len(self.field_topeak.GetValue()) == 0:
            scaling_top=280
        else :
            scaling_top = float(self.field_topeak.GetValue())
        print(scaling_top)
        self.GetParent().right_panel.plot_data(typlot, scaling_top)

class MainFrame(wx.Frame):
    def __init__(self):
        wx.Frame.__init__(self, None, title="icOS toolbox", size = (800,600))
        # self.raw_spec = {}
        # self.constant_spec = {}
        # self.ready_spec = {}
        # self.baseline_blue = 600
        # self.baseline_red = 800
        # self.nopeak_blue = 300
        # self.nopeak_red = 320
        # self.scaling_top = 500

        # Create splitter
        self.splitter = wx.SplitterWindow(self)
        # Create left and right panels
        self.splitter.left_panel = LeftPanel(self.splitter)
        self.splitter.right_panel = RightPanel(self.splitter)
        # Add panels to splitter
        self.splitter.SplitVertically(self.splitter.left_panel, self.splitter.right_panel, 200)
        self.splitter.SetSashGravity(0.5)
        # Set main sizer
        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(self.splitter, 1, wx.EXPAND)
        self.SetSizer(sizer)
        self.Bind(wx.EVT_CLOSE, self.on_close)
        self.Show()
    def on_close(self, event):
        # self.Close()
        self.Destroy()



if __name__ == "__main__":
    app = wx.App()
    frame = MainFrame()
    app.MainLoop()