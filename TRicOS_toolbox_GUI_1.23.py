# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 15:07:56 2023
@author: NCARAMEL
"""

import importlib

#instlaling packages if not present 

try:
    wx = importlib.import_module('wx')
    print("wxPython is already installed.")
except ImportError:
    print("wxPython is not installed. Installing now...")
    # Install SciPy using pip
    try:
        import pip
    except ImportError:
        print("pip is not installed. Please install pip to continue.")
    else:
        pip.main(['install', 'wxPython'])
        print("wxPython has been installed.")
        wx = importlib.import_module('wx')

try:
    pd = importlib.import_module('pandas')
    print("pandas is already installed.")
except ImportError:
    print("pandas is not installed. Installing now...")
    # Install SciPy using pip
    try:
        import pip
    except ImportError:
        print("pip is not installed. Please install pip to continue.")
    else:
        pip.main(['install', 'pandas'])
        print("pandas has been installed.")
        pd = importlib.import_module('pandas')

try:
    import matplotlib
    print("matplotlib is already installed.")
except ImportError:
    print("matplotlib is not installed. Installing now...")
    # Install SciPy using pip
    try:
        import pip
    except ImportError:
        print("pip is not installed. Please install pip to continue.")
    else:
        pip.main(['install', 'matplotlib'])
        print("matplotlib has been installed.")
        import matplotlib


from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas
from statistics import mean
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from matplotlib.backends.backend_wx import NavigationToolbar2Wx
matplotlib.use('wxAgg')
plt.rcParams["figure.figsize"] = (10/2.54,7.5/2.54)
try:
    import platform
    print("platform is already installed.")
except ImportError:
    print("platform is not installed. Installing now...")
    # Install SciPy using pip
    try:
        import pip
    except ImportError:
        print("pip is not installed. Please install pip to continue.")
    else:
        pip.main(['install', 'platform'])
        print("platform has been installed.")
        import platform


try:
    import os
    print("os is already installed.")
except ImportError:
    print("os is not installed. Installing now...")
    # Install SciPy using pip
    try:
        import pip
    except ImportError:
        print("pip is not installed. Please install pip to continue.")
    else:
        pip.main(['install', 'os'])
        print("os has been installed.")
        import os

try:
    pd = importlib.import_module('pandas')
    print("pandas is already installed.")
except ImportError:
    print("pandas is not installed. Installing now...")
    # Install SciPy using pip
    try:
        import pip
    except ImportError:
        print("pip is not installed. Please install pip to continue.")
    else:
        pip.main(['install', 'pandas'])
        print("pandas has been installed.")
        pd = importlib.import_module('pandas')

try:
    mth = importlib.import_module('math')
    print("math is already installed.")
except ImportError:
    print("math is not installed. Installing now...")
    # Install SciPy using pip
    try:
        import pip
    except ImportError:
        print("pip is not installed. Please install pip to continue.")
    else:
        pip.main(['install', 'math'])
        print("math has been installed.")
        mth = importlib.import_module('math')

try:
    sns = importlib.import_module('seaborn')
    print("seaborn is already installed.")
except ImportError:
    print("seaborn is not installed. Installing now...")
    # Install SciPy using pip
    try:
        import pip
    except ImportError:
        print("pip is not installed. Please install pip to continue.")
    else:
        pip.main(['install', 'seaborn'])
        print("seaborn has been installed.")
        sns = importlib.import_module('seaborn')

try:
    np = importlib.import_module('numpy')
    print("numpy is already installed.")
except ImportError:
    print("numpy is not installed. Installing now...")
    # Install SciPy using pip
    try:
        import pip
    except ImportError:
        print("pip is not installed. Please install pip to continue.")
    else:
        pip.main(['install', 'numpy'])
        print("numpy has been installed.")
        np = importlib.import_module('numpy')

try:
    sp = importlib.import_module('scipy')
    print("SciPy is already installed.")
except ImportError:
    print("SciPy is not installed. Installing now...")
    # Install SciPy using pip
    try:
        import pip
    except ImportError:
        print("pip is not installed. Please install pip to continue.")
    else:
        pip.main(['install', 'scipy'])
        print("SciPy has been installed.")
        sp = importlib.import_module('scipy')
from scipy import signal 


if 'app' in vars():
    del app


system = platform.system()
if system == "Windows":
    path = os.path.join("C:\\", "path", "to", "save", "output")
else:
    path = os.path.join("/", "path", "to", "save", "output")

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
def absorbance(tmp):
    ourdata=tmp.copy()
    if 'A' in ourdata.columns :
        return(ourdata.drop(columns=['I','bgd',"I0"]).copy())
        print('avantes absorbance saved for spectrum')
    else :
        ourdata['absor']=None
        for wl in ourdata.index:
            tmpdat=float(ourdata.I[wl] - ourdata.bgd[wl])
            tmpref=float(ourdata.I0[wl] - ourdata.bgd[wl])
            if tmpdat/tmpref < 0:
                tmpabs=0
            else :
                tmpabs=-np.log(tmpdat/tmpref)
            # dfmi.loc['wl'=, 'absor']
            ourdata.loc[wl,'A']=tmpabs
        return(ourdata.drop(columns=['I','bgd',"I0"]).copy())
floatize=np.vectorize(float)   

class GenPanel(wx.Panel):
    raw_lamp={}
    raw_spec = {}
    const_spec = {}
    ready_spec = {}
    diffspec=pd.DataFrame(data=None,columns=['wl','A'])

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
        
    def plot_data(self,typecorr,scaling_top):
        self.figure.clear()
        ax = self.figure.add_subplot()
        if typecorr == 'raw':
            print('plotting raw data')
            listmax=[]
            listmin=[]
            for i in GenPanel.raw_spec :
                a=GenPanel.raw_spec[i].A[GenPanel.raw_spec[i].wl.between(320,800)].max()
                if not (mth.isinf(a) | mth.isnan(a)):
                    listmax.append(a)
                a=GenPanel.raw_spec[i].A[GenPanel.raw_spec[i].wl.between(800,900)].min()
                if not (mth.isinf(a) | mth.isnan(a)):
                    listmin.append(a)
            # for i in GenPanel.raw_spec :
            #     ax.plot(GenPanel.raw_spec[i].wl, GenPanel.raw_spec[i].A)
            globmax=max(listmax)
            globmin=min(listmin)
            ax.set_xlabel('Wavelength [nm]', fontsize=20)  
            ax.xaxis.set_label_coords(x=0.5, y=-0.08)      
            ax.set_ylabel('Absorbance [AU]', fontsize=20)               
            ax.yaxis.set_label_coords(x=-0.1, y=0.5)       
            palette=sns.color_palette(palette='bright', n_colors=len(GenPanel.raw_spec))   
            n=0   
            if self.GetParent().left_panel.mass_center_checkbox.GetValue() :
                centroids = self.GetParent().left_panel.mass_center(typecorr = typecorr)                                        
            for i in GenPanel.raw_spec :                          
                             
                
                if self.GetParent().left_panel.mass_center_checkbox.GetValue() :
                    ax.plot(GenPanel.raw_spec[i].wl,                  
                        GenPanel.raw_spec[i].A ,                   
                        linewidth=2,                    
                       
                        label=i +" mass center = " +format(centroids[i], '.3f'), 
                        color=palette[n]) 
                    ax.axvline(centroids[i], color = palette[n], ls = '-.')
                else:
                                        
                            if self.GetParent().left_panel.scaling_checkbox.GetValue() :
                                ax.plot(GenPanel.raw_spec[i].wl,                  
                                        GenPanel.raw_spec[i].A ,                   
                                        linewidth=2,
                                        label=i +" max abs = " +format(GenPanel.raw_spec[i][GenPanel.raw_spec[i].wl.between(scaling_top-5,scaling_top+5)].A.idxmax(), '.3f'), 
                                        color=palette[n])  
                            else :
                                ax.plot(GenPanel.raw_spec[i].wl,                  
                                        GenPanel.raw_spec[i].A ,                   
                                        linewidth=2,
                                        label=i +" max abs = " +format(GenPanel.raw_spec[i].A.idxmax(), '.3f'), 
                                        color=palette[n])    
                n=n+1
            ax.set_title('raw in crystallo absorbance spectra', fontsize=20, fontweight='bold')  
            ax.set_xlim([250, 1000])
            ax.set_ylim([globmin-0.05, globmax+0.1])
            ax.tick_params(labelsize=16)
            ax.legend(loc='upper right', shadow=True, prop={'size':8})
            self.canvas.draw()
        
        elif typecorr == 'const':
            print('plotting constant corrected data')
            listmax=[]
            listmin=[] 
            for i in GenPanel.const_spec :
                a=GenPanel.const_spec[i].A[GenPanel.const_spec[i].wl.between(320,800)].max()
                if not mth.isinf(a) | mth.isnan(a):
                    listmax.append(a)
                a=GenPanel.const_spec[i].A[GenPanel.const_spec[i].wl.between(800,900)].min()
                if not mth.isinf(a) | mth.isnan(a):
                    listmin.append(a)
            globmax=max(listmax)
            globmin=min(listmin)
            # create the fig and axis objects
            # fig, ax = plt.subplots()
            ax.set_xlabel('Wavelength [nm]', fontsize=20)
            ax.xaxis.set_label_coords(x=0.5, y=-0.08)
            ax.set_ylabel('Absorbance [AU]', fontsize=20)
            ax.yaxis.set_label_coords(x=-0.1, y=0.5)
            palette=sns.color_palette(palette='bright', n_colors=len(GenPanel.const_spec))
            n=0
            if self.GetParent().left_panel.mass_center_checkbox.GetValue() :
                centroids = self.GetParent().left_panel.mass_center(typecorr = typecorr)  
            for i in GenPanel.const_spec :
                
                if self.GetParent().left_panel.mass_center_checkbox.GetValue() :
                    ax.plot(GenPanel.const_spec[i].wl,
                        GenPanel.const_spec[i].A ,
                        linewidth=2,
                        label=i+" mass center = " +format(centroids[i], '.3f'),
                        color=palette[n])
                    ax.axvline(centroids[i], color = palette[n], ls = '-.')
                else :
                    if self.GetParent().left_panel.scaling_checkbox.GetValue() :
                        ax.plot(GenPanel.const_spec[i].wl,
                                GenPanel.const_spec[i].A ,
                                linewidth=2,
                                label=i+"Max abs peak ="+format(GenPanel.const_spec[i][GenPanel.const_spec[i].wl.between(scaling_top-10,scaling_top+10)].A.idxmax(), '.2f'),
                                color=palette[n])
                    else :
                        ax.plot(GenPanel.const_spec[i].wl,
                                GenPanel.const_spec[i].A ,
                                linewidth=2,
                                label=i+"Max abs peak ="+format(GenPanel.const_spec[i].A.idxmax(), '.2f'),
                                color=palette[n])
                n=n+1
            ax.set_title('only scaled in crystallo absorbance spectra (no scattering correction)', fontsize=20, fontweight='bold')
            ax.set_xlim([200, 1000])
            ax.set_ylim([globmin-0.1, globmax+0.2])
            ax.tick_params(labelsize=10)
            ax.yaxis.set_ticks(np.arange(int(10*globmin-1)/10, int(10*globmax+1)/10, 0.1))
            ax.legend(loc='upper right', shadow=True, prop={'size':7})
            self.canvas.draw()
        elif typecorr == 'ready':
            print('plotting scattering corrected spectra')
            listmax=[]
            listmin=[]
            for i in GenPanel.ready_spec :
                a=GenPanel.ready_spec[i].A[GenPanel.ready_spec[i].wl.between(320,800)].max()
                if not (mth.isinf(a) | mth.isnan(a)):
                    listmax.append(a)
                a=GenPanel.ready_spec[i].A[GenPanel.ready_spec[i].wl.between(800,900)].min()
                if not (mth.isinf(a) | mth.isnan(a)):
                    listmin.append(a)
            globmax=max(listmax)
            globmin=min(listmin) 
            ax.set_xlabel('Wavelength [nm]', fontsize=20)  
            ax.xaxis.set_label_coords(x=0.5, y=-0.08)      
            ax.set_ylabel('Absorbance [AU]', fontsize=20)               
            ax.yaxis.set_label_coords(x=-0.1, y=0.5)       
            palette=sns.color_palette(palette='bright', n_colors=len(GenPanel.ready_spec))   
            n=0  
            if self.GetParent().left_panel.mass_center_checkbox.GetValue() :
                centroids = self.GetParent().left_panel.mass_center(typecorr = typecorr)                                            
            for i in GenPanel.ready_spec :
                if self.GetParent().left_panel.mass_center_checkbox.GetValue() :
                    ax.plot(GenPanel.ready_spec[i].wl,                  
                          GenPanel.ready_spec[i].A ,                   
                          linewidth=2,
                          label=i +" mass center = " +format(centroids[i], '.3f'), 
                          color=palette[n]) 
                    ax.axvline(centroids[i], color = palette[n], ls = '-.')
                else :
                    if self.GetParent().left_panel.scaling_checkbox.GetValue() :
                        ax.plot(GenPanel.ready_spec[i].wl,                  
                                GenPanel.ready_spec[i].A ,                   
                                linewidth=2,
                                label=i +" mass center = " +format(GenPanel.ready_spec[i][GenPanel.ready_spec[i].wl.between(scaling_top-10,scaling_top+10)].A.idxmax(), '.2f'), 
                                color=palette[n]) 
                    else :
                        ax.plot(GenPanel.ready_spec[i].wl,                  
                                GenPanel.ready_spec[i].A ,                   
                                linewidth=2,
                                label=i +" mass center = " +format(GenPanel.ready_spec[i].A.idxmax(), '.2f'), 
                                color=palette[n]) 
                n=n+1
            ax.set_title('scattering corrected in crystallo absorbance spectra', fontsize=20, fontweight='bold')  
            ax.set_xlim([250, 1000])
            ax.set_ylim([globmin-0.05, globmax+0.1])
            ax.tick_params(labelsize=16)
            ax.legend(loc='upper right', shadow=True, prop={'size':8})
            self.canvas.draw()
        elif typecorr == 'diff':            
            ax.set_xlabel('Wavelength [nm]', fontsize=20)  
            ax.xaxis.set_label_coords(x=0.5, y=-0.08)      
            ax.set_ylabel('Absorbance [AU]', fontsize=20)               
            ax.yaxis.set_label_coords(x=-0.1, y=0.5)       
            palette=sns.color_palette(palette='bright', n_colors=1)#len(GenPanel.diffspec)) 
            n=0
            if self.GetParent().left_panel.mass_center_checkbox.GetValue() :
                ax.plot(GenPanel.diffspec.wl,                  
                      GenPanel.diffspec.A ,                   
                      linewidth=2,
                      label=i +" mass center = " +format(centroids[i], '.3f'), 
                      color=palette[n]) 
                ax.axvline(centroids[i], color = palette[n], ls = '-.')
            else :
                if self.GetParent().left_panel.scaling_checkbox.GetValue() :
                    ax.plot(GenPanel.diffspec.wl,                  
                            GenPanel.diffspec.A ,                   
                            linewidth=2,
                            label=" mass center = " +format(GenPanel.diffspec[GenPanel.diffspec.wl.between(scaling_top-10,scaling_top+10)].A.idxmax(), '.2f'), 
                            color=palette[n]) 
                    
            n=n+1
            ax.set_title('difference in crystallo absorbance spectrum', fontsize=20, fontweight='bold')  
            ax.set_xlim([250, 1000])
            ax.set_ylim([GenPanel.diffspec.A.min()-0.05, GenPanel.diffspec.A.max()+0.1])
            ax.tick_params(labelsize=16)
            ax.legend(loc='upper right', shadow=True, prop={'size':8})
            self.canvas.draw()
            
            
class LeftPanel(GenPanel):
    
    def __init__(self, parent):
        wx.Panel.__init__(self, parent, style = wx.SUNKEN_BORDER)
        self.button_openfile = wx.Button(self, label="Open File")
        self.button_openfile.Bind(wx.EVT_BUTTON, self.on_open_file)
        
        # checkbox for TR-icOS data
        self.TRicOS_checkbox = wx.CheckBox(self, label = 'TR-icOS data ?', style = wx.CHK_2STATE)
        
        # print raw data again
        self.button_rawdat = wx.Button(self, label="Back to raw data")
        self.button_rawdat.Bind(wx.EVT_BUTTON, self.backtoraw)
        # constant baseline correction 
        self.StaticBox_const = wx.StaticBox(self, label = "Constant Baseline")
        constboxsizer = wx.StaticBoxSizer(self.StaticBox_const, wx.VERTICAL)
        self.label_topeak = wx.StaticText(self, label="wl of the peak of interest", style = wx.ALIGN_CENTER_HORIZONTAL)
        self.field_topeak = wx.TextCtrl(self, style = wx.TE_CENTER , value = '280')
        self.label_baseline_blue = wx.StaticText(self, label="Baseline blue-side Boundary", style = wx.ALIGN_CENTER_HORIZONTAL)
        self.field_baseline_blue = wx.TextCtrl(self, style = wx.TE_CENTER, value = '600')
        self.label_baseline_red = wx.StaticText(self, label="Baseline red-side Boundary", style = wx.ALIGN_CENTER_HORIZONTAL)
        self.field_baseline_red = wx.TextCtrl(self, style = wx.TE_CENTER, value = '800')
        self.button_constancorr = wx.Button(self, label="Correct for constant baseline")
        self.button_constancorr.Bind(wx.EVT_BUTTON, self.on_constant_corr)

        # scaling ?
        self.scaling_checkbox = wx.CheckBox(self, label = 'Scaling ?', style = wx.CHK_2STATE)
        # smoothing ?
        self.smoothing_checkbox = wx.CheckBox(self, label = 'Smoothing ?', style = wx.CHK_2STATE)
        
        
        #sizer block
        constboxsizer.Add(self.field_topeak, 1, wx.ALIGN_CENTER | wx.ALL, border = 2)
        constboxsizer.Add(self.label_topeak, 1, wx.ALIGN_CENTER, border = 0)
        constboxsizer.Add(self.field_baseline_blue, 1, wx.ALIGN_CENTER | wx.ALL, border = 2)
        constboxsizer.Add(self.label_baseline_blue, 1, wx.ALIGN_CENTER, border = 0)
        constboxsizer.Add(self.field_baseline_red, 1, wx.ALIGN_CENTER | wx.ALL, border = 2)
        constboxsizer.Add(self.label_baseline_red, 1, wx.ALIGN_CENTER, border  = 0)
        constboxsizer.Add(self.button_constancorr, 1, wx.EXPAND | wx.ALL, border = 2)
        checkboxsizer= wx.BoxSizer(wx.HORIZONTAL)
        checkboxsizer.Add(self.scaling_checkbox, 1, wx.ALIGN_CENTER)
        checkboxsizer.Add(self.smoothing_checkbox, 1, wx.ALIGN_CENTER)
        constboxsizer.Add(checkboxsizer, 1, wx.EXPAND , border = 5)
        
        
        #Scattering correction 
        self.StaticBox_scat = wx.StaticBox(self, label = "Scattering Baseline")
        scatboxsizer = wx.StaticBoxSizer(self.StaticBox_scat, wx.VERTICAL)
        self.button_scattercor = wx.Button(self, label="Correct for Scattering")
        self.button_scattercor.Bind(wx.EVT_BUTTON, self.on_scat_corr)
        self.label_nopeak_blue = wx.StaticText(self, label="blue-side boundary of the peakless segment", style = wx.ALIGN_CENTER_HORIZONTAL)
        self.field_nopeak_blue = wx.TextCtrl(self, style = wx.TE_CENTER)
        self.label_nopeak_red = wx.StaticText(self, label="red-side boundary of the peakless segment", style = wx.ALIGN_CENTER_HORIZONTAL)
        self.field_nopeak_red = wx.TextCtrl(self, style = wx.TE_CENTER)
        self.label_leeway_factor = wx.StaticText(self, label="expected OD in the segment (% max peak)", style = wx.ALIGN_CENTER_HORIZONTAL)
        self.field_leeway_factor = wx.TextCtrl(self, style = wx.TE_CENTER)
        # diagnostic plots ?
        self.diagplots_checkbox = wx.CheckBox(self, label = 'no diagnostic plots ?', style = wx.CHK_2STATE)
        
        #divergances
        self.box_div = wx.StaticBox(self, label = 'Segment divergences')
        divboxsizer = wx.StaticBoxSizer(self.box_div, wx.VERTICAL)
        #UV        
        self.labelUV = wx.StaticText(self, label = 'UV', style = wx.ALIGN_CENTER_HORIZONTAL)
        self.field_weighUV = wx.TextCtrl(self, value = '1', style = wx.TE_CENTER)
        UVsizer = wx.BoxSizer(wx.HORIZONTAL)
        UVsizer.Add(self.labelUV, 1, wx.ALIGN_CENTER, border = 2)
        UVsizer.Add(self.field_weighUV, 1, wx.ALIGN_CENTER)
        divboxsizer.Add(UVsizer, 1, wx.ALIGN_CENTER, border = 2)
        
        #peakless        
        self.labelpeakless = wx.StaticText(self, label = 'peakless', style = wx.ALIGN_CENTER_HORIZONTAL)
        self.field_weighpeakless = wx.TextCtrl(self, value = '1', style = wx.TE_CENTER)
        peaklesssizer = wx.BoxSizer(wx.HORIZONTAL)
        peaklesssizer.Add(self.labelpeakless, 1, wx.ALIGN_CENTER, border = 2)
        peaklesssizer.Add(self.field_weighpeakless, 1, wx.ALIGN_CENTER)
        divboxsizer.Add(peaklesssizer, 1, wx.ALIGN_CENTER, border = 2)
        
        #baseline        
        self.labelbaseline = wx.StaticText(self, label = 'baseline', style = wx.ALIGN_CENTER_HORIZONTAL)
        self.field_weighbaseline = wx.TextCtrl(self, value = '1', style = wx.TE_CENTER)
        baselinesizer = wx.BoxSizer(wx.HORIZONTAL)
        baselinesizer.Add(self.labelbaseline, 1, wx.ALIGN_CENTER, border = 2)
        baselinesizer.Add(self.field_weighbaseline, 1, wx.ALIGN_CENTER)
        divboxsizer.Add(baselinesizer, 1, wx.ALIGN_CENTER, border = 2)
        
        
        scatboxsizer.Add(self.field_nopeak_blue, 0, wx.ALIGN_CENTER | wx.ALL, border = 0)
        scatboxsizer.Add(self.label_nopeak_blue, 0, wx.ALIGN_CENTER, border = 0)
        scatboxsizer.Add(self.field_nopeak_red, 0, wx.ALIGN_CENTER | wx.ALL, border = 0)
        scatboxsizer.Add(self.label_nopeak_red, 0, wx.ALIGN_CENTER, border = 0)
        scatboxsizer.Add(self.label_leeway_factor, 0, wx.ALIGN_CENTER, border = 0)
        scatboxsizer.Add(self.field_leeway_factor, 0, wx.ALIGN_CENTER | wx.ALL, border = 0)
        
        
        
        # scatboxsizer.AddSpacer(5)
        scatboxsizer.Add(divboxsizer, 1, wx.EXPAND, border = 5)
        scatboxsizer.Add(self.button_scattercor, 1, wx.EXPAND | wx.ALL, border = 2)
        scatboxsizer.Add(self.diagplots_checkbox, 1, wx.ALIGN_CENTER)
                
        # difference spectra
        self.button_diffspec = wx.Button(self, label = 'calculate difference spectrum')
        self.button_diffspec.Bind(wx.EVT_BUTTON, self.on_diff_spec)
        
        
        # mass center
        self.mass_center_checkbox = wx.CheckBox(self, label = 'Mass center calculation ?', style = wx.CHK_2STATE)        
        
        # remove a spec
        self.button_drop_spec = wx.Button(self, label='Remove a spectrum')
        self.button_drop_spec.Bind(wx.EVT_BUTTON, self.on_drop_spec)
        
        # save
        self.button_save = wx.Button(self, label="Save figure and spectra")
        self.button_save.Bind(wx.EVT_BUTTON, self.on_save)
        
        # Add widgets to the right panel sizer
        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(self.button_openfile, 1, wx.EXPAND | wx.ALL, border = 2)
        sizer.Add(self.TRicOS_checkbox, 1, wx.ALIGN_CENTER)
        sizer.Add(self.button_rawdat, 1, wx.EXPAND | wx.ALL, border = 2)
        sizer.Add(constboxsizer, 1, wx.EXPAND, border = 5)
        
        sizer.Add(scatboxsizer, 1, wx.EXPAND, border = 5)
        sizer.Add(self.button_diffspec, 1, wx.EXPAND | wx.ALL, border = 2)
        sizer.Add(self.mass_center_checkbox, 1, wx.ALIGN_CENTER)
        sizer.Add(self.button_drop_spec, 1, wx.EXPAND | wx.ALL, border = 2)
        sizer.Add(self.button_save, 1, wx.EXPAND | wx.ALL, border = 2)
        self.SetSizer(sizer)
        # self.SetBackgroundColour('grey') 
        
        
    def on_open_file(self, event):
        self.typecorr = 'raw'
        if self.GetParent().left_panel.TRicOS_checkbox.GetValue() :
            wildcard = "TXT files (*.txt)|*.txt|All files (*.*)|*.*"
            dialog = wx.FileDialog(self, "Choose one or several files", wildcard=wildcard, style=wx.FD_OPEN | wx.FD_MULTIPLE)
            toaverage=[]
            if dialog.ShowModal() == wx.ID_OK:
                file_paths = dialog.GetPaths()
                for file_path in file_paths:
                    if "ms" in file_path :
                        name_correct=file_path.replace('ms', '000us')
                        os.rename(file_path, name_correct)
                    elif "s" in file_path and "ms" not in file_path and "us" not in file_path: 
                        name_correct=file_path.replace('s','000000us')
                        os.rename(file_path, name_correct)
                    file_name = file_path.split('/')[-1][0:-4]
                    # print(file_name)
                    if file_path[-4:] == '.txt':
                        toaverage.append(file_name)
                        GenPanel.raw_lamp[file_name] = pd.read_csv(filepath_or_buffer= file_path,
                                  sep= ";",
                                  decimal=".",
                                  skiprows=8,
                                  # index_col=0,
                                  skip_blank_lines=True,
                                  header=None,
                                  skipfooter=0,
                                  # names=['wl','I','bgd','I0','A'],
                                  engine="python")
                        
                        GenPanel.raw_lamp[file_name].index=GenPanel.raw_lamp[list(GenPanel.raw_lamp.keys())[0]].index
                        isthereabs=False
                        if len(GenPanel.raw_lamp[file_name].columns) == 5:
                            GenPanel.raw_lamp[file_name].columns=['wl','I', 'bgd', 'I0', 'A']
                            isthereabs=True
                        elif len(GenPanel.raw_lamp[file_name].columns) == 4:
                            GenPanel.raw_lamp[file_name].columns=['wl','I', 'bgd', 'I0']
                        
                        
                        GenPanel.raw_lamp[file_name].index=GenPanel.raw_lamp[file_name].wl
                        
                        
                average_signal=GenPanel.raw_lamp[list(GenPanel.raw_lamp.keys())[0]].copy()
                average_signal.I=0
                if isthereabs:
                    average_signal.A=0
                average_signal['wl']=floatize(average_signal.index)
                print(GenPanel.raw_lamp.keys())
                
                for nomfich in toaverage:
                    print(nomfich)
                    for wavelength in average_signal.wl:
                        if average_signal.loc[wavelength,'I']==0:
                            average_signal.loc[wavelength,'I']=GenPanel.raw_lamp[nomfich].loc[wavelength,'I']
                        else:
                            average_signal.loc[wavelength,'I']=(average_signal.loc[wavelength,'I']+GenPanel.raw_lamp[nomfich].loc[wavelength,'I'])/2
                        if isthereabs :
                            if average_signal.loc[wavelength,'A']==0:
                                average_signal.loc[wavelength,'A']=GenPanel.raw_lamp[nomfich].loc[wavelength,'A']
                            else:
                                average_signal.loc[wavelength,'A']=(average_signal.loc[wavelength,'A']+GenPanel.raw_lamp[nomfich].loc[wavelength,'A'])/2
                # print(average_signal)
                avgname=''.join(toaverage)
                GenPanel.raw_spec[avgname]=absorbance(average_signal.copy())
                        # print(GenPanel.raw_lamp[file_name].columns)
                print(f"File '{avgname}' added spectra list with data: {GenPanel.raw_spec[avgname].A}")
                self.update_right_panel('raw')
            dialog.Destroy()
        else :
            self.typecorr = 'raw'
            wildcard = "TXT files (*.txt)|*.txt|All files (*.*)|*.*"
            dialog = wx.FileDialog(self, "Choose one or several files", wildcard=wildcard, style=wx.FD_OPEN | wx.FD_MULTIPLE)
            if dialog.ShowModal() == wx.ID_OK:
                file_paths = dialog.GetPaths()
                for file_path in file_paths:
                    file_name = file_path.split('/')[-1][0:-4]
                    print(file_name)
                    if file_path[-4:] == '.txt':
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
        self.typecorr = 'const'
        baseline_blue = float(self.field_baseline_blue.GetValue())
        baseline_red = float(self.field_baseline_red.GetValue())
        if self.GetParent().left_panel.scaling_checkbox.GetValue() :  
            scaling_top = float(self.field_topeak.GetValue())
        for i in GenPanel.raw_spec:
            segmentend=GenPanel.raw_spec[i].wl.between(baseline_blue,baseline_red, inclusive='both')
            tmp=GenPanel.raw_spec[i].copy()
            tmp.A-=mean(GenPanel.raw_spec[i].A[segmentend])
            if self.GetParent().left_panel.scaling_checkbox.GetValue() :
                tmp.A*=1/tmp.A[tmp.wl.between(scaling_top-10,scaling_top+10,inclusive='both')].max()
            if self.GetParent().left_panel.smoothing_checkbox.GetValue() :
                tmp.A=sp.signal.savgol_filter(x=tmp.A.copy(),     #This is the smoothing function, it takes in imput the y-axis data directly and fits a polynom on each section of the data at a time
                                              window_length=21,  #This defines the section, longer sections means smoother data but also bigger imprecision
                                              polyorder=3)       #The order of the polynom, more degree = less smooth, more precise (and more ressource expensive)
            GenPanel.const_spec[i]=tmp.copy()
            GenPanel.const_spec[i].index=GenPanel.raw_spec[i].wl
            print(f"Spectrum '{i}' corrected: {GenPanel.const_spec[i].A}")
        self.update_right_panel(self.typecorr)
        
    def on_scat_corr(self, event):
        self.typecorr = 'ready'
        baseline_blue = float(self.field_baseline_blue.GetValue())
        baseline_red = float(self.field_baseline_red.GetValue())
        leewayfac= float(self.field_leeway_factor.GetValue())
        if self.GetParent().left_panel.scaling_checkbox.GetValue() :  
            scaling_top = float(self.field_topeak.GetValue())
        nopeak_blue = float(self.field_nopeak_blue.GetValue())
        nopeak_red = float(self.field_nopeak_red.GetValue())
        
        n=0
        # this plots each fitted baseline against the raw data, highlighting the chose segments
        for i in GenPanel.raw_spec :
            tmp=GenPanel.raw_spec[i].copy()
            if self.GetParent().left_panel.smoothing_checkbox.GetValue() :
                tmp.A=sp.signal.savgol_filter(x=tmp.A.copy(),     #This is the smoothing function, it takes in imput the y-axis data directly and fits a polynom on each section of the data at a time
                                              window_length=21,  #This defines the section, longer sections means smoother data but also bigger imprecision
                                              polyorder=3)
            rightborn=GenPanel.raw_spec[i].A[GenPanel.raw_spec[i].wl.between(200,250)].idxmax()+20
            leftborn=GenPanel.raw_spec[i].A[GenPanel.raw_spec[i].wl.between(200,250)].idxmax()
            segment1 = GenPanel.raw_spec[i].wl.between(leftborn,rightborn, inclusive='both')
            segment2 = GenPanel.raw_spec[i].wl.between(nopeak_blue,nopeak_red, inclusive='both')
            segmentend=GenPanel.raw_spec[i].wl.between(baseline_blue,baseline_red, inclusive='both')
            segment=segment1+segment2+segmentend
            #peakless visible segment
            sigmafor3segment=[float(self.field_weighUV.GetValue()),float(self.field_weighpeakless.GetValue()),float(self.field_weighbaseline.GetValue())]
            forfit=tmp.copy()
            if self.GetParent().left_panel.scaling_checkbox.GetValue() :  
                forfit.A[segment2]-=leewayfac*forfit.A[scaling_top-10:scaling_top+10].max()
            else :
                forfit.A[segment2]-=leewayfac*forfit.A[310:800].max()
            x=forfit.wl[segment].copy()
            y=forfit.A[segment].copy()
            initialParameters = np.array([1e9,1])
            m=len(forfit.A[segment1])
            sigma=m*[sigmafor3segment[0]]
            m=len(forfit.A[segment2])
            sigma=sigma + m*[sigmafor3segment[1]]
            m=len(forfit.A[segmentend])
            sigma=sigma + m*[sigmafor3segment[2]]
            para, pcov = sp.optimize.curve_fit(f=fct_baseline, xdata=x, ydata=y, p0=initialParameters, sigma=sigma)
            baseline=tmp.copy()
            baseline.A=fct_baseline(baseline.wl.copy(), *para)
            corrected=tmp.copy()
            corrected.A=tmp.A.copy()-baseline.A
            if self.GetParent().left_panel.scaling_checkbox.GetValue() :
                corrected.A*=1/corrected.A[corrected.wl.between(scaling_top-10,scaling_top+10,inclusive='both')].max()
            GenPanel.ready_spec[i]=corrected
            # tmp, baseline=baselinefitcorr_3seg_smooth(tmp,  segment1, segment2, segmentend, sigmafor3segment)
            if not self.GetParent().left_panel.diagplots_checkbox.GetValue() : 
                vars()['fig' + str(n)], vars()['ax' + str(n)] = plt.subplots()
                vars()['ax' + str(n)].set_title(str(i))
                vars()['ax' + str(n)].plot(GenPanel.raw_spec[i].wl,GenPanel.raw_spec[i].A)
                vars()['ax' + str(n)].plot(baseline.wl,baseline.A)
                vars()['ax' + str(n)].plot(GenPanel.raw_spec[i].wl[segment1], GenPanel.raw_spec[i].A[segment1], color = 'lime')
                vars()['ax' + str(n)].plot(GenPanel.raw_spec[i].wl[segment2], GenPanel.raw_spec[i].A[segment2], color = 'magenta')
                vars()['ax' + str(n)].plot(GenPanel.raw_spec[i].wl[segmentend], GenPanel.raw_spec[i].A[segmentend], color = 'crimson') 
                vars()['fig' + str(n)].show()
            n+=1
        self.update_right_panel(self.typecorr)
        
    def mass_center(self, typecorr):  #make typecorr a global left panel value to handle the difference spectrum 
        baseline_blue = float(self.field_baseline_blue.GetValue())
        baseline_red = float(self.field_baseline_red.GetValue())
        scaling_top = float(self.field_topeak.GetValue())
        if typecorr == 'raw':
            centroids={}
            for i in GenPanel.raw_spec:
                peakpos = float(GenPanel.raw_spec[i].A[GenPanel.raw_spec[i].wl.between(scaling_top-25,scaling_top+25)].idxmax())
                half_max = (GenPanel.raw_spec[i].A[peakpos] - GenPanel.raw_spec[i].A[baseline_blue : baseline_red].mean()) / 2
                seg = np.where(GenPanel.raw_spec[i].A > half_max + GenPanel.raw_spec[i].A[baseline_blue : baseline_red].mean())
                n=0
                # to the blue
                area=[]
                while list(GenPanel.raw_spec[i].index).index(peakpos)+n in seg[0] and GenPanel.raw_spec[i].A.iloc[list(GenPanel.raw_spec[i].index).index(peakpos)+n] > GenPanel.raw_spec[i].A.iloc[list(GenPanel.raw_spec[i].index).index(peakpos)+n:list(GenPanel.raw_spec[i].index).index(peakpos)+n+25].min():
                    area.append(GenPanel.raw_spec[i].wl.iloc[list(GenPanel.raw_spec[i].index).index(peakpos)+n])
                    n+=1
                #to the red
                n = 1
                while list(GenPanel.raw_spec[i].index).index(peakpos)-n in seg[0] and GenPanel.raw_spec[i].A.iloc[list(GenPanel.raw_spec[i].index).index(peakpos)-n] > GenPanel.raw_spec[i].A.iloc[list(GenPanel.raw_spec[i].index).index(peakpos)-n-25:list(GenPanel.raw_spec[i].index).index(peakpos)-n].min():
                    area.insert(0,GenPanel.raw_spec[i].wl.iloc[list(GenPanel.raw_spec[i].index).index(peakpos)-n])
                    n+=1
                a=np.sum(np.array(GenPanel.raw_spec[i].wl[area])*np.array(GenPanel.raw_spec[i].A[area])/np.sum(np.array(GenPanel.raw_spec[i].A[area])))
                centroids[i] = a
        elif typecorr == 'const' :
            centroids={}
            for i in GenPanel.const_spec:
                peakpos = float(GenPanel.const_spec[i].A[GenPanel.const_spec[i].wl.between(scaling_top-25,scaling_top+25)].idxmax())
                half_max = (GenPanel.const_spec[i].A[peakpos] - GenPanel.const_spec[i].A[baseline_blue : baseline_red].mean()) / 2
                seg = np.where(GenPanel.const_spec[i].A > half_max + GenPanel.const_spec[i].A[baseline_blue : baseline_red].mean())
                n=0
                # to the blue
                area=[]
                while list(GenPanel.const_spec[i].index).index(peakpos)+n in seg[0] and GenPanel.const_spec[i].A.iloc[list(GenPanel.const_spec[i].index).index(peakpos)+n] > GenPanel.const_spec[i].A.iloc[list(GenPanel.const_spec[i].index).index(peakpos)+n:list(GenPanel.const_spec[i].index).index(peakpos)+n+25].min():
                    area.append(GenPanel.const_spec[i].wl.iloc[list(GenPanel.const_spec[i].index).index(peakpos)+n])
                    n+=1
                #to the red
                n = 1
                while list(GenPanel.const_spec[i].index).index(peakpos)-n in seg[0] and GenPanel.const_spec[i].A.iloc[list(GenPanel.const_spec[i].index).index(peakpos)-n] > GenPanel.const_spec[i].A.iloc[list(GenPanel.const_spec[i].index).index(peakpos)-n-25:list(GenPanel.const_spec[i].index).index(peakpos)-n].min():
                    area.insert(0,GenPanel.const_spec[i].wl.iloc[list(GenPanel.const_spec[i].index).index(peakpos)-n])
                    n+=1
                a=np.sum(np.array(GenPanel.const_spec[i].wl[area])*np.array(GenPanel.const_spec[i].A[area])/np.sum(np.array(GenPanel.const_spec[i].A[area])))
                centroids[i] = a
        elif typecorr == 'ready' :
            centroids={}
            for i in GenPanel.ready_spec:
                peakpos = float(GenPanel.ready_spec[i].A[GenPanel.ready_spec[i].wl.between(scaling_top-25,scaling_top+25)].idxmax())
                half_max = (GenPanel.ready_spec[i].A[peakpos] - GenPanel.ready_spec[i].A[baseline_blue : baseline_red].mean()) / 2
                seg = np.where(GenPanel.ready_spec[i].A > half_max + GenPanel.ready_spec[i].A[baseline_blue : baseline_red].mean())
                n=0
                # to the blue
                area=[]
                while list(GenPanel.ready_spec[i].index).index(peakpos)+n in seg[0] and GenPanel.ready_spec[i].A.iloc[list(GenPanel.ready_spec[i].index).index(peakpos)+n] > GenPanel.ready_spec[i].A.iloc[list(GenPanel.ready_spec[i].index).index(peakpos)+n:list(GenPanel.ready_spec[i].index).index(peakpos)+n+25].min():
                    area.append(GenPanel.ready_spec[i].wl.iloc[list(GenPanel.ready_spec[i].index).index(peakpos)+n])
                    n+=1
                #to the red
                n = 1
                while list(GenPanel.ready_spec[i].index).index(peakpos)-n in seg[0] and GenPanel.ready_spec[i].A.iloc[list(GenPanel.ready_spec[i].index).index(peakpos)-n] > GenPanel.ready_spec[i].A.iloc[list(GenPanel.ready_spec[i].index).index(peakpos)-n-25:list(GenPanel.ready_spec[i].index).index(peakpos)-n].min():
                    area.insert(0,GenPanel.ready_spec[i].wl.iloc[list(GenPanel.ready_spec[i].index).index(peakpos)-n])
                    n+=1
                a=np.sum(np.array(GenPanel.ready_spec[i].wl[area])*np.array(GenPanel.ready_spec[i].A[area])/np.sum(np.array(GenPanel.ready_spec[i].A[area])))
                centroids[i] = a
        return centroids

        
    def on_diff_spec(self, event):
        file_chooser = FileChooser(self, "Choose Two Files", 2, list(GenPanel.raw_spec.keys()))
        if file_chooser.ShowModal() == wx.ID_OK:
            selections = file_chooser.check_list_box.GetCheckedStrings()
            print(selections)
            #add if statements to handle the diff spectra for all 
        if self.typecorr == 'raw':
            GenPanel.diffspec.wl = GenPanel.raw_spec[selections[0]].wl
            GenPanel.diffspec.index = GenPanel.diffspec.wl
            print(GenPanel.diffspec)
            GenPanel.diffspec.A = GenPanel.raw_spec[selections[0]].A-GenPanel.raw_spec[selections[1]].A
        elif self.typecorr == 'const':
            GenPanel.diffspec.wl = GenPanel.const_spec[selections[0]].wl
            GenPanel.diffspec.index = GenPanel.diffspec.wl
            print(GenPanel.diffspec)
            GenPanel.diffspec.A = GenPanel.const_spec[selections[0]].A-GenPanel.const_spec[selections[1]].A
        elif self.typecorr == 'ready':
            GenPanel.diffspec.wl = GenPanel.ready_spec[selections[0]].wl
            GenPanel.diffspec.index = GenPanel.diffspec.wl
            print(GenPanel.diffspec)
            GenPanel.diffspec.A = GenPanel.ready_spec[selections[0]].A-GenPanel.ready_spec[selections[1]].A
        self.update_right_panel('diff')
    
    def on_drop_spec(self, event): #htis should open a Filechooser dialog and remove the 
        file_chooser = FileChooser(self, "Choose one or more files to drop", None, list(GenPanel.raw_spec.keys()))
        if file_chooser.ShowModal() == wx.ID_OK:
            selections = file_chooser.check_list_box.GetCheckedStrings()
            for i in selections:
#                if len(GenPanel.const_spec.keys()) == len(GenPanel.raw_spec.keys()):
                try:
                    del GenPanel.const_spec[i]
                    print(f"deleting file(s) {i} from const")
#                elif len(GenPanel.ready_spec.keys()) == len(GenPanel.raw_spec.keys()):
                except KeyError:
                    print(i + 'was not deleted from const_spec as it has never been constant corrected')
                try:
                    del GenPanel.ready_spec[i]
                    print(f"deleting file(s) {i} from ready")
                except KeyError:
                    print(i + 'was not deleted from ready_spec as it has never been scattering corrected')
                del GenPanel.raw_spec[i]
                print(f"deleting files(s) {i} from raw")
             
 #this needs to be update panel with the LeftPanel.typercor variable
            self.update_right_panel(self.typecorr)
    
    def on_save(self, event):
        wildcard = "CSV files (*.csv)|*.csv|All files (*.*)|*.*"
        dialog = wx.FileDialog(self, "Save File(s)", wildcard=wildcard, style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT)
        if dialog.ShowModal() == wx.ID_OK:
            totalpath = dialog.GetPath()
            # file_path2 = file_path.split('/')[:-1]
            if platform.system() == 'Windows' :
                file_path=''
                for i in totalpath.split('\\')[:-1]:
                    file_path+=i+'\\'
                print(file_path)
                file_name = totalpath.split('\\')[-1][0:-4]
            else:# or platform.system() == 'MacOS'
                file_path=''
                for i in totalpath.split('/')[:-1]:
                    file_path+=i+'/'
                print(file_path)
                file_name = totalpath.split('/')[-1]
                
        dialog.Destroy()
        towrite_raw_spectra=GenPanel.raw_spec[next(iter(GenPanel.raw_spec))].drop(columns=['wl','A'])
        for spec in GenPanel.raw_spec:
            towrite_raw_spectra[spec]=GenPanel.raw_spec[spec].A
            print("File" + file_path + f" '{spec}' saved in: raw_{file_name}.csv in column {spec}")
        towrite_raw_spectra.to_csv(file_path + 'raw_' +  file_name + ".csv", index=True)
        if len(GenPanel.const_spec)==len(GenPanel.raw_spec):
            towrite_constant_spectra=GenPanel.const_spec[next(iter(GenPanel.const_spec))].drop(columns=['wl','A'])
            for spec in GenPanel.const_spec:
                towrite_constant_spectra[spec]=GenPanel.const_spec[spec].A
                print("File" + file_path + f" '{spec}' saved in: constant_{file_name}.csv in column {spec}")
            towrite_constant_spectra.to_csv(file_path + 'constant_' +  file_name + ".csv", index=True)
        if len(GenPanel.ready_spec)==len(GenPanel.raw_spec):
            towrite_ready_spectra=GenPanel.ready_spec[next(iter(GenPanel.ready_spec))].drop(columns=['wl','A'])
            for spec in GenPanel.ready_spec:
                towrite_ready_spectra[spec]=GenPanel.ready_spec[spec].A
                print("File" + file_path + f" '{spec}' saved in: ready_{file_name}.csv in column {spec}")
            towrite_ready_spectra.to_csv(file_path + 'ready_' +  file_name + ".csv", index=True)
        self.GetParent().right_panel.figure.savefig(file_path + file_name + ".svg", dpi=900 , transparent=True,bbox_inches='tight')
        self.GetParent().right_panel.figure.savefig(file_path + file_name + ".png", dpi=900, transparent=True,bbox_inches='tight')
        self.GetParent().right_panel.figure.savefig(file_path + file_name + ".pdf", dpi=900, transparent=True,bbox_inches='tight')
        print("Figure saved at: " + file_path + file_name + '.png')
        
        
    def update_right_panel(self, typecorr):
        if len(self.field_topeak.GetValue()) == 0:
            scaling_top=280
        else :
            scaling_top = float(self.field_topeak.GetValue())
        print(scaling_top)
        self.GetParent().right_panel.plot_data(typecorr, scaling_top)
        
        
    def backtoraw(self, event):
        self.typecorr='raw'
        if len(self.field_topeak.GetValue()) == 0:
            scaling_top=280
        else :
            scaling_top = float(self.field_topeak.GetValue())
        print(scaling_top)
        self.GetParent().right_panel.plot_data('raw', scaling_top)
            


class FileChooser(wx.Dialog):
    def __init__(self, parent, title, numtodrop, files):
        super().__init__(parent, title=title)
        self.numtodrop = numtodrop
        self.files = files

        self.check_list_box = wx.CheckListBox(self, choices=self.files)

        self.btn_ok = wx.Button(self, wx.ID_OK)
        self.btn_ok.Bind(wx.EVT_BUTTON, self.on_ok)
        self.btn_ok.Enable(False)
        self.btn_cancel = wx.Button(self, wx.ID_CANCEL)
        # self.InitUI()
        
        btn_sizer = wx.BoxSizer(wx.HORIZONTAL)
        btn_sizer.Add(self.btn_ok)
        btn_sizer.Add(self.btn_cancel)

        main_sizer = wx.BoxSizer(wx.VERTICAL)
        main_sizer.Add(self.check_list_box, proportion=1, flag=wx.EXPAND)
        main_sizer.Add(btn_sizer, flag=wx.ALIGN_RIGHT)

        self.SetSizer(main_sizer)

        self.Bind(wx.EVT_CHECKLISTBOX, self.on_checklistbox)
    
    def on_checklistbox(self, event):
        selections = self.check_list_box.GetChecked()
        if self.numtodrop != None:
            self.btn_ok.Enable(len(selections) == self.numtodrop)
        else:
            self.btn_ok.Enable(len(selections)>0)

    def on_ok(self, event):
        self.EndModal(wx.ID_OK)
    
    
    # def InitUI(self): # now this needs to behave differently based on th emenu option chosen
    #     menubar = wx.MenuBar()
    #     fileMenu = wx.Menu()
    #     options = ["raw", "const", "scat"]
    #     for option in options:
    #         item = fileMenu.Append(wx.ID_ANY, option)
    #         self.Bind(wx.EVT_MENU, self.OnOption, item)
    #     menubar.Append(fileMenu, '&Options')
    #     self.GetParent().GetParent().GetParent().SetMenuBar(menubar) # this needs to be mapped to the frame

    def OnOption(self, event):
        selected_option = self.GetMenuBar().FindItemById(event.GetId()).GetLabel()
        wx.MessageBox("You selected " + selected_option)

class MainFrame(wx.Frame):
    def __init__(self):
        wx.Frame.__init__(self, None, title="icOS toolbox", size = (1400,1000))
        # Create splitter
        self.splitter = wx.SplitterWindow(self)
        # Create left and right panels
        self.splitter.left_panel = LeftPanel(self.splitter)
        self.splitter.right_panel = RightPanel(self.splitter)
        # Add panels to splitter
        self.splitter.SplitVertically(self.splitter.left_panel, self.splitter.right_panel, 350)
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

#TODO : fix the window size to have it as a percentage of the screen it is being displayed on 

if __name__ == "__main__":
    app = wx.App()
    frame = MainFrame()
    app.MainLoop()