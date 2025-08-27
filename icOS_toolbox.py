# -*- coding: utf-8 -*-

"""
Created on Wed Jan 18 15:07:56 2023
@author: NCARAMEL
"""
import warnings
warnings.simplefilter("ignore")
import importlib
import math
#instlaling packages if not present 

try:
    #import wx as wx
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

# try:
#     import matplotlib
#     print("matplotlib is already installed.")
# except ImportError:
#     print("matplotlib is not installed. Installing now...")
#     # Install SciPy using pip
#     try:
#         import pip
#     except ImportError:
#         print("pip is not installed. Please install pip to continue.")
#     else:
#         pip.main(['install', 'matplotlib'])
#         print("matplotlib has been installed.")
#         import matplotlib


# from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas
from statistics import mean
# from matplotlib.figure import Figure
 
# from matplotlib.backends.backend_wx import NavigationToolbar2Wx
# matplotlib.use('wxAgg')
# plt.rcParams["figure.figsize"] = (3.33/2.54,2.5/2.54)
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
    from scipy import signal
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
# from scipy import signal 

try:
    re = importlib.import_module('re')
    print("re is already installed.")
except ImportError:
    print("re is not installed. Installing now...")
    # Install SciPy using pip
    try:
        import pip
    except ImportError:
        print("pip is not installed. Please install pip to continue.")
    else:
        pip.main(['install', 're'])
        print("re has been installed.")
        wx = importlib.import_module('re')
try :
    import wxmplot.interactive as wi
    from wxmplot import PlotPanel
    print("wxmplot is already installed.")
except ImportError:
    print("wxmplot is not installed. Installing now...")
    # Install SciPy using pip
    try:
        import pip
    except ImportError:
        print("pip is not installed. Please install pip to continue.")
    else:
        pip.main(['install', 'wxmplot'])
        print("wxmplot has been installed.")
        import wxmplot.interactive as wi
        from wxmplot import PlotPanel
    
def rgb_to_hex(rgb):
    """
    Convert RGB tuple of floats to HEX string.

    Parameters:
    rgb (tuple): Tuple containing three floats representing RGB values (0-1 range).

    Returns:
    str: HEX string representation of the RGB color.
    """
    r, g, b = rgb
    r = int(r * 255)
    g = int(g * 255)
    b = int(b * 255)
    return '#{0:02x}{1:02x}{2:02x}'.format(r, g, b)



# import matplotlib
import matplotlib.pyplot as plt  

plt.rcParams.update({'font.size': 50})
# matplotlib.use('QTAgg')
from matplotlib.colors import LinearSegmentedColormap
from collections import Counter
import tempfile



if 'app' in vars():
    del app


system = platform.system()
if system == "Windows":
    path = os.path.join("C:\\", "path", "to", "save", "output")
else:
    path = os.path.join("/", "path", "to", "save", "output")


import numpy as np  # Importing numpy library for mathematical operations

def straightforward_solution(x, a, b, c, d):
    """
    This function calculates a reflection coefficient using the given parameters.

    Parameters:
    - x: Input value for the function
    - a, b, c, d, e: Coefficients for the reflection calculation

    Returns:
    - Reflection coefficient calculated based on the given formula

    Formula:
    The function computes the reflection coefficient using the following formula:
    (e/(x^4)) + |((a + b/(x^2)) - (c + d/(x^2))) / ((a + b/(x^2)) + (c + d/(x^2)))|^2
    where '^' denotes exponentiation and '|' denotes Absolute value.
    """

    # Calculating the reflection coefficient using the provided formula
    return a*x + b/np.power(x,4)+c/np.power(x,2)+d


def full_correction(x, a, b, c, d, e):
    """
    This function calculates a reflection coefficient using the given parameters.

    Parameters:
    - x: Input value for the function
    - a, b, c, d, e: Coefficients for the reflection calculation

    Returns:
    - Reflection coefficient calculated based on the given formula

    Formula:
    The function computes the reflection coefficient using the following formula:
    (e/(x^4)) + |((a + b/(x^2)) - (c + d/(x^2))) / ((a + b/(x^2)) + (c + d/(x^2)))|^2
    where '^' denotes exponentiation and '|' denotes Absolute value.
    """

    # Calculating the reflection coefficient using the provided formula
    reflection_coefficient = (e / (np.power(x, 4))) + np.power(
        np.abs(((a + b / np.power(x, 2)) - (c + d / np.power(x, 2))) /
               ((a + b / np.power(x, 2)) + (c + d / np.power(x, 2)))), 2)

    return reflection_coefficient

def fct_monoexp(x, a, b, tau):
    return a + b*np.exp(-x/tau)



def fct_Hill(x, ini, maximum, Km, rate):
    """
    Hill equation function.

    Parameters:
        x (float or array-like): Input variable (e.g., ligand concentration).
        Vmax (float): Maximum response of the system.
        Km (float): Concentration of x at half-maximal response (or dissociation constant).
        n (float): Hill coefficient, describing the steepness of the curve.

    Returns:
        float or array-like: Response of the system.
    """
    return ini+(maximum-ini)/(1+np.power(Km/x,rate))

def custom_correction(x, a, b, n):
    """
    A function that models a custom baseline
    Parameters:
    x (numpy array): x values
    a (float): The parameter a for the function
    b (float): The parameter b for the function
    n (float): The power for the function fitted, usually starts at 4
    Returns:
    numpy array: The values of the function evaluated at the given x values
    """
    return a/np.power(x,n)+b



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



def parse_float(s):
    s = str(s).strip()
    if not s:
        raise ValueError("empty")
    # Allow European decimal comma
    s = s.replace(",", ".")
    x = float(s)  # will raise ValueError if invalid
    if not math.isfinite(x):  # reject 'nan', 'inf', etc.
        raise ValueError("non-finite")
    return x

def Absorbance(tmp):
    """
    Function to calculate Absorbance from spectral data.

    Parameters:
    tmp (DataFrame): Input DataFrame containing spectral data.

    Returns:
    DataFrame: DataFrame with Absorbance values calculated.

    This function takes a DataFrame `tmp` containing spectral data as input. If the DataFrame
    contains a column labeled 'A', it returns a copy of the DataFrame without the columns 'I',
    'bgd', and 'I0', indicating that Absorbance data is already present. If 'A' is not found,
    it calculates Absorbance values using the formula Absorbance = -log10((Intensity - Background) / (Reference - Background)).
    The calculated Absorbance values are stored in a new column 'A' in the DataFrame.
    """

    ourdata = tmp.copy()

    # Check if 'A' column exists in DataFrame
    if 'A' in ourdata.columns:
        print('avantes Absorbance saved for spectrum')
        # If 'A' column exists, return DataFrame without 'I', 'bgd', and 'I0' columns
        return ourdata.drop(columns=['I', 'bgd', "I0"]).copy()
    else:
        # If 'A' column does not exist, calculate Absorbance and store in 'A' column
        ourdata['Absor'] = None
        for wl in ourdata.index:
            tmpdat = float(ourdata.I[wl] - ourdata.bgd[wl])
            tmpref = float(ourdata.I0[wl] - ourdata.bgd[wl])
            if tmpref == 0:  # Prevent division by zero
                tmpAbs = 0
            elif tmpdat / tmpref < 0:
                tmpAbs = 0
            else:
                tmpAbs = -np.log(tmpdat / tmpref)
            ourdata.loc[wl, 'A'] = tmpAbs  # Store calculated Absorbance value in 'A' column

        # Return DataFrame without 'I', 'bgd', and 'I0' columns
        return ourdata.drop(columns=['I', 'bgd', "I0"]).copy()


def Absorbance_multiscan(tmp): #TODO
    """
    Function to calculate Absorbance from spectral data from a file that contains several scans.

    Parameters:
    tmp (DataFrame): Input DataFrame containing spectral data.

    Returns:
    DataFrame: DataFrame with Absorbance values calculated.

    This function takes a DataFrame `tmp` containing spectral data as input. If the DataFrame
    contains a column labeled 'A', it returns a copy of the DataFrame without the columns 'I',
    'bgd', and 'I0', indicating that Absorbance data is already present. If 'A' is not found,
    it calculates Absorbance values using the formula Absorbance = -log10((Intensity - Background) / (Reference - Background)).
    The calculated Absorbance values are stored in a new column 'A' in the DataFrame.
    """

    ourdata = tmp.copy()
    # print(ourdata.columns)
    wl=ourdata.index[0]
    print(ourdata['ref']-ourdata['dark'])
    print(ourdata['dark'][wl])
    # print(ourdata.index)
    # Check if 'A' column exists in DataFrame

    for wl in ourdata.index:
        tmpref = ourdata['ref'][wl] - ourdata['dark'][wl]
        for i in ourdata.columns[3:]:
            tmpdat = ourdata[i][wl] - ourdata['dark'][wl]
            if tmpref == 0:  # Prevent division by zero
                tmpAbs = 0
            elif tmpdat / tmpref < 0:
                tmpAbs = 0
            else:
                tmpAbs = -np.log(tmpdat / tmpref)
            ourdata.loc[wl, i] = tmpAbs  # Store calculated Absorbance value in 'A' column
    ourdata.drop(columns=['dark', 'ref'], inplace=True)
    raw_spec={}
    for i in ourdata.columns[3:]:
        raw_spec[i]=ourdata[['wl',i]]
        raw_spec[i].columns=['wl','A']
        # Return a dictionary containing DataFrames with 'wl' and 'A' as columns
    return raw_spec


floatize=np.vectorize(float)   


def longest_digit_sequence(input_string):
    """
    Find the longest digit sequence in a given string.

    Parameters:
    input_string (str): The input string to search for digit sequences.

    Returns:
    str: The longest digit sequence found in the input string.

    This function uses regular expression to find all digit sequences in the input string.
    It then identifies the longest digit sequence using the max function with the 'key' parameter set to len.
    If no digit sequences are found, it returns None.
    """
    # Use regular expression to find all digit sequences in the string
    digit_sequences = re.findall(r'\d+', input_string)
    
    # Find the longest digit sequence
    longest_sequence = max(digit_sequences, key=len, default=None)
    
    return longest_sequence






def guess_separator(line):
    # Define potential separators
    if ';  ' in line :
        guessed_separator = ';'
        guessed_decimal = ','
        return guessed_separator, guessed_decimal
        
    elif ',' in line and not '.' in line:
        separators = ['\t', ';', ' '] # for french formats with comma as decimals
        guessed_decimal=','
    else:
        separators = ['\t', ',', ';', ' ']
        guessed_decimal='.'
    max_separators = 0
    guessed_separator = None
    
    # Iterate through separators and count occurrences
    
    for sep in separators:
        count = line.count(sep)
        if count > max_separators:
            max_separators = count
            guessed_separator = sep
    return guessed_separator, guessed_decimal

def universal_opener(file_path):
    
    delimiter_list=[]
    decimal_list=[]
    with open(file_path, 'r') as infile:
        # with open(output_file, 'w') as outfile:
        content=infile.read()
        if 'JASCO' in content:
            print('JASCO spectrum')
            with open(file_path, 'r') as infile:
                linecounter=0
                for line in infile:
                    separator, dec = guess_separator(line)
                    if linecounter>20 and linecounter < content.count('\n')-47:
                        delimiter_list.append(separator)
                        decimal_list.append(dec)
                    linecounter+=1
            
        else:
            with open(file_path, 'r') as infile:
                for line in infile:
                
                    separator, dec = guess_separator(line)
                # print(separator)
                    delimiter_list.append(separator)
                    decimal_list.append(dec)
                # print(separator)

    counter = Counter(delimiter_list)
    most_common = counter.most_common(1)
    delimiter=most_common[0][0]
    
    counter = Counter(decimal_list)
    most_common = counter.most_common(1)
    decimal=most_common[0][0]
    
    
    # Temporary file to store filtered lines
    temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False)

    with open(file_path, 'r') as file:
        for line in file:
            # Check if line starts with a number and contains delimiter
            if line[0].isdigit() and delimiter in line and decimal in line:
                temp_file.write(line)

    temp_file.close()

    # Read the temporary file using pandas
    df = pd.read_csv(temp_file.name, 
                     delimiter=delimiter,
                     decimal=decimal,
                     names=['wl','A'],
                     engine="python")
    df.index=df.wl
    # Remove temporary file
    os.unlink(temp_file.name)

    return df

def multiscan_opener(file_path):
    delimiter_list=[]
    decimal_list=[]
    with open(file_path, 'r') as infile:
        for line in infile:
            # print(';  ' in line)
            separator, dec = guess_separator(line)
            # print(separator, dec)
        # print(separator)
            delimiter_list.append(separator)
            decimal_list.append(dec)
    counter = Counter(delimiter_list)
    most_common = counter.most_common(1)
    delimiter=most_common[0][0]
    
    counter = Counter(decimal_list)
    most_common = counter.most_common(1)
    decimal=most_common[0][0]
    
    temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False)

    with open(file_path, 'r') as file:
        for line in file:
            # Check if line starts with a number and contains delimiter
            cter=0
            nodigit=0
            for i in line[0:10]:
                if i.isdigit() or i == delimiter or i == decimal: 
                    cter+=1
                else:
                    nodigit+=1
            if cter > nodigit:
                    temp_file.write(line)
    
    temp_file.close()

    
    
    raw_scan = pd.read_csv(temp_file.name, 
                     delimiter=delimiter,
                     decimal=decimal,
                     # names=['wl','A'],
                     engine="python",
                     header=None)
    raw_scan.columns = ['wl','dark','ref'] + [f'scan{i+1}' for i in range(len(raw_scan.columns) - 3)]
    raw_scan.index = raw_scan.wl
    
    with open(file_path, 'r') as infile:
        # with open(output_file, 'w') as outfile:
        content=infile.read()
        if 'Measurement mode: Scope' in content and '1204051U1' in content:
            print('CALAIDOSCOPE scope spectrum')
            timestamps=pd.read_csv(file_path, header=None, sep=';', skiprows=9, nrows=1, names=raw_scan.columns)
            # tmpstamp=tmpstamp.iloc[:,3:]
            # np.array(tmpstamp[0])
    return Absorbance_multiscan(raw_scan), timestamps


class GenPanel(wx.Panel):
    raw_lamp={}
    raw_spec = {}
    const_spec = {}
    ready_spec = {}
    diffserie ={}
    diffspec = pd.DataFrame(data=None,columns=['wl','A'])
    list_spec = pd.DataFrame(data=None, columns = ['file_name','time_code','Abs','laser_dent_blue','laser_dent_red'])
    list_spec.index = list_spec.file_name
    smoothing='savgol'
    correction='full'
    
class MainFrame(wx.Frame):
    def __init__(self):
        wx.Frame.__init__(self, None, title="icOS toolbox", size=(1000, 800))
        
        # Create splitter
        self.splitter = wx.SplitterWindow(self)
        
        # Create left panel with notebook
        self.splitter.left_panel = LeftPanel(self.splitter)
        
        # Create right panel
        self.splitter.right_panel = RightPanel(self.splitter)
        
        # Add panels to splitter
        self.splitter.SplitVertically(self.splitter.left_panel, self.splitter.right_panel, 450)
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

class Modified_plot_panel(PlotPanel):
    def __init__(self, parent):
        PlotPanel.__init__(self, parent, dpi=150, fontsize=3,size=(700, 700))
    
    def plot_many_modified(self, datalist, side='left', title=None,
                  xlabel=None, ylabel=None, show_legend=False, zoom_limits=None, **kws):
        """
        plot many traces at once, taking a list of (x, y) pairs, with lines and a spectral palette
        """
        def unpack_tracedata(tdat, **kws):
            if (isinstance(tdat, dict) and
                'xdata' in tdat and 'ydata' in tdat):
                xdata = tdat.pop('xdata')
                ydata = tdat.pop('ydata')
                out = kws
                out.update(tdat)
            elif isinstance(tdat, (list, tuple)):
                out = kws
                xdata = tdat[0]
                ydata = tdat[1]
            return (xdata, ydata, out)

        
        conf = self.conf
        opts = dict(side=side, title=title, xlabel=xlabel, ylabel=ylabel,
                    delay_draw=True, show_legend=False)
        opts.update(kws)
        x0, y0, opts = unpack_tracedata(datalist[0], **opts)

        nplot_traces = len(conf.traces)
        nplot_request = len(datalist)
        if nplot_request > nplot_traces:
            linecolors = conf.linecolors
            ncols = len(linecolors)
            for i in range(nplot_traces, nplot_request+5):
                conf.init_trace(i,  linecolors[i%ncols], 'dashed')
        palette = [rgb_to_hex(x) for x in sns.color_palette(palette='Spectral', n_colors=len(datalist))]
        self.plot(x0, y0, markersize=0, color=palette[0], style='line', fill=False,  **opts)
        i=1
        for dat in datalist[1:]:
            x, y, opts = unpack_tracedata(dat, delay_draw=True)
            self.oplot(x, y, markersize=0, color=palette[i], style='line', fill=False, **opts)
            i+=1

        self.reset_formats()
        self.set_zoomlimits(zoom_limits)
        self.conf.show_legend = show_legend
        if show_legend:
            conf.draw_legend(delay_draw=True)
        conf.relabel(delay_draw=True)
        self.draw()
        # self.canvas.Refresh()
        
        
        
        
    def plot_quality(self, datalist, title = 'Quality', xlabel='Wavelength', ylabel = 'Absorbance [AU]', 
                     I0=None, side='left', zoom_limits=None, show_legend=False, **kws):
        """
        plot many traces at once, taking a list of (x, y) pairs
        """
        def unpack_tracedata(tdat, **kws):
            if (isinstance(tdat, dict) and
                'xdata' in tdat and 'ydata' in tdat):
                xdata = tdat.pop('xdata')
                ydata = tdat.pop('ydata')
                out = kws
                out.update(tdat)
            elif isinstance(tdat, (list, tuple)):
                out = kws
                xdata = tdat[0]
                ydata = tdat[1]
            return (xdata, ydata, out)


        conf = self.conf
        opts = dict(side=side, title=title, xlabel=xlabel, ylabel=ylabel,
                    delay_draw=True, show_legend=False)
        opts.update(kws)
        # x0, y0 = datalist[0][0], datalist[0][1]
        x0, y0, opts = unpack_tracedata(datalist[0])#, **opts)
        
        nplot_traces = len(conf.traces)
        nplot_request = len(datalist)
        if nplot_request > nplot_traces:
            linecolors = conf.linecolors
            ncols = len(linecolors)
            for i in range(nplot_traces, nplot_request+5):
                conf.init_trace(i,  linecolors[i%ncols], 'dashed')
        # palette = [rgb_to_hex(x) for x in sns.color_palette(palette='Spectral', n_colors=len(datalist))]
        
        colors = [[1.0, 0.8, 0.4], [0.4, 1.0, 0.4] ]  # Red to Green
        cmap = LinearSegmentedColormap.from_list("Custom", colors, N=len(I0))

        normalized_I=np.array((I0-1000)/(10000))

        for i in range(0, len(normalized_I)):
            if normalized_I[i]>1:
                normalized_I[i]=1
            elif normalized_I[i]<0:
                normalized_I[i]=0

        palette=[rgb_to_hex(x[:3]) for x in cmap(normalized_I)]
        print(x0,y0)
        self.plot(x0, y0, marker='o', markersize=4, linewidth=0, color=palette[i], alpha=0.5,  delay_draw=True)
        i=1
        for dat in datalist[1:]: #for i in range (1, len(datalist)):
            # x, y = datalist[i][0], datalist[i][1]
            x, y, opts = unpack_tracedata(dat, delay_draw=True)
            self.oplot(x, y, marker='o', markersize=4, linewidth=0, style = 'line', color=palette[i], alpha=0.5, delay_draw=True)
            i+=1

        self.reset_formats()
        self.set_zoomlimits(zoom_limits)
        self.conf.show_legend = show_legend
        if show_legend:
            conf.draw_legend(delay_draw=True)
        conf.relabel(delay_draw=True)
        self.draw()
        # self.canvas.Refresh()

        

class RightPanel(GenPanel):
    def __init__(self, parent):
        wx.Panel.__init__(self, parent, style = wx.FULL_REPAINT_ON_RESIZE | wx.SUNKEN_BORDER)
        self.plot_panel = Modified_plot_panel(self)
        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(self.plot_panel, proportion = 1, flag = wx.EXPAND)
        self.SetSizer(sizer)
        
#TODO change the plotting option to add a way to change the plot function to plot-many when there are more than 100 curves
    def plot_data(self,typecorr,scaling_top):
        self.plot_panel.clear()

        if self.GetParent().left_panel.tab1.TRicOS_checkbox.GetValue() :
            pal='Spectral'
        else :
            pal='Spectral'
            
        if typecorr == 'raw':
            self.plot_panel.clear()
   
            palette=sns.color_palette(palette=pal, n_colors=len(GenPanel.raw_spec))   
            n=0   
            if len(GenPanel.raw_spec) > 30 :
                list_toplot=[]
            if self.GetParent().left_panel.tab1.mass_center_checkbox.GetValue() :
                centroids = self.GetParent().left_panel.tab1.mass_center(typecorr = typecorr)                                        
            for i in GenPanel.raw_spec : #GenPanel.raw_spec : 
                if self.GetParent().left_panel.tab1.mass_center_checkbox.GetValue() :
                    if len(GenPanel.raw_spec) > 30 :
                        list_toplot.append((np.array(GenPanel.raw_spec[i].wl),                  
                            np.array(GenPanel.raw_spec[i].A)))
                    else : 
                        self.plot_panel.oplot(np.array(GenPanel.raw_spec[i].wl),                  
                                              np.array(GenPanel.raw_spec[i].A) ,                   
                                              linewidth=2,  
                                              style='line',
                                              marker=None,markersize=0,                  
                        
                        label=i +" mass center = " +format(centroids[i], '.3f'), 
                        color=rgb_to_hex(palette[n])) 
                    self.plot_panel.axvline(centroids[i], color = palette[n], ls = '-.') #TODO fix this
                else:
                                        
                            if self.GetParent().left_panel.tab1.scaling_checkbox.GetValue()  and scaling_top != 0 :
                                tmp= GenPanel.raw_spec[i].A / GenPanel.raw_spec[i].A[GenPanel.raw_spec[i].wl.between(scaling_top-20,scaling_top+20,inclusive='both')].mean()
                                # print(palette[n], rgb_to_hex(palette[n]))
                                if len(GenPanel.raw_spec) > 30 :
                                    list_toplot.append((np.array(GenPanel.raw_spec[i].wl),                  
                                        np.array(tmp)))
                                else: 
                                    self.plot_panel.oplot(np.array(GenPanel.raw_spec[i].wl),                  
                                        np.array(tmp) ,                   
                                        linewidth=2,
                                        style='line',
                                        marker=None,markersize=0,
                                        label=i +" max Abs = " +format(GenPanel.raw_spec[i][GenPanel.raw_spec[i].wl.between(scaling_top-20,scaling_top+20)].A.idxmax(), '.3f'), 
                                        color=rgb_to_hex(palette[n]) ,ylabel='Absorbance [AU]', xlabel='Wavelength [nm]') 
                                # if self.GetParent().left_panel.tab1.scaling_checkbox.GetValue() :
                                #     for spec in GenPanel.raw_spec:
                                #         GenPanel.raw_spec[spec].A *=1/GenPanel.raw_spec[spec].A[GenPanel.raw_spec[spec].wl.between(scaling_top-5,scaling_top+5,inclusive='both')].mean()
                            elif self.GetParent().left_panel.tab1.scaling_checkbox.GetValue()  and scaling_top == 0 :
                                tmp_scaling_top = float(GenPanel.raw_spec[i].A[GenPanel.raw_spec[i].wl.between(300,800,inclusive='both')].idxmax())
                                # print(tmp_scaling_top)
                                tmp= GenPanel.raw_spec[i].A / GenPanel.raw_spec[i].A[GenPanel.raw_spec[i].wl.between(tmp_scaling_top-10,tmp_scaling_top+10,inclusive='both')].mean()
                                if len(GenPanel.raw_spec) > 30 :
                                    list_toplot.append((np.array(GenPanel.raw_spec[i].wl),                  
                                        np.array(tmp)))
                                else: 
                                    self.plot_panel.oplot(np.array(GenPanel.raw_spec[i].wl),                  
                                        np.array(tmp) , 
                                        style='line',
                                        marker=None,markersize=0,                                    
                                        linewidth=2,
                                        label=i +" max Abs = " +format(GenPanel.raw_spec[i][GenPanel.raw_spec[i].wl.between(tmp_scaling_top-10,tmp_scaling_top+10)].A.idxmax(), '.3f'), 
                                        color=rgb_to_hex(palette[n]) ,ylabel='Absorbance [AU]', xlabel='Wavelength [nm]') 
                            else :
                                if len(GenPanel.raw_spec) > 30 :
                                    list_toplot.append((np.array(GenPanel.raw_spec[i].wl),                  
                                        np.array(GenPanel.raw_spec[i].A)))
                                else:
                                    self.plot_panel.oplot(np.array(GenPanel.raw_spec[i].wl),                  
                                        np.array(GenPanel.raw_spec[i].A) ,                   
                                        linewidth=2,
                                        style='line',
                                        marker=None,markersize=0,
                                        label=i +" peak max = " +format(GenPanel.raw_spec[i][GenPanel.raw_spec[i].wl.between(scaling_top-20,scaling_top+20)].A.idxmax(), '.3f'), 
                                        color=rgb_to_hex(palette[n]) ,ylabel='Absorbance [AU]', xlabel='Wavelength [nm]')   
                n=n+1
            if len(GenPanel.raw_spec) > 30 :
                self.plot_panel.plot_many_modified(datalist=list_toplot,ylabel='Absorbance [AU]', xlabel='Wavelength [nm]')
  
        elif typecorr == 'const':
            self.plot_panel.clear()
            palette=sns.color_palette(palette=pal, n_colors=len(GenPanel.const_spec))
            print('plotting constant corrected data')
            list_toplot=[]
            n=0
            if self.GetParent().left_panel.tab1.mass_center_checkbox.GetValue() :
                centroids = self.GetParent().left_panel.tab1.mass_center(typecorr = typecorr)  
            for i in GenPanel.list_spec.file_name : 
                if len(GenPanel.const_spec) > 30 :
                    list_toplot.append((np.array(GenPanel.const_spec[i].wl),
                                          np.array(GenPanel.const_spec[i].A)))
                else:
                    if self.GetParent().left_panel.tab1.mass_center_checkbox.GetValue() :
                            self.plot_panel.oplot(np.array(GenPanel.const_spec[i].wl),
                                                      np.array(GenPanel.const_spec[i].A) ,
                                                      linewidth=2,
                                                      style='line',
                                                      marker=None,markersize=0,
                                                      label=i+" mass center = " +format(centroids[i], '.3f'), 
                                                      color=rgb_to_hex(palette[n]) ,ylabel='Absorbance [AU]', xlabel='Wavelength [nm]') 
                            self.plot_panel.axvline(centroids[i], color = palette[n], ls = '-.') #TODO fix this
                    else :
                        if self.GetParent().left_panel.tab1.scaling_checkbox.GetValue() and scaling_top != 0 :
                            self.plot_panel.oplot(np.array(GenPanel.const_spec[i].wl),
                                                          np.array(GenPanel.const_spec[i].A) ,
                                                          linewidth=2,
                                                          style='line',
                                                          marker=None,markersize=0,
                                                          label=i+"Max Abs peak ="+format(GenPanel.const_spec[i][GenPanel.const_spec[i].wl.between(scaling_top-20,scaling_top+20)].A.idxmax(), '.2f'), 
                                                          color=rgb_to_hex(palette[n]) ,ylabel='Absorbance [AU]', xlabel='Wavelength [nm]') 
                        else :
                            self.plot_panel.oplot(np.array(GenPanel.const_spec[i].wl),
                                                          np.array(GenPanel.const_spec[i].A) ,
                                                          linewidth=2,
                                                          style='line',
                                                          marker=None,markersize=0,
                                                          label=i+"Max Abs peak ="+format(GenPanel.const_spec[i][GenPanel.const_spec[i].wl.between(scaling_top-20,scaling_top+20)].A.idxmax(), '.2f'),
                                                          color=rgb_to_hex(palette[n]) ,ylabel='Absorbance [AU]', xlabel='Wavelength [nm]') 
                            
                n=n+1
            if len(GenPanel.const_spec) > 30 :
                self.plot_panel.plot_many_modified(datalist=list_toplot,ylabel='Absorbance [AU]', xlabel='Wavelength [nm]')
     
        
        elif typecorr == 'ready':
            self.plot_panel.clear()

            print('plotting scattering corrected spectra')
            
            palette=sns.color_palette(palette=pal, n_colors=len(GenPanel.ready_spec))   
            n=0  
            list_toplot=[]
            
            if self.GetParent().left_panel.tab1.mass_center_checkbox.GetValue() :
                centroids = self.GetParent().left_panel.tab1.mass_center(typecorr = typecorr)                                            
            for i in GenPanel.list_spec.file_name : #GenPanel.ready_spec :
                if len(GenPanel.ready_spec) > 30:
                    list_toplot.append((np.array(GenPanel.ready_spec[i].wl), np.array(GenPanel.ready_spec[i].A)))
                else:
                    if self.GetParent().left_panel.tab1.mass_center_checkbox.GetValue() :
                        self.plot_panel.oplot(np.array(GenPanel.ready_spec[i].wl),                  
                              np.array(GenPanel.ready_spec[i].A) ,                   
                              linewidth=2,
                              style='line',
                              marker=None,markersize=0,
                              label=i +" mass center = " +format(centroids[i], '.3f'), 
                              color=rgb_to_hex(palette[n]) ,ylabel='Absorbance [AU]', xlabel='Wavelength [nm]') 
                        self.plot_panel.axvline(centroids[i], color = palette[n], ls = '-.') #TODO fix this
                    else :
                        if self.GetParent().left_panel.tab1.scaling_checkbox.GetValue() and scaling_top != 0 :
                            self.plot_panel.oplot(np.array(GenPanel.ready_spec[i].wl),                  
                                    np.array(GenPanel.ready_spec[i].A) ,                   
                                    linewidth=2,
                                    style='line',
                                    marker=None,markersize=0,
                                    label=i +" mass center = " +format(GenPanel.ready_spec[i][GenPanel.ready_spec[i].wl.between(scaling_top-20,scaling_top+20)].A.idxmax(), '.2f'), 
                                    color=rgb_to_hex(palette[n]) ,ylabel='Absorbance [AU]', xlabel='Wavelength [nm]') 
                        else :
                            self.plot_panel.oplot(np.array(GenPanel.ready_spec[i].wl),                  
                                    np.array(GenPanel.ready_spec[i].A) ,                   
                                    linewidth=2,
                                    style='line',
                                    marker=None,markersize=0,
                                    label=i +" mass center = " +format(GenPanel.ready_spec[i].A.idxmax(), '.2f'), 
                                    color=rgb_to_hex(palette[n]) ,ylabel='Absorbance [AU]', xlabel='Wavelength [nm]') 
                n=n+1
            if len(GenPanel.ready_spec) > 30:
                self.plot_panel.plot_many_modified(datalist=list_toplot,ylabel='Absorbance [AU]', xlabel='Wavelength [nm]')
        elif typecorr == 'diff':            
            self.plot_panel.clear()

            n=0
            if self.GetParent().left_panel.tab1.mass_center_checkbox.GetValue() :
                self.plot_panel.oplot(np.array(GenPanel.diffspec.wl),                  
                      np.array(GenPanel.diffspec.A) ,                   
                      linewidth=2, 
                      style='line',
                      marker=None,markersize=0,
                      color=rgb_to_hex(palette[n]) ,ylabel='Absorbance [AU]', xlabel='Wavelength [nm]',
                      title = 'Difference spectrum') 
            #     self.plot_panel.axvline(centroids[i], color = palette[n], ls = '-.')
            else :
#                if self.GetParent().left_panel.tab1.scaling_checkbox.GetValue() :
                self.plot_panel.oplot(np.array(GenPanel.diffspec.wl),                  
                        np.array(GenPanel.diffspec.A) ,                   
                        linewidth=2, 
                        color=rgb_to_hex(palette[n]) ,ylabel='Absorbance [AU]', xlabel='Wavelength [nm]',
                        title = 'Difference spectrum') 
                    
            n=n+1
        elif typecorr == '2D_plot':
            self.plot_panel.clear()
            
        elif typecorr == 'time-trace':
            self.plot_panel.clear()
            wavelength = str(self.GetParent().left_panel.tab2.field_timetrace.GetValue())
            print('trying to print the time-trace at ' + wavelength + 'nm')
           
            n=0                          #this is just a counter for the palette, it's ugly as hell but hey, it works 
            # for i in GenPanel.list_spec.index:
            startfit=float(self.GetParent().left_panel.tab2.field_kinetic_start.GetValue())
            dose=float(self.GetParent().left_panel.tab2.abcisse_field.GetValue())
            self.plot_panel.oplot(     (np.array(GenPanel.list_spec.time_code) -startfit) * dose, #TODO fix that             , 
                    np.array(GenPanel.list_spec.Abs) ,
                    marker='o', markersize=4, color = 'blue', linewidth=0,
                    ylabel='Absorbance [AU]', xlabel='Time [s]', 
                    title = 'Absorbance at ' + wavelength + ' over time') 
            for i in GenPanel.list_spec.index :
                print(GenPanel.list_spec.loc[i, 'time_code'], GenPanel.list_spec.loc[i, 'Abs'])

        elif typecorr == 'kinetic_fit':
            self.plot_panel.clear()
            wavelength = str(self.GetParent().left_panel.tab2.field_timetrace.GetValue())
            print('trying to print the time-trace at ' + wavelength + 'nm')

            n=0                          #this is just a counter for the palette, it's ugly as hell but hey, it works 
            startfit=float(self.GetParent().left_panel.tab2.field_kinetic_start.GetValue())
            dose=float(self.GetParent().left_panel.tab2.abcisse_field.GetValue())
            self.plot_panel.oplot((np.array(GenPanel.list_spec.time_code) -startfit) * dose,
                                  np.array(GenPanel.list_spec.Abs) ,
                                  color = 'blue',
                                  marker='o', markersize=4, linewidth=0, alpha=0.5,
                                  style=None,
                                  ylabel='Absorbance [AU]', xlabel='Time [s]', 
                                  label= 'abs at ' + wavelength, legend_on=True) 
                  
            # print(GenPanel.list_spec.time_code, GenPanel.list_spec.Abs)
            print(self.GetParent().left_panel.tab2.model.x,self.GetParent().left_panel.tab2.model.y)
            self.plot_panel.oplot(np.array(self.GetParent().left_panel.tab2.model.x),
                    np.array(self.GetParent().left_panel.tab2.model.y),
                    linewidth=4, 
                    alpha = 0.5,
                    style='line',
                    marker=None,markersize=0,
                    label="modelled kinetic with tau="+format(self.GetParent().left_panel.tab2.para_kin_fit[-1], '.3f'),    
                    ylabel='Absorbance [AU]', xlabel='Time [s]', 
                    title = 'Absorbance at ' + wavelength + 'nm over time after laser pulse',
                    color='red', legend_on=True)
 
        elif typecorr == 'SVD' :
            self.plot_panel.clear()
            laser_blue=GenPanel.list_spec.laser_dent_blue.min()
            laser_red=GenPanel.list_spec.laser_dent_red.max()
            tokeep=[np.isnan(laser_blue)  or np.isnan(laser_red)  or x<laser_blue or x>laser_red for x in GenPanel.raw_spec[list(GenPanel.raw_spec.keys())[0]].wl[GenPanel.raw_spec[list(GenPanel.raw_spec.keys())[0]].wl.between(300,800)]]
    
            palette=sns.color_palette(palette='Spectral', n_colors=min(5,len(GenPanel.raw_spec)))   
            list_toplot=[]
            for i in range(0,min(5,len(GenPanel.raw_spec)-1)):
                if len(GenPanel.raw_spec) > 30:
                    list_toplot.append((np.array(GenPanel.raw_spec[list(GenPanel.raw_spec.keys())[0]].wl[GenPanel.raw_spec[list(GenPanel.raw_spec.keys())[0]].wl.between(300,800)][tokeep]),
                                        np.array(self.GetParent().left_panel.tab2.scaled_spec_lSV[:,i])))
                # tmp['SVn'+str(i)]=self.scaled_time_factors[i]
                self.plot_panel.oplot(np.array(GenPanel.raw_spec[list(GenPanel.raw_spec.keys())[0]].wl[GenPanel.raw_spec[list(GenPanel.raw_spec.keys())[0]].wl.between(300,800)][tokeep]),
                        np.array(self.GetParent().left_panel.tab2.scaled_spec_lSV[:,i]), 
                        linewidth=2,  
                        style='line',
                        marker=None,markersize=0,                  
                        label='SV n° ' + str(i) ,
                        title = 'left Singular Vectors',
                        xlabel = 'Wavelength [nm]', 
                        ylabel = 'Absorbance [AU]',
                        color=rgb_to_hex(palette[i]))      
            if len(GenPanel.raw_spec)> 30 :
                self.plot_panel.plot_many_modified(datalist=list_toplot, title = 'left Singular Vectors',
                                                   xlabel = 'Wavelength [nm]', 
                                                   ylabel = 'Absorbance [AU]')
        elif typecorr == 'diffserie' :
            palette=sns.color_palette(palette='Spectral', n_colors=len(GenPanel.raw_spec))
            self.plot_panel.clear()
            i=0
            list_toplot=[]
            for spec in GenPanel.diffserie : 
                if len(GenPanel.diffserie) > 30 : 
                    list_toplot.append((np.array(GenPanel.diffserie[spec].wl),np.array(GenPanel.diffserie[spec].A)))
                else:    
                    self.plot_panel.oplot(np.array(GenPanel.diffserie[spec].wl), 
                                          np.array(GenPanel.diffserie[spec].A),
                                          linewidth=2,   
                                          style='line',
                                          marker=None,markersize=0,                 
                                          label=spec + '- dark',
                                          title = 'Difference spectra series',
                                          xlabel = 'Wavelength [nm]', 
                                          ylabel = 'Absorbance [AU]',
                                          color=rgb_to_hex(palette[i+1])) 
                
                i+=1
            if len(GenPanel.diffserie) > 30 :
                self.plot_panel.plot_many_modified(datalist=list_toplot, 
                                                   title = 'Difference spectra series',
                                                   xlabel = 'Wavelength [nm]', 
                                                   ylabel = 'Absorbance [AU]',)
        elif typecorr == 'quality_plot': #TODO introduce a fix for this plot by creating a custom plotting function with a color list as one of the intakes
            self.plot_panel.clear()
            chosen_spectrum = self.GetParent().left_panel.tab3.selection
            print('plotting quality')  
            list_toplot=[]
            for i in GenPanel.raw_lamp[chosen_spectrum].index:
                list_toplot.append((np.array([GenPanel.raw_spec[chosen_spectrum].wl[i]]),                  
                                    np.array([GenPanel.raw_spec[chosen_spectrum].A[i]])))
            self.plot_panel.plot_quality(datalist=list_toplot,I0=np.array(GenPanel.raw_lamp[chosen_spectrum].I0))





class LeftPanel(wx.Panel):
    def __init__(self, parent):
        wx.Panel.__init__(self, parent)
        
        # Create notebook
        self.notebook = wx.Notebook(self)
        
        # Create tAbs
        self.tab1 = TabOne(self.notebook)
        self.tab2 = TabTwo(self.notebook)
        self.tab3 = TabThree(self.notebook)
        # Add tAbs to notebook
        self.notebook.AddPage(self.tab1, "Main")
        self.notebook.AddPage(self.tab2, "Kinetic")
        self.notebook.AddPage(self.tab3, "Expert Settings")
        # Set sizer for notebook
        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(self.notebook, 1, wx.EXPAND)
        self.SetSizer(sizer)

class TabOne(wx.Panel):
    def __init__(self, parent):
        wx.Panel.__init__(self, parent, style = wx.SUNKEN_BORDER)
        
        # Add previous content of LeftPanel here
        # Example:
# class LeftPanel(GenPanel):
    
    # def __init__(self, parent):
        # wx.Panel.__init__(self, parent, style = wx.SUNKEN_BORDER)
        self.button_openfile = wx.Button(self, label="Open File")
        self.button_openfile.Bind(wx.EVT_BUTTON, self.on_open_file)
        
        # checkbox for TR-icOS data
        #TODO change that to be a rollout menu that can iterate between icOS ; TRicOS and Fluo with a third option being 'user supplied'
        self.bigsizer_checkboxes = wx.BoxSizer(wx.VERTICAL)
        self.sizer_checkboxes = wx.BoxSizer(wx.HORIZONTAL)        
        self.TRicOS_checkbox = wx.CheckBox(self, label = 'TR-icOS data ?', style = wx.CHK_2STATE)
        # self.FLUO_checkbox = wx.CheckBox(self, label = 'Fluorescence data ?', style = wx.CHK_2STATE)
        # self.titlegend_checkbox = wx.CheckBox(self, label = 'Toggle off title/legend')
        self.sizer_checkboxes.Add(self.TRicOS_checkbox, flag=wx.ALL, border=3)
        # self.sizer_checkboxes.Add(self.FLUO_checkbox, flag=wx.ALL, border=3)
        #self.sizer_checkboxes.Add(self.titlegend_checkbox, flag=wx.ALL, border=3)
        
        # checkbox for Multi read files (such as that of the cailaidoscope)
        #TODO change that to be a rollout menu that can iterate between icOS ; TRicOS and Fluo with a third option being 'user supplied'
     
        self.Multiscan_checkbox = wx.CheckBox(self, label = 'Multi scan file ?', style = wx.CHK_2STATE)
        # self.FLUO_checkbox = wx.CheckBox(self, label = 'Fluorescence data ?', style = wx.CHK_2STATE)
        # self.titlegend_checkbox = wx.CheckBox(self, label = 'Toggle off title/legend')
        self.sizer_checkboxes.Add(self.Multiscan_checkbox, flag=wx.ALL, border=3)
        # self.sizer_checkboxes.Add(self.FLUO_checkbox, flag=wx.ALL, border=3)
        #self.sizer_checkboxes.Add(self.titlegend_checkbox, flag=wx.ALL, border=3)
        self.averaging_checkbox = wx.CheckBox(self, label= 'averaging', style = wx.CHK_2STATE)
        self.sizer_checkboxes.Add(self.averaging_checkbox, flag=wx.ALL, border=3)
        
        self.bigsizer_checkboxes.Add(self.sizer_checkboxes, 1, wx.ALIGN_CENTER)
        
        self.sizer_checkboxes_2 = wx.BoxSizer(wx.HORIZONTAL)    
        # scaling ?
        self.scaling_checkbox = wx.CheckBox(self, label = 'Scaling ?', style = wx.CHK_2STATE)
        # constchecksizer.Add(self.scaling_checkbox, wx.ALIGN_CENTER | wx.ALL)# border = 2)
        # smoothing ?
        self.smoothing_checkbox = wx.CheckBox(self, label = 'Smoothing ?', style = wx.CHK_2STATE)
        # constchecksizer.Add(self.smoothing_checkbox, wx.ALIGN_CENTER | wx.ALL)# border = 2)
        self.sizer_checkboxes_2.Add(self.scaling_checkbox, flag=wx.ALL, border=3)
        self.sizer_checkboxes_2.Add(self.smoothing_checkbox, flag=wx.ALL, border=3)
        self.bigsizer_checkboxes.Add(self.sizer_checkboxes_2, 1, wx.ALIGN_CENTER)
        
        
        
        # print raw data again
        self.button_rawdat = wx.Button(self, label="Back to raw data")
        self.button_rawdat.Bind(wx.EVT_BUTTON, self.backtoraw)
        # constant baseline correction 
        self.StaticBox_const = wx.StaticBox(self, label = "Constant Baseline")
        constboxsizer = wx.StaticBoxSizer(self.StaticBox_const, wx.VERTICAL)
        
        topeaksizer=wx.BoxSizer(wx.VERTICAL)
        self.label_topeak = wx.StaticText(self, label="wl of the peak of interest", style = wx.ALIGN_CENTER_HORIZONTAL)
        topeaksizer.Add(self.label_topeak, 1, wx.ALIGN_CENTER | wx.BOTTOM, border=0)#, border = 2)
        self.field_topeak = wx.TextCtrl(self, style = wx.TE_CENTER , value = '280')
        topeaksizer.Add(self.field_topeak, 1, wx.ALIGN_CENTER | wx.BOTTOM, border=0)#, border = 2)
        
        bluebasesizer=wx.BoxSizer(wx.VERTICAL)
        self.label_baseline_blue = wx.StaticText(self, label="Baseline blue-side Boundary", style = wx.ALIGN_CENTER_HORIZONTAL)
        bluebasesizer.Add(self.label_baseline_blue, 1, wx.ALIGN_CENTER | wx.BOTTOM, border=0)#, border = 2)
        self.field_baseline_blue = wx.TextCtrl(self, style = wx.TE_CENTER, value = '600')
        bluebasesizer.Add(self.field_baseline_blue, 1, wx.ALIGN_CENTER | wx.BOTTOM, border=0)#, border = 2)
        
        redbasesizer=wx.BoxSizer(wx.VERTICAL)
        self.label_baseline_red = wx.StaticText(self, label="Baseline red-side Boundary", style = wx.ALIGN_CENTER_HORIZONTAL)
        redbasesizer.Add(self.label_baseline_red, 1, wx.ALIGN_CENTER | wx.BOTTOM, border=0)#, border = 2)
        self.field_baseline_red = wx.TextCtrl(self, style = wx.TE_CENTER, value = '800')
        redbasesizer.Add(self.field_baseline_red, 1, wx.ALIGN_CENTER | wx.BOTTOM, border=0)#, border = 2)
        
        constcorrsizer=wx.BoxSizer(wx.HORIZONTAL)
        constcorrsizer.Add(topeaksizer, 1, wx.ALIGN_CENTER | wx.ALL, border = 3)
        constcorrsizer.Add(bluebasesizer, 1, wx.ALIGN_CENTER | wx.ALL, border = 3)
        constcorrsizer.Add(redbasesizer, 1, wx.ALIGN_CENTER | wx.ALL, border = 3)
        
        self.button_constancorr = wx.Button(self, label="Correct for constant baseline")
        self.button_constancorr.Bind(wx.EVT_BUTTON, self.on_constant_corr)
        
        # constchecksizer=wx.BoxSizer(wx.HORIZONTAL)
        # constboxsizer.Add(self.button_constancorr, wx.EXPAND | wx.ALL)# border = 2)
       
        
        #sizer block
        constboxsizer.Add(constcorrsizer, 1, wx.ALIGN_CENTER | wx.ALL, border = 0)
        constboxsizer.Add(self.button_constancorr, 1, wx.EXPAND | wx.ALL, border = 0)
        
        # constboxsizer.Add(constchecksizer, 0, wx.ALIGN_CENTER | wx.ALL, border = 2)
        
        
        #Scattering correction 
        self.StaticBox_scat = wx.StaticBox(self, label = "Scattering Baseline")
        scatboxsizer = wx.StaticBoxSizer(self.StaticBox_scat, wx.VERTICAL)
        self.button_scattercor = wx.Button(self, label="Correct for Scattering")
        self.button_scattercor.Bind(wx.EVT_BUTTON, self.on_scat_corr)
        
        bluenopeakizer=wx.BoxSizer(wx.VERTICAL)
        self.label_nopeak_blue = wx.StaticText(self, label="blue-side peakless", style = wx.ALIGN_CENTER_HORIZONTAL)
        bluenopeakizer.Add(self.label_nopeak_blue, 1, wx.ALIGN_CENTER | wx.ALL)
        self.field_nopeak_blue = wx.TextCtrl(self, style = wx.TE_CENTER)
        bluenopeakizer.Add(self.field_nopeak_blue, 1, wx.ALIGN_CENTER | wx.ALL)

        
        rednopeakizer=wx.BoxSizer(wx.VERTICAL)
        self.label_nopeak_red = wx.StaticText(self, label="red-side peakless", style = wx.ALIGN_CENTER_HORIZONTAL)
        rednopeakizer.Add(self.label_nopeak_red, 1, wx.ALIGN_CENTER | wx.BOTTOM, border=0)
        self.field_nopeak_red = wx.TextCtrl(self, style = wx.TE_CENTER)
        rednopeakizer.Add(self.field_nopeak_red, 1, wx.ALIGN_CENTER| wx.BOTTOM, border=0)
        
        leewaysizer=wx.BoxSizer(wx.VERTICAL)
        self.label_leeway_factor = wx.StaticText(self, label="expected OD in blue", style = wx.ALIGN_CENTER_HORIZONTAL)
        leewaysizer.Add(self.label_leeway_factor, 1, wx.ALIGN_CENTER | wx.BOTTOM, border=0)
        self.field_leeway_factor = wx.TextCtrl(self, style = wx.TE_CENTER)
        leewaysizer.Add(self.field_leeway_factor, 1, wx.ALIGN_CENTER | wx.BOTTOM, border=0)
        
        # diagnostic plots ?
        self.diagplots_checkbox = wx.CheckBox(self, label = 'no diagnostic plots ?', style = wx.CHK_2STATE)
        
        #divergences
        self.box_div = wx.StaticBox(self, label = 'Segment divergences')
        divboxsizer = wx.StaticBoxSizer(self.box_div, wx.HORIZONTAL)
        #UV        
        self.labelUV = wx.StaticText(self, label = 'UV', style = wx.ALIGN_CENTER_HORIZONTAL)
        self.field_weighUV = wx.TextCtrl(self, value = '1', style = wx.TE_CENTER)
        UVsizer = wx.BoxSizer(wx.VERTICAL)
        UVsizer.Add(self.labelUV, 1, wx.ALIGN_CENTER | wx.BOTTOM, border=0)
        UVsizer.Add(self.field_weighUV, 1, wx.ALIGN_CENTER | wx.TOP, border=3)
        divboxsizer.Add(UVsizer, 1, wx.ALIGN_CENTER, border = 2)
        
        #peakless        
        self.labelpeakless = wx.StaticText(self, label = 'peakless', style = wx.ALIGN_CENTER_HORIZONTAL)
        self.field_weighpeakless = wx.TextCtrl(self, value = '1', style = wx.TE_CENTER)
        peaklesssizer = wx.BoxSizer(wx.VERTICAL)
        peaklesssizer.Add(self.labelpeakless, 1, wx.ALIGN_CENTER | wx.BOTTOM, border=0)
        peaklesssizer.Add(self.field_weighpeakless, 1, wx.ALIGN_CENTER | wx.BOTTOM, border=3)
        divboxsizer.Add(peaklesssizer, 1, wx.ALIGN_CENTER, border = 2)
        
        #baseline        
        self.labelbaseline = wx.StaticText(self, label = 'baseline', style = wx.ALIGN_CENTER_HORIZONTAL)
        self.field_weighbaseline = wx.TextCtrl(self, value = '1', style = wx.TE_CENTER)
        baselinesizer = wx.BoxSizer(wx.VERTICAL)
        baselinesizer.Add(self.labelbaseline, 1, wx.ALIGN_CENTER | wx.BOTTOM, border=0)
        baselinesizer.Add(self.field_weighbaseline, 1, wx.ALIGN_CENTER | wx.BOTTOM, border=3)
        divboxsizer.Add(baselinesizer, 1, wx.ALIGN_CENTER, border = 2)
        
        corrscatsizer=wx.BoxSizer(wx.HORIZONTAL)
        
        corrscatsizer.Add(self.button_scattercor, 1, wx.EXPAND | wx.ALL, border = 2)
        corrscatsizer.Add(self.diagplots_checkbox, 1, wx.ALIGN_CENTER)
        scatboxsizer.Add(corrscatsizer, 1, wx.ALIGN_CENTER | wx.ALL, border = 0)
        
        fieldscatsizer=wx.BoxSizer(wx.HORIZONTAL)
        fieldscatsizer.Add(bluenopeakizer, 1, wx.ALIGN_CENTER | wx.ALL, border = 3)
        fieldscatsizer.Add(rednopeakizer, 1, wx.ALIGN_CENTER | wx.ALL, border = 3)
        fieldscatsizer.Add(leewaysizer, 1, wx.ALIGN_CENTER | wx.ALL, border = 3)
        scatboxsizer.Add(fieldscatsizer, 1, wx.ALIGN_CENTER | wx.ALL, border = 0)
        
        
        # scatboxsizer.AddSpacer(5)
        scatboxsizer.Add(divboxsizer, 1, wx.EXPAND, border = 5)
        
                
        # difference spectra
        self.button_diffspec = wx.Button(self, label = 'calculate difference spectrum')
        self.button_diffspec.Bind(wx.EVT_BUTTON, self.on_diff_spec)
        
        
        # mass center
        self.mass_center_checkbox = wx.CheckBox(self, label = 'Mass center calculation ?', style = wx.CHK_2STATE)        
        
        # remove a spec
        self.button_drop_spec = wx.Button(self, label='Remove a spectrum')
        self.button_drop_spec.Bind(wx.EVT_BUTTON, self.on_drop_spec)
        
        # kinetics 
        # self.field_timetrace = wx.TextCtrl(self, value = '280', style = wx.TE_CENTER)
        # self.label_timetrace = wx.StaticText(self, label = 'Kinetics', style = wx.ALIGN_CENTER_HORIZONTAL)
        
        # self.button_timetrace = wx.Button(self, label = 'Time-trace')
        # self.button_timetrace.Bind(wx.EVT_BUTTON, self.on_timetrace)
        
        # save
        self.button_save = wx.Button(self, label="Save figure and spectra")
        self.button_save.Bind(wx.EVT_BUTTON, self.on_save)
        
        # Add widgets to the right panel sizer
        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(self.button_openfile, 1, wx.EXPAND | wx.ALL, border = 2)
        sizer.Add(self.bigsizer_checkboxes, 1, wx.ALIGN_CENTER)
        # sizer.Add(self.FLUO_checkbox, 1, wx.ALIGN_CENTER)
        # self.sizer_checkboxes
        sizer.Add(self.button_rawdat, 1, wx.EXPAND | wx.ALL, border = 2)
        sizer.Add(constboxsizer, 1, wx.EXPAND, border = 2)
        
        sizer.Add(scatboxsizer, 1, wx.EXPAND, border = 5)
        sizer.Add(self.button_diffspec, 1, wx.EXPAND | wx.ALL, border = 2)
        sizer.Add(self.mass_center_checkbox, 1, wx.ALIGN_CENTER)
        sizer.Add(self.button_drop_spec, 1, wx.EXPAND | wx.ALL, border = 2)
        # sizer.Add(self.label_timetrace, 0, wx.ALIGN_CENTER, border = 0)
        # sizer.Add(self.field_timetrace, 0, wx.ALIGN_CENTER | wx.ALL, border = 0)
          
        # sizer.Add(self.button_timetrace, 1, wx.EXPAND | wx.ALL, border = 2)
        sizer.Add(self.button_save, 1, wx.EXPAND | wx.ALL, border = 2)
        self.SetSizer(sizer)
        # self.SetBackgroundColour('grey') 
        
        
    def on_open_file(self, event):
        self.typecorr = 'raw'
        if platform.system() == 'Windows' :
            dirsep='\\'
        else:# or platform.system() == 'MacOS'
            dirsep='/'
        if self.GetParent().GetParent().tab1.TRicOS_checkbox.GetValue() :
            file_chooser = FileChooser(self, "Do you want to average the spectra ?", 1, ['Averaging', 'Open a series'])
            if file_chooser.ShowModal() == wx.ID_OK:
                self.avg=file_chooser.check_list_box.GetCheckedStrings()[0]
                print(self.avg)
            if self.avg == 'Averaging':
                wildcard = "TXT files (*.txt)|*.txt|All files (*.*)|*.*"
                dialog = wx.FileDialog(self, "Choose one or several files", wildcard=wildcard, style=wx.FD_OPEN | wx.FD_MULTIPLE)
                toaverage=[]
                if dialog.ShowModal() == wx.ID_OK:
                    file_paths = dialog.GetPaths()
                    
                    for file_path in file_paths:
                        pathtospec=''
                        for i in file_path.split(dirsep)[0:-1]:
                            pathtospec+=i+dirsep
                        tmpname = file_path.split(dirsep)[-1]
                        # print(pathtospec)
                        # print(tmpname)
                        if re.search(r'\d+ms', tmpname) :
                            name_correct=tmpname.replace(max(re.findall(r'\d+ms', file_path), key = len), max(re.findall(r'\d+ms', file_path), key = len)[0:-2] + '000us')
                            os.rename(file_path, pathtospec + name_correct)
                            file_path=pathtospec + name_correct
                        elif re.search(r'\d+s', tmpname) and not re.search(r'\d+ms', tmpname) and not re.search(r'\d+us', tmpname): 
                            name_correct=tmpname.replace(max(re.findall(r'\d+s', file_path), key = len), max(re.findall(r'\d+s', file_path), key = len)[0:-1] + '000000us')
                            os.rename(file_path, pathtospec + name_correct)
                            file_path=pathtospec + name_correct
                        file_name = file_path.split(dirsep)[-1][0:-4]
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
                            isthereAbs=False
                            if len(GenPanel.raw_lamp[file_name].columns) == 5:
                                GenPanel.raw_lamp[file_name].columns=['wl','I', 'bgd', 'I0', 'A']
                                isthereAbs=True
                            elif len(GenPanel.raw_lamp[file_name].columns) == 4:
                                GenPanel.raw_lamp[file_name].columns=['wl','I', 'bgd', 'I0']
                            
                            
                            GenPanel.raw_lamp[file_name].index=GenPanel.raw_lamp[file_name].wl
                            
                            
                    average_signal=GenPanel.raw_lamp[list(GenPanel.raw_lamp.keys())[0]].copy()
                    average_signal.I=0
                    if isthereAbs:
                        average_signal.A=0
                    average_signal['wl']=floatize(average_signal.index)
                    # print(GenPanel.raw_lamp.keys())
                    
                    for nomfich in toaverage:
                        # print(nomfich)
                        for wavelength in average_signal.wl:
                            if average_signal.loc[wavelength,'I']==0:
                                average_signal.loc[wavelength,'I']=GenPanel.raw_lamp[nomfich].loc[wavelength,'I']
                            else:
                                average_signal.loc[wavelength,'I']=(average_signal.loc[wavelength,'I']+GenPanel.raw_lamp[nomfich].loc[wavelength,'I'])/2
                            if isthereAbs :
                                if average_signal.loc[wavelength,'A']==0:
                                    average_signal.loc[wavelength,'A']=GenPanel.raw_lamp[nomfich].loc[wavelength,'A']
                                else:
                                    average_signal.loc[wavelength,'A']=(average_signal.loc[wavelength,'A']+GenPanel.raw_lamp[nomfich].loc[wavelength,'A'])/2
                    # print(average_signal)
                    avgname=toaverage[0]#''.join(toaverage)
                    GenPanel.raw_spec[avgname]=Absorbance(average_signal.copy())
                            # print(GenPanel.raw_lamp[file_name].columns)
                    # print(f"File '{avgname}' added spectra list with data: {GenPanel.raw_spec[avgname].A}")
                    GenPanel.list_spec.loc[avgname,'file_name']=avgname
                    if 'dark' in avgname :
                        GenPanel.list_spec.loc[avgname,'time_code']=1
                    elif re.search(r'__\d+__', avgname):
                        print('THERE WE GO')
                        GenPanel.list_spec.loc[avgname,'time_code']=int(re.search(r'__\d+__', avgname)[0].replace('_',''))
                    else :
                        # GenPanel.list_spec.loc[avgname,'time_code']=int(max(re.findall(r'\d+us', avgname), key = len)[0:-2])#longest_digit_sequence(file_name)
                        try:
                            GenPanel.list_spec.loc[avgname,'time_code'] = int(max(re.findall(r'\d+us', avgname), key=len)[0:-2])
                        except (ValueError, IndexError):
                            GenPanel.list_spec.loc[avgname,'time_code'] = 0
                    GenPanel.list_spec.loc[avgname,'Abs']=GenPanel.raw_spec[avgname].loc[min(GenPanel.raw_spec[avgname]['wl'], key=lambda x: abs(x - 280)),'A']
                    GenPanel.list_spec.loc[avgname,'laser_dent_blue']=np.nan
                    GenPanel.list_spec.loc[avgname, 'laser_dent_red']=np.nan
                    self.update_right_panel('raw')
                dialog.Destroy()
            elif self.avg == 'Open a series':
                print('opening a series')
                wildcard = "TXT files (*.txt)|*.txt|All files (*.*)|*.*"
                dialog = wx.FileDialog(self, "Choose one or several files", wildcard=wildcard, style=wx.FD_OPEN | wx.FD_MULTIPLE)
                toaverage=[]
                if dialog.ShowModal() == wx.ID_OK:
                    file_paths = dialog.GetPaths()
                    
                    for file_path in file_paths:
                        pathtospec=''
                        for i in file_path.split(dirsep)[0:-1]:
                            pathtospec+=i+dirsep
                        tmpname = file_path.split(dirsep)[-1]
                        # print(pathtospec)
                        # print(tmpname)
                        if re.search(r'\d+ms', tmpname) :
                            name_correct=tmpname.replace(max(re.findall(r'\d+ms', file_path), key = len), max(re.findall(r'\d+ms', file_path), key = len)[0:-2] + '000us')
                            os.rename(file_path, pathtospec + name_correct)
                            file_path=pathtospec + name_correct
                        elif re.search(r'\d+s', tmpname) and not re.search(r'\d+ms', tmpname) and not re.search(r'\d+us', tmpname): 
                            name_correct=tmpname.replace(max(re.findall(r'\d+s', file_path), key = len), max(re.findall(r'\d+s', file_path), key = len)[0:-1] + '000000us')
                            os.rename(file_path, pathtospec + name_correct)
                            file_path=pathtospec + name_correct
                            
                            
                        file_name = file_path.split(dirsep)[-1][0:-4]
                        # print(file_name)
                        if file_path[-4:] == '.txt':
                            # toaverage.append(file_name)
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
                            isthereAbs=False
                            if len(GenPanel.raw_lamp[file_name].columns) == 5:
                                GenPanel.raw_lamp[file_name].columns=['wl','I', 'bgd', 'I0', 'A']
                                isthereAbs=True
                            elif len(GenPanel.raw_lamp[file_name].columns) == 4:
                                GenPanel.raw_lamp[file_name].columns=['wl','I', 'bgd', 'I0']
                            
                            
                            GenPanel.raw_lamp[file_name].index=GenPanel.raw_lamp[file_name].wl
                            GenPanel.raw_spec[file_name]=Absorbance(GenPanel.raw_lamp[file_name].copy())
                            # print(f"File '{file_name}' added spectra list with data: {GenPanel.raw_spec[file_name].A}")
                            GenPanel.list_spec.loc[file_name,'file_name']=file_name
                            if 'dark' in file_name :
                                GenPanel.list_spec.loc[file_name,'time_code']=1
                            elif re.search(r'__\d+__', file_name):
                                print('THERE WE GO')
                                GenPanel.list_spec.loc[file_name,'time_code']=int(re.search(r'__\d+__', file_name)[0].replace('_',''))
                            else :
                                # GenPanel.list_spec.loc[file_name,'time_code']=int(max(re.findall(r'\d+us', file_name), key = len)[0:-2])#longest_digit_sequence(file_name)
                                try:
                                    GenPanel.list_spec.loc[file_name,'time_code'] = int(max(re.findall(r'\d+us', file_name), key=len)[0:-2])
                                except (ValueError, IndexError):
                                    GenPanel.list_spec.loc[file_name,'time_code'] = 0
                            GenPanel.list_spec.loc[file_name,'Abs']=GenPanel.raw_spec[file_name].loc[min(GenPanel.raw_spec[file_name]['wl'], key=lambda x: abs(x - 280)),'A']
                            GenPanel.list_spec.loc[file_name,'laser_dent_blue']=np.nan
                            GenPanel.list_spec.loc[file_name, 'laser_dent_red']=np.nan
                        
                    
                    self.update_right_panel('raw')
                dialog.Destroy()
        elif self.GetParent().GetParent().tab1.Multiscan_checkbox.GetValue() :
            wildcard = "TXT files (*.txt)|*.txt|(*.TXT)|*.TXT|All files (*.*)|*.*"
            dialog = wx.FileDialog(self, "Choose one file", wildcard=wildcard, style=wx.FD_OPEN | wx.FD_MULTIPLE)
            if dialog.ShowModal() == wx.ID_OK:
                file_paths = dialog.GetPaths()
                for file_path in file_paths:
                    pathtospec=''
                    # for i in file_path.split(dirsep)[0:-1]:
                    #     pathtospec+=i+dirsep
                    # filename_raw = file_path.split(dirsep)[-1]
                    
                    
                    GenPanel.raw_spec, tmpstamps = multiscan_opener(file_path)
                    
                    # read the time-stamp for each spectrum 
                    
                    
                    for spec in  GenPanel.raw_spec:
                            GenPanel.list_spec.loc[spec,'file_name']=spec
                            # print(int(max(re.findall(r'pH\d+', name_correct), key = len)[2:]))
                            GenPanel.list_spec.loc[spec,'Abs']=GenPanel.raw_spec[spec].loc[min(GenPanel.raw_spec[spec]['wl'], key=lambda x: abs(x - 280)),'A']
                            GenPanel.list_spec.loc[spec,'laser_dent_blue']=np.nan
                            GenPanel.list_spec.loc[spec, 'laser_dent_red']=np.nan
                            GenPanel.list_spec.loc[spec,'time_code']=tmpstamps[spec][0]
                    
                    print(f"File '{spec}' added to dictionary with data: {GenPanel.raw_spec[spec].A}")
                # print(GenPanel.list_spec)
                self.update_right_panel('raw')
            dialog.Destroy()
       
        else :
            # self.typecorr = 'raw'
            wildcard = "TXT files (*.txt)|*.txt|All files (*.*)|*.*"
            dialog = wx.FileDialog(self, "Choose one or several files", wildcard=wildcard, style=wx.FD_OPEN | wx.FD_MULTIPLE)
            if dialog.ShowModal() == wx.ID_OK:
                file_paths = dialog.GetPaths()
                if self.GetParent().GetParent().tab1.averaging_checkbox.GetValue():
                    print('averaging spectra')
                    toaverage={}
                    for file_path in file_paths:
                         pathtospec=''
                        
                         tmpname = file_path.split(dirsep)[-1]
                         name_correct = file_path.split(dirsep)[-1][0:-4]
                         toaverage[name_correct]=universal_opener(file_path)
                    #define avg name
                    name_correct=name_correct+'_avg' #so that the avg can be compared to all spectra without confusion
                    #add averaged spectrum to raw_spec
                    print(toaverage.keys())
                    tmp=toaverage[list(toaverage.keys())[0]]
                    tmp.A=0
                    for spec in toaverage:
                        tmp.A+=toaverage[spec].A
                    tmp.A=tmp.A/len(toaverage.keys())
                    GenPanel.raw_spec[name_correct]=tmp
                    print(GenPanel.raw_spec[name_correct])
                    GenPanel.list_spec.loc[name_correct,'file_name']=name_correct
                    if 'dark' in name_correct :
                        GenPanel.list_spec.loc[name_correct,'time_code']=1
                    elif re.search(r'__\d+__', name_correct):
                        print('THERE WE GO')
                        GenPanel.list_spec.loc[name_correct,'time_code']=int(re.search(r'__\d+__', name_correct)[0].replace('_',''))
                    else :
                        try :
                            GenPanel.list_spec.loc[name_correct,'time_code']=int(max(re.findall(r'\d+us', name_correct), key = len)[0:-2])
                        except ValueError :
                            try : 
                                GenPanel.list_spec.loc[name_correct,'time_code']=int(max(re.findall(r'pH\d+', name_correct), key = len)[2:])
                            except ValueError :
                                try : 
                                    GenPanel.list_spec.loc[name_correct,'time_code']=int(max(re.findall(r'__\d+__', name_correct), key = len)[2:-2])
                                except :
                                    try : 
                                        GenPanel.list_spec.loc[name_correct,'time_code']=int(max(re.findall(r'\d+', name_correct), key = len))
                                    except :
                                        GenPanel.list_spec.loc[name_correct,'time_code']=0
                        
                            # print(int(max(re.findall(r'pH\d+', name_correct), key = len)[2:]))
                    GenPanel.list_spec.loc[name_correct,'Abs']=GenPanel.raw_spec[name_correct].loc[min(GenPanel.raw_spec[name_correct]['wl'], key=lambda x: abs(x - 280)),'A']
                    GenPanel.list_spec.loc[name_correct,'laser_dent_blue']=np.nan
                    GenPanel.list_spec.loc[name_correct, 'laser_dent_red']=np.nan
                    print(f"File '{name_correct}' added to dictionary with data: {GenPanel.raw_spec[name_correct].A}")
                else:
                     for file_path in file_paths:
                         pathtospec=''
                         for i in file_path.split(dirsep)[0:-1]:
                             pathtospec+=i+dirsep
                         tmpname = file_path.split(dirsep)[-1]
                         # print(pathtospec)
                         # print(tmpname)
                         if re.search(r'\d+ms', tmpname) :
                             name_correct=tmpname.replace(max(re.findall(r'\d+ms', file_path), key = len), max(re.findall(r'\d+ms', file_path), key = len)[0:-2] + '000us')
                             os.rename(file_path, pathtospec + name_correct)
                             file_path=pathtospec + name_correct
                         elif re.search(r'\d+s', tmpname) and not re.search(r'\d+ms', tmpname) and not re.search(r'\d+us', tmpname): 
                             name_correct=tmpname.replace(max(re.findall(r'\d+s', file_path), key = len), max(re.findall(r'\d+s', file_path), key = len)[0:-1] + '000000us')
                             os.rename(file_path, pathtospec + name_correct)
                             file_path=pathtospec + name_correct
                         
                         name_correct = file_path.split(dirsep)[-1][0:-4]
                         # print(name_correct)
                         if file_path[-4:] == '.txt' or file_path[-4:] == '.asc' or file_path[-4:] == '.csv':
                             GenPanel.raw_spec[name_correct] = universal_opener(file_path)
                             # GenPanel.raw_spec[name_correct].index=GenPanel.raw_spec[name_correct].wl
                             print(GenPanel.raw_spec[name_correct])
                             GenPanel.list_spec.loc[name_correct,'file_name']=name_correct
                         if 'dark' in name_correct :
                             GenPanel.list_spec.loc[name_correct,'time_code']=1
                         elif re.search(r'__\d+__', name_correct):
                             print('THERE WE GO')
                             GenPanel.list_spec.loc[name_correct,'time_code']=int(re.search(r'__\d+__', name_correct)[0].replace('_',''))
                         else :
                             try :
                                 GenPanel.list_spec.loc[name_correct,'time_code']=int(max(re.findall(r'\d+us', name_correct), key = len)[0:-2])
                             except ValueError :
                                 try : 
                                     GenPanel.list_spec.loc[name_correct,'time_code']=int(max(re.findall(r'pH\d+', name_correct), key = len)[2:])
                                 except ValueError :
                                     try : 
                                         GenPanel.list_spec.loc[name_correct,'time_code']=int(max(re.findall(r'__\d+__', name_correct), key = len)[2:-2])
                                     except :
                                         try : 
                                             GenPanel.list_spec.loc[name_correct,'time_code']=int(max(re.findall(r'\d+', name_correct), key = len))
                                         except :
                                             GenPanel.list_spec.loc[name_correct,'time_code']=0
                             
                                 # print(int(max(re.findall(r'pH\d+', name_correct), key = len)[2:]))
                         GenPanel.list_spec.loc[name_correct,'Abs']=GenPanel.raw_spec[name_correct].loc[min(GenPanel.raw_spec[name_correct]['wl'], key=lambda x: abs(x - 280)),'A']
                         GenPanel.list_spec.loc[name_correct,'laser_dent_blue']=np.nan
                         GenPanel.list_spec.loc[name_correct, 'laser_dent_red']=np.nan
                         
                         
                         print(f"File '{name_correct}' added to dictionary with data: {GenPanel.raw_spec[name_correct].A}")
                # print(GenPanel.list_spec)
                self.update_right_panel('raw')
            dialog.Destroy()
        GenPanel.list_spec.sort_values(by = ['time_code'], axis=0, ascending=True, inplace=True)
        print(GenPanel.list_spec.time_code)
        # Plot the DataFrame
        
        
        
    def on_constant_corr(self, event):
        #TODO need to fix the scaling
        self.typecorr = 'const'
        baseline_blue = float(self.field_baseline_blue.GetValue())
        baseline_red = float(self.field_baseline_red.GetValue())
        if self.GetParent().GetParent().tab1.scaling_checkbox.GetValue() :  
            scaling_top = float(self.field_topeak.GetValue())
        for i in GenPanel.raw_spec:
            segmentend=GenPanel.raw_spec[i].wl.between(baseline_blue,baseline_red, inclusive='both')
            tmp=GenPanel.raw_spec[i].copy()
            tmp.A-=mean(GenPanel.raw_spec[i].A[segmentend])
            if self.GetParent().GetParent().tab1.scaling_checkbox.GetValue() :
                if scaling_top == 0 : 
                    tmp_scaling_top=float(tmp.A[tmp.wl.between(300,800,inclusive='both')].idxmax())
                    print(tmp_scaling_top)
                    tmp.A*=1/tmp.A[tmp.wl.between(tmp_scaling_top-5,tmp_scaling_top+5,inclusive='both')].mean()
                else:
                    tmp.A*=1/tmp.A[tmp.wl.between(scaling_top-5,scaling_top+5,inclusive='both')].mean()
            if self.GetParent().GetParent().tab1.smoothing_checkbox.GetValue() :
                if GenPanel.smoothing == 'savgol':
                    tmp.A=signal.savgol_filter(x=tmp.A.copy(),     #This is the smoothing function, it takes in imput the y-axis data directly and fits a polynom on each section of the data at a time
                                                  window_length=int(self.GetParent().GetParent().tab3.smooth_window_field.GetValue()),  #This defines the section, longer sections means smoother data but also bigger imprecision self.GetParent().left_panel.tab3.smooth_window_field.GetValue()
                                                  polyorder=3)       #The order of the polynom, more degree = less smooth, more precise (and more ressource expensive)
                elif GenPanel.smoothing == 'rolling':
                    tmp.A = tmp.A.rolling(window=int(self.GetParent().GetParent().tab3.smooth_window_field.GetValue())).mean()
            GenPanel.const_spec[i]=tmp.copy()
            GenPanel.const_spec[i].index=GenPanel.raw_spec[i].wl
            print(f"Spectrum '{i}' corrected: {GenPanel.const_spec[i].A}")
        self.update_right_panel(self.typecorr)
        
    def on_scat_corr(self, event):
        self.typecorr = 'ready'
        baseline_blue = float(self.field_baseline_blue.GetValue())
        baseline_red = float(self.field_baseline_red.GetValue())
        leewayfac= float(self.field_leeway_factor.GetValue())
        if self.GetParent().GetParent().tab1.scaling_checkbox.GetValue() :  
            scaling_top = float(self.field_topeak.GetValue())
        nopeak_blue = float(self.field_nopeak_blue.GetValue())
        nopeak_red = float(self.field_nopeak_red.GetValue())
        
        n=0
        # this plots each fitted baseline against the raw data, highlighting the chose segments
        for i in GenPanel.raw_spec :
            tmp=GenPanel.raw_spec[i].copy()
            if self.GetParent().GetParent().tab1.smoothing_checkbox.GetValue() :
                if GenPanel.smoothing == 'savgol':
                    tmp.A=signal.savgol_filter(x=tmp.A.copy(),     #This is the smoothing function, it takes in imput the y-axis data directly and fits a polynom on each section of the data at a time
                                                  window_length=int(self.GetParent().GetParent().tab3.smooth_window_field.GetValue()),  #This defines the section, longer sections means smoother data but also bigger imprecision
                                                  polyorder=3)       #The order of the polynom, more degree = less smooth, more precise (and more ressource expensive)
                elif GenPanel.smoothing == 'rolling':
                    tmp.A = tmp.A.rolling(window=int(self.GetParent().GetParent().tab3.smooth_window_field.GetValue())).mean()
            rightborn=GenPanel.raw_spec[i].A[GenPanel.raw_spec[i].wl.between(200,250)].idxmax()+20
            leftborn=GenPanel.raw_spec[i].A[GenPanel.raw_spec[i].wl.between(200,250)].idxmax()
            segment1 = GenPanel.raw_spec[i].wl.between(leftborn,rightborn, inclusive='both')
            segment2 = GenPanel.raw_spec[i].wl.between(nopeak_blue,nopeak_red, inclusive='both')
            segmentend=GenPanel.raw_spec[i].wl.between(baseline_blue,baseline_red, inclusive='both')
            segment=segment1+segment2+segmentend
            #peakless visible segment
            sigmafor3segment=[float(self.field_weighUV.GetValue()),float(self.field_weighpeakless.GetValue()),float(self.field_weighbaseline.GetValue())]
            forfit=tmp.copy()
            if self.GetParent().GetParent().tab1.scaling_checkbox.GetValue() :  
                forfit.A[segment2]-=leewayfac*forfit.A[scaling_top-10:scaling_top+10].max()
            else :
                forfit.A[segment2]-=leewayfac*forfit.A[310:800].max()
            x=forfit.wl[segment].copy()
            y=forfit.A[segment].copy()
            
            m=len(forfit.A[segment1])
            sigma=m*[sigmafor3segment[0]]
            m=len(forfit.A[segment2])
            sigma=sigma + m*[sigmafor3segment[1]]
            m=len(forfit.A[segmentend])
            sigma=sigma + m*[sigmafor3segment[2]]
            
            if GenPanel.correction == 'rayleigh':
                # initialParameters = np.array([1e9,1])
                para, pcov = sp.optimize.curve_fit(f=fct_baseline, xdata=x, ydata=y, sigma=sigma)
                baseline=tmp.copy()
                baseline.A=fct_baseline(baseline.wl.copy(), *para)
            elif GenPanel.correction == 'full':
                # initialParameters = np.array([-100, 1e10, 11, 1, 6e+09])
                para, pcov = sp.optimize.curve_fit(f=full_correction, xdata=x, ydata=y, sigma=sigma)
                baseline=tmp.copy()
                baseline.A=full_correction(baseline.wl.copy(), *para)
            elif GenPanel.correction == 'custom':
                # initialParameters = np.array([1e9,1,4])
                para, pcov = sp.optimize.curve_fit(f=custom_correction, xdata=x, ydata=y, sigma=sigma)
                baseline=tmp.copy()
                baseline.A=custom_correction(baseline.wl.copy(), *para)
            elif GenPanel.correction == 'straight':
                # initialParameters = np.array([-9.98277459e-04, 4.47299554e+09, 4.0e+04 ,  1.79112630e+00])
                para, pcov = sp.optimize.curve_fit(f=straightforward_solution, xdata=x, ydata=y, sigma=sigma)
                baseline=tmp.copy()
                baseline.A=straightforward_solution(baseline.wl.copy(), *para)
            elif GenPanel.correction == 'lin_rayleigh':
                para_lin, pcov = sp.optimize.curve_fit(f=linbase, xdata=forfit.wl[segmentend].copy(), ydata=forfit.A[segmentend].copy(), sigma=len(forfit.A[segmentend]))
                forfit.A=forfit.A-linbase(forfit.wl.copy(), *para_lin)
                para, pcov = sp.optimize.curve_fit(f=fct_baseline, xdata=x, ydata=y, sigma=sigma)
                baseline=tmp.copy()
                baseline.A=fct_baseline(baseline.wl.copy(), *para)+linbase(baseline.wl.copy(), *para_lin)
                
            
            corrected=tmp.copy()
            corrected.A=tmp.A.copy()-baseline.A
            if self.GetParent().GetParent().tab1.scaling_checkbox.GetValue() :
                if scaling_top == 0 : 
                    scaling_top=corrected.A[corrected.wl.between(300,800,inclusive='both')].max()
                    corrected.A*=1/corrected.A[corrected.wl.between(scaling_top-5,scaling_top+5,inclusive='both')].mean()
                else:
                    corrected.A*=1/corrected.A[corrected.wl.between(scaling_top-5,scaling_top+5,inclusive='both')].mean()
                
            GenPanel.ready_spec[i]=corrected
            # tmp, baseline=baselinefitcorr_3seg_smooth(tmp,  segment1, segment2, segmentend, sigmafor3segment)
            if not self.GetParent().GetParent().tab1.diagplots_checkbox.GetValue() : 
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
            # print(GenPanel.list_spec.index)
            # sorting the selection from late to early
            sele_timecode=[]
            sele_timecode.append(GenPanel.list_spec.time_code[selections[0]])
            sele_timecode.append(GenPanel.list_spec.time_code[selections[1]])
            print(sele_timecode)
            self.sorted_selections = [x for _, x in sorted(zip(sele_timecode, selections), key=lambda pair: pair[0], reverse=True)]
            print(self.sorted_selections)
            
            print(self.typecorr)
            #add if statements to handle the diff spectra for all 
        if self.typecorr == 'raw':
            GenPanel.diffspec.wl = GenPanel.raw_spec[self.sorted_selections[0]].wl
            GenPanel.diffspec.index = GenPanel.diffspec.wl
            GenPanel.diffspec.A = GenPanel.raw_spec[self.sorted_selections[0]].A-GenPanel.raw_spec[self.sorted_selections[1]].A
            # print(GenPanel.diffspec[350:700])
        elif self.typecorr == 'const':
            GenPanel.diffspec.wl = GenPanel.const_spec[self.sorted_selections[0]].wl
            GenPanel.diffspec.index = GenPanel.diffspec.wl
            GenPanel.diffspec.A = GenPanel.const_spec[self.sorted_selections[0]].A-GenPanel.const_spec[self.sorted_selections[1]].A
            # print(GenPanel.diffspec[350:700])
        elif self.typecorr == 'ready':
            GenPanel.diffspec.wl = GenPanel.ready_spec[self.sorted_selections[0]].wl
            GenPanel.diffspec.index = GenPanel.diffspec.wl
            GenPanel.diffspec.A = GenPanel.ready_spec[self.sorted_selections[0]].A-GenPanel.ready_spec[self.sorted_selections[1]].A
            # print(GenPanel.diffspec[350:700])
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
                GenPanel.list_spec.drop(labels=i, inplace=True)   #[list(GenPanel.list_spec.file_name == i)]
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
                dirsep='\\'
            else:# or platform.system() == 'MacOS'
                dirsep='/'
            file_path=''
            for i in totalpath.split(dirsep)[:-1]:
                file_path+=i+dirsep
            print(file_path)
            file_name = totalpath.split(dirsep)[-1]
                
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
        # wavelength = str(self.field_timetrace.GetValue())
        # GenPanel.list_spec.to_csv(file_path + 'time-trace_' + wavelength + '_nm.csv', index=True)
        
        self.GetParent().GetParent().GetParent().right_panel.figure.savefig(file_path + file_name + ".svg", dpi=900 , transparent=True,bbox_inches='tight')
        self.GetParent().GetParent().GetParent().right_panel.figure.savefig(file_path + file_name + ".png", dpi=900, transparent=True,bbox_inches='tight')
        self.GetParent().GetParent().GetParent().right_panel.figure.savefig(file_path + file_name + ".pdf", dpi=900, transparent=True,bbox_inches='tight')
        print("Figure saved at: " + file_path + file_name + '.png')
        
        
    def update_right_panel(self, typecorr):
        if len(self.field_topeak.GetValue()) == 0:
            scaling_top=280
        else :
            scaling_top = float(self.field_topeak.GetValue())
        print(scaling_top)
        self.GetParent().GetParent().GetParent().right_panel.plot_data(typecorr, scaling_top)
        
        
    def backtoraw(self, event):
        self.typecorr='raw'
        if len(self.field_topeak.GetValue()) == 0:
            scaling_top=280
        else :
            scaling_top = float(self.field_topeak.GetValue())
        print(scaling_top)
        
        self.GetParent().GetParent().GetParent().right_panel.plot_data('raw', scaling_top)

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
        selections = self.check_list_box.GetCheckedItems()
        if self.numtodrop != None:
            self.btn_ok.Enable(len(selections) == self.numtodrop)
        else:
            self.btn_ok.Enable(len(selections)>0)

    def on_ok(self, event):
        self.EndModal(wx.ID_OK)
    
    

    def OnOption(self, event):
        selected_option = self.GetMenuBar().FindItemById(event.GetId()).GetLabel()
        wx.MessageBox("You selected " + selected_option)

class TabTwo(wx.Panel):
    def __init__(self, parent):
        wx.Panel.__init__(self, parent)
        # Add content for Tab 2 here
        # kinetics 
        sizer=wx.BoxSizer(wx.VERTICAL)
        
        self.field_timetrace = wx.TextCtrl(self, value = '280', style = wx.TE_CENTER)
        self.label_timetrace = wx.StaticText(self, label = 'Kinetics', style = wx.ALIGN_CENTER_HORIZONTAL)
        
        self.abcisse_field = wx.TextCtrl(self, value = '1', style = wx.TE_CENTER)
        self.abcisse_label = wx.StaticText(self, label = 'Time [μs]', style = wx.ALIGN_CENTER_HORIZONTAL)
        
        
        self.button_timetrace = wx.Button(self, label = 'Time-trace')
        self.button_timetrace.Bind(wx.EVT_BUTTON, self.on_timetrace)
        
        
        
        sizer_kinetics = wx.BoxSizer(wx.HORIZONTAL)
        sizer_timetrace = wx.BoxSizer(wx.VERTICAL)
        sizer_timetrace.Add(self.label_timetrace, 1, wx.ALL, border = 2)
        sizer_timetrace.Add(self.field_timetrace, 2, wx.ALL, border = 2)
        
        sizer_abcisse = wx.BoxSizer(wx.VERTICAL)
        sizer_abcisse.Add(self.abcisse_label, 1, wx.ALL, border = 2)
        sizer_abcisse.Add(self.abcisse_field, 2, wx.ALL, border = 2)
        
        sizer_kinetics.Add(sizer_timetrace,1,  wx.ALL, border = 2)
        sizer_kinetics.Add(sizer_abcisse,2,  wx.ALL, border = 2)
        # self.logscale_checkbox = wx.CheckBox(self, label = 'Log scale ?', style = wx.CHK_2STATE)
        # sizer_kinetics.Add(self.logscale_checkbox ,2, wx.ALL, border = 2)
        
        self.kintypebutton = wx.Button(self, label="Kinetic model")
        self.kintypebutton.Bind(wx.EVT_RIGHT_DOWN, self.OnContextMenu_kinetic)
        sizer_kinetics.Add(self.kintypebutton, 3, wx.EXPAND | wx.ALL, border = 2)
        self.options=['Monoexponential', 'Hill equation', 'Strict Monoexponential']
        self.kin_model_type = 'Monoexponential'
        
        
        sizer.Add(sizer_kinetics, 1, wx.EXPAND | wx.ALL, border = 2)
        sizer.Add(self.button_timetrace, 1, wx.EXPAND | wx.ALL, border = 2)
        
        
        
        self.button_diffserie = wx.Button(self, label = 'Difference spectra')
        self.button_diffserie.Bind(wx.EVT_BUTTON, self.on_diffserie)
        sizer.Add(self.button_diffserie,1, wx.EXPAND | wx.ALL, border = 2)
        
        #2D_plot
        self.button_2D_plot = wx.Button(self, label = '2D plot')
        self.button_2D_plot.Bind(wx.EVT_BUTTON, self.on_2D_plot)
        sizer.Add(self.button_2D_plot, 1, wx.EXPAND | wx.ALL, border = 2)

        #deuxDplot_labels
        
        # deuxDplot fit #TODO : make a vertical sizer for each of them and stack them horizontally. 
        self.label_deuxDplot_start = wx.StaticText(self, label = 'Blue side end 2D plot', style = wx.ALIGN_CENTER_HORIZONTAL)
        self.field_deuxDplot_start = wx.TextCtrl(self, value = '275', style = wx.TE_CENTER)
        
        self.label_deuxDplot_end = wx.StaticText(self, label = 'Red side end 2D plot', style = wx.ALIGN_CENTER_HORIZONTAL)
        self.field_deuxDplot_end = wx.TextCtrl(self, value = '750', style = wx.TE_CENTER)
        
        
        deuxDplot_label_sizer=wx.BoxSizer(wx.HORIZONTAL)
        deuxDplot_label_sizer.Add(self.label_deuxDplot_start, 1, wx.ALL, border=2)
        deuxDplot_label_sizer.Add(self.label_deuxDplot_end, 1, wx.ALL, border=2)
        # deuxDplot_label_sizer.Add(self.label_deuxDplot_rate, 1, wx.ALL, border=2)
        # deuxDplot_label_sizer.Add(self.field_deuxDplot_rate, 1, wx.ALL, border=2)
        
        sizer.Add(deuxDplot_label_sizer,1, wx.EXPAND | wx.ALL, border = 0)
        
        
        deuxDplot_field_sizer=wx.BoxSizer(wx.HORIZONTAL)
        deuxDplot_field_sizer.Add(self.field_deuxDplot_start, 1, wx.ALL, border=2)
        deuxDplot_field_sizer.Add(self.field_deuxDplot_end, 1, wx.ALL, border=2)
        # sizer.Add(deuxDplot_label_sizer,1, wx.EXPAND | wx.ALL, border = 1)
        sizer.Add(deuxDplot_field_sizer,1, wx.EXPAND | wx.ALL, border = 0)
        
        # kinetic fit #TODO : make a vertical sizer for each of them and stack them horizontally. 
        self.label_kinetic_start = wx.StaticText(self, label = 'Start', style = wx.ALIGN_CENTER_HORIZONTAL)
        self.field_kinetic_start = wx.TextCtrl(self, value = '10', style = wx.TE_CENTER)
        
        self.label_kinetic_end = wx.StaticText(self, label = 'End', style = wx.ALIGN_CENTER_HORIZONTAL)
        self.field_kinetic_end = wx.TextCtrl(self, value = '12000000', style = wx.TE_CENTER)
        
        
        
        
        kinetic_label_sizer=wx.BoxSizer(wx.HORIZONTAL)
        kinetic_label_sizer.Add(self.label_kinetic_start, 1, wx.ALL, border=2)
        kinetic_label_sizer.Add(self.label_kinetic_end, 1, wx.ALL, border=2)
        # kinetic_label_sizer.Add(self.label_kinetic_rate, 1, wx.ALL, border=2)
        # kinetic_label_sizer.Add(self.field_kinetic_rate, 1, wx.ALL, border=2)
        
        sizer.Add(kinetic_label_sizer,1, wx.EXPAND | wx.ALL, border = 0)
        
        
        kinetic_field_sizer=wx.BoxSizer(wx.HORIZONTAL)
        kinetic_field_sizer.Add(self.field_kinetic_start, 1, wx.ALL, border=2)
        kinetic_field_sizer.Add(self.field_kinetic_end, 1, wx.ALL, border=2)
        # sizer.Add(kinetic_label_sizer,1, wx.EXPAND | wx.ALL, border = 1)
        sizer.Add(kinetic_field_sizer,1, wx.EXPAND | wx.ALL, border = 0)
        
        #SVD
        self.button_SVD = wx.Button(self, label = 'Singular Value Decomposition')
        self.button_SVD.Bind(wx.EVT_BUTTON, self.on_SVD)
        sizer.Add(self.button_SVD, 1, wx.EXPAND | wx.ALL, border = 2)
        kin_par_sizer=wx.BoxSizer(wx.HORIZONTAL)
        
        kin_const_sizer=wx.BoxSizer(wx.VERTICAL)
        self.label_kinetic_constant = wx.StaticText(self, label = 'constant', style = wx.ALIGN_CENTER_HORIZONTAL)        
        self.field_kinetic_constant = wx.TextCtrl(self, style = wx.TE_CENTER)
        kin_const_sizer.Add(self.label_kinetic_constant, 1, wx.CENTER)
        kin_const_sizer.Add(self.field_kinetic_constant, 1, wx.CENTER)
        kin_par_sizer.Add(kin_const_sizer, 1, wx.CENTER)
        
        kin_scal_sizer=wx.BoxSizer(wx.VERTICAL)
        self.label_kinetic_scalar = wx.StaticText(self, label = 'scalar', style = wx.ALIGN_CENTER_HORIZONTAL)        
        self.field_kinetic_scalar = wx.TextCtrl(self, style = wx.TE_CENTER)
        kin_scal_sizer.Add(self.label_kinetic_scalar, 1, wx.CENTER)
        kin_scal_sizer.Add(self.field_kinetic_scalar, 1, wx.CENTER)
        kin_par_sizer.Add(kin_scal_sizer, 1, wx.CENTER)
        
        #insert the sizers in the 
        kin_rate_sizer=wx.BoxSizer(wx.VERTICAL)
        self.label_kinetic_rate = wx.StaticText(self, label = 'Rate', style = wx.ALIGN_CENTER_HORIZONTAL)
        self.field_kinetic_rate = wx.TextCtrl(self, style = wx.TE_CENTER)
        kin_rate_sizer.Add(self.label_kinetic_rate, 1, wx.CENTER)
        kin_rate_sizer.Add(self.field_kinetic_rate, 1, wx.CENTER)
        kin_par_sizer.Add(kin_rate_sizer, 1, wx.CENTER)
        
        sizer.Add(kin_par_sizer, 1, wx.EXPAND | wx.ALL, border = 2)
        
        
        self.kin_button = wx.Button(self, label = 'Kinetic fit')
        self.kin_button.Bind(wx.EVT_BUTTON, self.on_kinetic_fit)
        sizer.Add(self.kin_button, 1, wx.EXPAND  | wx.ALL, border = 2)
        
        
        # save
        self.button_save = wx.Button(self, label="Save figure and spectra")
        self.button_save.Bind(wx.EVT_BUTTON, self.on_save)
        sizer.Add(self.button_save, 1, wx.EXPAND | wx.ALL, border = 2)
        self.SetSizer(sizer)
        
        # self.logscale = self.logscale_checkbox.GetValue()
        
    def OnContextMenu_kinetic(self, event):
        menu=wx.Menu()
        
        for index, option in enumerate(self.options):
            item_id = wx.ID_HIGHEST + index
            item = menu.Append(item_id, option)
            self.Bind(wx.EVT_MENU, self.OnMenuSelect, item)
            
        self.PopupMenu(menu)
        menu.Destroy()
        
    def OnMenuSelect(self, event):
        item = event.GetId() - wx.ID_HIGHEST
        if 0 <= item < len(self.options):
            option = self.options[item]
            self.kin_model_type = option
            print("Chosen option:", self.kin_model_type)
        else:
            print("Error: Invalid option selected")
        
        
        # pass  # Add your content here
    def on_timetrace(self, event):
        #TODO : add the possibility of fitting a constant to the data points
        
        # if self.logscale_checkbox.GetValue():
        #     print('Logarithmic scale has been chosen')
        # else :
        #     print('Linear scale has been chosen')
        wavelength = float(self.field_timetrace.GetValue())
        print(wavelength)
        if self.GetParent().GetParent().tab1.typecorr == 'raw' :
            for i in GenPanel.list_spec.index :
                GenPanel.list_spec.loc[i, 'Abs'] =  GenPanel.raw_spec[i].loc[min(GenPanel.raw_spec[i]['wl'], key=lambda x: abs(x - wavelength)),'A']
        if self.GetParent().GetParent().tab1.typecorr == 'const' :
            for i in GenPanel.list_spec.index :
                GenPanel.list_spec.loc[i, 'Abs'] =  GenPanel.const_spec[i].loc[min(GenPanel.const_spec[i]['wl'], key=lambda x: abs(x - wavelength)),'A']
        if self.GetParent().GetParent().tab1.typecorr == 'ready' :
            for i in GenPanel.list_spec.index :
                GenPanel.list_spec.loc[i, 'Abs'] =  GenPanel.ready_spec[i].loc[min(GenPanel.ready_spec[i]['wl'], key=lambda x: abs(x - wavelength)),'A']
        print(GenPanel.list_spec)
        self.update_right_panel('time-trace')
    
    def on_kinetic_fit(self,event):
        # file_chooser = FileChooser(self, "Which model do you want to fit", 1, ['Monoexponential', 'Hill equation'])
        # if file_chooser.ShowModal() == wx.ID_OK:
        #     self.kin_model_type=file_chooser.check_list_box.GetCheckedStrings()[0]
        #     print(self.kin_model_type)
        startfit = float(self.field_kinetic_start.GetValue())
        endfit = float(self.field_kinetic_end.GetValue())
        # p0=[float(self.field_kinetic_constant.GetValue()),float(self.field_kinetic_scalar.GetValue()),float(self.field_kinetic_rate.GetValue())]
        
        
        
        # print('this is the intial value of the rate: ',str(p0))
        # rate0 = float(self.field_kinetic_rate.GetValue())
        x=(np.array(GenPanel.list_spec.time_code[GenPanel.list_spec.time_code.between(startfit,endfit)]) -startfit) * float(self.abcisse_field.GetValue()) #TODO fix that 
        y=np.array(GenPanel.list_spec.Abs[GenPanel.list_spec.time_code.between(startfit,endfit)])
        # print(x,y)
        #TODO decide whether we should add initial parameters to the fit or not. 
        
        self.model = pd.DataFrame(columns=['x','y'])
        # if self.logscale_checkbox.GetValue():
        #     self.model.x = np.geomspace(x.min(), x.max(), 1000)
        #     if self.kin_model_type == 'Monoexponential':
        #         self.model.y = fct_monoexp(np.geomspace(x.min(), x.max(), 1000), *self.para_kin_fit)
        #     elif self.kin_model_type == 'Hill equation':
        #         self.model.y = fct_Hill(np.geomspace(x.min(), x.max(), 1000), *self.para_kin_fit)
        # else : 
        self.model.x = np.linspace(x.min(), x.max(), 1000)
        
        if self.kin_model_type == 'Monoexponential':
            sigma = np.array(len(x)*[1])
            print([y[-1], y[0]-y[-1], -1/x[int(len(x)/2)]])
            self.para_kin_fit, pcov = sp.optimize.curve_fit(fct_monoexp, x,y, sigma = sigma)
            self.model.y = fct_monoexp(np.linspace(x.min(), x.max(), 1000), *self.para_kin_fit)
        elif self.kin_model_type == 'Strict Monoexponential':
            try:
                strict_constant = parse_float(self.field_kinetic_constant.GetValue())
                sigma = np.array(len(x)*[1])
                print([y[-1], y[0]-y[-1], -1/x[int(len(x)/2)]])
                print(f"strict y startpoint of {strict_constant} used")
                try:
                    strict_scalar=parse_float(self.field_kinetic_scalar.GetValue())
                    print(f"strict y endpoint of {strict_scalar} used")
                    def fct_monoexp_strict(x,tau): 
                        return(strict_constant + strict_scalar*(1-np.exp(-x/tau)))
                except :
                    def fct_monoexp_strict(x,b,tau): 
                        return(strict_constant + b*(1-np.exp(-x/tau)))
                self.para_kin_fit, pcov = sp.optimize.curve_fit(fct_monoexp_strict, x,y, sigma = sigma)
                self.model.y = fct_monoexp_strict(np.linspace(x.min(), x.max(), 1000), *self.para_kin_fit)
            except Exception:
                # Mark invalid and notify
                self.field_kinetic_constant.SetBackgroundColour("pink")
                self.field_kinetic_constant.SetFocus()
                self.field_kinetic_constant.Refresh()
                wx.MessageBox(
                    "Please enter a valid number (e.g., 12.3 or 12,3).",
                    "Invalid input",
                    wx.OK | wx.ICON_ERROR,
                )
                return
        elif self.kin_model_type == 'Hill equation':
            sigma = np.array(len(x)*[1])
            # p0=[y[0], y.max(), x.max()/2 ,-1/x.max()]
            self.para_kin_fit, pcov = sp.optimize.curve_fit(fct_Hill, x,y, sigma = sigma)
            self.model.y = fct_Hill(np.linspace(x.min(), x.max(), 1000), *self.para_kin_fit)
        #print(p0)
        print(self.para_kin_fit)
        
        self.update_right_panel('kinetic_fit')


    
    def on_diffserie(self, event):
        # n=len(GenPanel.raw_spec)-1
        # GenPanel.diffserie[]
        if self.GetParent().GetParent().tab1.typecorr == 'raw' :
            for spec in list(GenPanel.list_spec.file_name)[1:]: #sorting by time
                GenPanel.diffserie[spec]=GenPanel.raw_spec[spec].copy()
                GenPanel.diffserie[spec].A=GenPanel.raw_spec[spec]-GenPanel.raw_spec[list(GenPanel.list_spec.file_name)[0]].A #storing the difference spectrum
        if self.GetParent().GetParent().tab1.typecorr == 'const' :
            for spec in list(GenPanel.list_spec.file_name)[1:]: #sorting by time
                GenPanel.diffserie[spec]=GenPanel.const_spec[spec].copy()
                GenPanel.diffserie[spec].A=GenPanel.const_spec[spec].A-GenPanel.const_spec[list(GenPanel.list_spec.file_name)[0]].A #storing the difference spectrum
        if self.GetParent().GetParent().tab1.typecorr == 'ready' :
            for spec in list(GenPanel.list_spec.file_name)[1:]: #sorting by time
                GenPanel.diffserie[spec]=GenPanel.ready_spec[spec].copy()
                GenPanel.diffserie[spec].A=GenPanel.ready_spec[spec].A-GenPanel.ready_spec[list(GenPanel.list_spec.file_name)[0]].A #storing the difference spectrum
        self.update_right_panel('diffserie')
    def on_2D_plot(self, event):
        if self.GetParent().GetParent().tab1.typecorr == 'const' :
            test=[]
            dose=float(self.abcisse_field.GetValue())
            start = int(self.field_kinetic_start.GetValue())
            endfit = int(self.field_kinetic_end.GetValue())
            print(start,endfit)
            print(list(GenPanel.list_spec.file_name)[1:][start:endfit])
            for spec in list(GenPanel.list_spec.file_name)[1:][start:endfit]:
                # test.append(np.array(GenPanel.diffserie[spec]))
                test.append(np.array(GenPanel.const_spec[spec].A[GenPanel.const_spec[spec].wl.between(280,700)]))
            GenPanel.Z=np.transpose(np.array(test))
            
            # fig, ax = plt.subplots()  
            
            plt.imshow(GenPanel.Z, 
                       aspect='auto', 
                       cmap='rainbow', 
                       origin='lower', 
                       extent=(start*dose,endfit*dose,280,750))

            plt.colorbar(label='Absorbance')
            # plt.set_xlabel([199.56,994.94])
            plt.xlabel('Time [s]')  # Replace with your actual label
            plt.ylabel('Wavelength [nm]')  # Replace with your actual label
            # plt.title('UV-vis absorbtion spectrum of lysosyme over time')
            plt.show()
        
        
        
        self.update_right_panel('2D_plot')
        
    def on_SVD(self, event):
        # if GenPanel.list_spec.laser_blue.isnull().all():
            # tokeep_dark = 
        # if ~all([x == None for x in GenPanel.list_spec.laser_dent_blue]) :
        laser_blue=GenPanel.list_spec.laser_dent_blue.min()
        laser_red=GenPanel.list_spec.laser_dent_red.max()

        # print(laser_blue, laser_red)
        # print (laser_blue == None, laser_red == None)
        #BUG to SVD here : the tokeep is failing when None and m becomes 0
        tokeep_dark=[np.isnan(laser_blue)  or np.isnan(laser_red)  or x<laser_blue or x>laser_red for x in GenPanel.raw_spec[list(GenPanel.list_spec.file_name)[0]].wl[GenPanel.raw_spec[list(GenPanel.list_spec.file_name)[0]].wl.between(300,800)]]
        # print(tokeep_dark)
        if self.GetParent().GetParent().tab1.typecorr == 'raw' :
            i=0
            n=len(GenPanel.raw_spec)-1
            m=len(GenPanel.raw_spec[list(GenPanel.raw_spec.keys())[0]].wl[GenPanel.raw_spec[list(GenPanel.raw_spec.keys())[0]].wl.between(300,800)][tokeep_dark])
            print(n, m)
            A=np.zeros((m,n),dtype=np.float32)
            for spec in list(GenPanel.list_spec.file_name)[1:]: #sorting by time
                tokeep=[np.isnan(laser_blue)  or np.isnan(laser_red)  or x<laser_blue or x>laser_red for x in GenPanel.raw_spec[spec].wl[GenPanel.raw_spec[spec].wl.between(300,800)]]
                A[:,i] = GenPanel.raw_spec[spec].A[GenPanel.raw_spec[spec].wl.between(300,800)][tokeep]-GenPanel.raw_spec[list(GenPanel.list_spec.file_name)[0]].A[GenPanel.raw_spec[spec].wl.between(300,800)][tokeep_dark] #storing the difference spectrum
                i+=1

        elif self.GetParent().GetParent().tab1.typecorr == 'const' :
            i=0
            n=len(GenPanel.const_spec)-1
            m=len(GenPanel.const_spec[list(GenPanel.const_spec.keys())[0]].wl[GenPanel.const_spec[list(GenPanel.const_spec.keys())[0]].wl.between(300,800)][tokeep_dark])
            # print(n, m)
            A=np.zeros((m,n),dtype=np.float32)
            for spec in list(GenPanel.list_spec.file_name)[1:]: #sorting by time
                tokeep=[np.isnan(laser_blue)  or np.isnan(laser_red)  or x<laser_blue or x>laser_red for x in GenPanel.const_spec[spec].wl[GenPanel.const_spec[spec].wl.between(300,800)]]
                # print(tokeep)
                # print(len(tokeep))
                # print(len(GenPanel.const_spec[list(GenPanel.list_spec.file_name)[0]].A[GenPanel.const_spec[spec].wl.between(300,800)][tokeep]))
                # print(len(GenPanel.const_spec[spec].A[GenPanel.const_spec[spec].wl.between(300,800)][tokeep]))
                # print(len(tokeep))
                # print(len(GenPanel.const_spec[spec].A[GenPanel.const_spec[spec].wl.between(300,800)]))
                # print(len(tokeep_dark))
                # print(len(GenPanel.const_spec[list(GenPanel.list_spec.file_name)[0]].A[GenPanel.const_spec[list(GenPanel.list_spec.file_name)[0]].wl.between(300,800)]))
                A[:,i] = GenPanel.const_spec[spec].A[GenPanel.const_spec[spec].wl.between(300,800)][tokeep]-GenPanel.const_spec[list(GenPanel.list_spec.file_name)[0]].A[GenPanel.const_spec[list(GenPanel.list_spec.file_name)[0]].wl.between(300,800)][tokeep_dark] #storing the difference spectrum
                
                i+=1

        elif self.GetParent().GetParent().tab1.typecorr == 'ready' :
            i=0
            n=len(GenPanel.ready_spec)-1
            m=len(GenPanel.ready_spec[list(GenPanel.ready_spec.keys())[0]].wl[GenPanel.ready_spec[list(GenPanel.ready_spec.keys())[0]].wl.between(300,800)][tokeep_dark])
            
            A=np.zeros((m,n),dtype=np.float32)
            for spec in list(GenPanel.list_spec.file_name)[1:]: #sorting by time
                tokeep=[np.isnan(laser_blue)  or np.isnan(laser_red)  or x<laser_blue or x>laser_red for x in GenPanel.ready_spec[spec].wl[GenPanel.ready_spec[spec].wl.between(300,800)]]
                A[:,i] = GenPanel.ready_spec[spec].A[GenPanel.ready_spec[spec].wl.between(300,800)][tokeep]-GenPanel.ready_spec[list(GenPanel.list_spec.file_name)[0]].A[GenPanel.ready_spec[spec].wl.between(300,800)][tokeep_dark] #storing the difference spectrum
                i+=1

        U, S, VT = np.linalg.svd(A) 
        # print(S)
        # print(VT)
        sigmatrix=np.array(np.zeros(shape=(m,n)),dtype=np.float32)
        # nSV=['SV' + str(i) for i in range(n+1)]
        # rSV={'time':list(GenPanel.list_spec.index)}
        
        for i in range(0,len(S)):
            print(i,S[i])
            sigmatrix[i,i]=S[i]
        self.scaled_time_factors=(sigmatrix @ VT)[0:n,:]
        self.scaled_spec_lSV=np.matrix(U) @ np.matrix(sigmatrix)
        
        
        print(self.scaled_time_factors)
        print(self.scaled_spec_lSV)
        
        # print(GenPanel.list_spec.time_code[1:])
        # print(GenPanel.list_spec.time_code[1:])
        # print(self.scaled_time_factors)
        # fig, ax = plt.subplots()     
        # self.plot_panel.set_xlabel('time-point (µs, in log scale)', fontsize=35)  
        # self.plot_panel.xaxis.set_label_coords(x=0.5, y=-0.13)      
        # self.plot_panel.set_ylabel('Magnitude', fontsize=35)               
        # self.plot_panel.yaxis.set_label_coords(x=-0.12, y=0.5)       
        print(n, m)
        
        
        dose=float(self.abcisse_field.GetValue())
        
        
        
        palette=sns.color_palette(palette='Spectral', n_colors=min(5,len(self.scaled_time_factors)))   
        for i in range(0,min(5,len(self.scaled_time_factors))):
            wi.plot(dose*np.array(GenPanel.list_spec.time_code[1:]),np.array(self.scaled_time_factors[i]), 
                    marker='o',
                    linewidth=0, 
                    style = 'line',
                    markersize= 4,
                    label='SV n° ' + str(i),
                    color=rgb_to_hex(palette[i]),
                    xlabel = 'time-point (µs, in log scale)',
                    ylabel = 'Magnitude',
                    title = 'right Singular Vector')
            
        self.update_right_panel('SVD')
        
    def on_save(self, event):
        wildcard = "CSV files (*.csv)|*.csv|All files (*.*)|*.*"
        dialog = wx.FileDialog(self, "Save File(s)", wildcard=wildcard, style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT)
        if dialog.ShowModal() == wx.ID_OK:
            totalpath = dialog.GetPath()
            # file_path2 = file_path.split('/')[:-1]
            if platform.system() == 'Windows' :
                dirsep='\\'
            else:# or platform.system() == 'MacOS'
                dirsep='/'
            file_path=''
            for i in totalpath.split(dirsep)[:-1]:
                file_path+=i+dirsep
            print(file_path)
            file_name = totalpath.split(dirsep)[-1]
                
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
        wavelength = str(self.GetParent().GetParent().tab2.field_timetrace.GetValue())
        GenPanel.list_spec.to_csv(file_path + 'time-trace_' + wavelength + '_nm.csv', index=True)
        
        # try:
        tmp={}
        for i in range(len(self.scaled_time_factors)):
            tmp['rSV'+str(i)]=self.scaled_time_factors[i]
        pd.DataFrame(data=tmp,index=GenPanel.list_spec.index[1:]).to_csv(file_path + file_name + '_rSV.csv',index=True)
        
        
        n=len(GenPanel.raw_spec)-1
        laser_blue = GenPanel.list_spec.laser_dent_blue.min()
        laser_red = GenPanel.list_spec.laser_dent_red.max()
        tokeep_dark=[np.isnan(laser_blue)  or np.isnan(laser_red)  or x<laser_blue or x>laser_red for x in GenPanel.raw_spec[list(GenPanel.list_spec.file_name)[0]].wl[GenPanel.raw_spec[list(GenPanel.list_spec.file_name)[0]].wl.between(300,800)]]
        self.lSV=pd.DataFrame(index=GenPanel.raw_spec[list(GenPanel.raw_spec.keys())[0]].wl[GenPanel.raw_spec[list(GenPanel.raw_spec.keys())[0]].wl.between(300,800)][tokeep_dark], columns=['SV' + str(i) for i in range(n+1)])
        
        
        tmp={}
        for i in range(len(self.scaled_spec_lSV)):
            tmp['lSV'+str(i)]=self.scaled_spec_lSV[:,i]
        pd.DataFrame(data=tmp,index=GenPanel.raw_spec[spec].wl[GenPanel.raw_spec[spec].wl.between(300,800)][tokeep_dark]-GenPanel.raw_spec[list(GenPanel.list_spec.file_name)[0]].A[GenPanel.raw_spec[spec].wl.between(300,800)][tokeep_dark]).to_csv(file_path + file_name + '_rSV.csv',index=True)
        
        # except NameError:
            # print('no SVD results to save')
        
        print(self.scaled_time_factors)
        print('separator')
        print(self.scaled_spec_lSV)
        
        # self.GetParent().GetParent().GetParent().right_panel.figure.savefig(file_path + file_name + ".svg", dpi=900 , transparent=True,bbox_inches='tight')
        # self.GetParent().GetParent().GetParent().right_panel.figure.savefig(file_path + file_name + ".png", dpi=900, transparent=True,bbox_inches='tight')
        # self.GetParent().GetParent().GetParent().right_panel.figure.savefig(file_path + file_name + ".pdf", dpi=900, transparent=True,bbox_inches='tight')
        # print("Figure saved at: " + file_path + file_name + '.png')
        n=len(GenPanel.raw_spec)-1
        self.rSV=pd.DataFrame(index=GenPanel.list_spec.time_code[1:], columns=['SV' + str(i) for i in range(n+1)])
        laser_blue = GenPanel.list_spec.laser_dent_blue.min()
        laser_red = GenPanel.list_spec.laser_dent_red.max()
        tokeep_dark=[np.isnan(laser_blue)  or np.isnan(laser_red)  or x<laser_blue or x>laser_red for x in GenPanel.raw_spec[list(GenPanel.list_spec.file_name)[0]].wl[GenPanel.raw_spec[list(GenPanel.list_spec.file_name)[0]].wl.between(300,800)]]
        self.lSV=pd.DataFrame(index=GenPanel.raw_spec[list(GenPanel.raw_spec.keys())[0]].wl[GenPanel.raw_spec[list(GenPanel.raw_spec.keys())[0]].wl.between(300,800)][tokeep_dark], columns=['SV' + str(i) for i in range(n+1)])
        for i in range(0,n):
            self.rSV['SVn'+str(i)]=self.scaled_time_factors[i]
            self.lSV['SVn'+str(i)]=self.scaled_spec_lSV[:,i] 
        self.rSV.to_csv(file_path + 'time-SV.csv', index=True)
        self.lSV.to_csv(file_path + 'spec-SV.csv', index=True)
        

    def update_right_panel(self, typecorr):
        if len(self.GetParent().GetParent().tab1.field_topeak.GetValue()) == 0:
            scaling_top=280
        else :
            scaling_top = float(self.GetParent().GetParent().tab1.field_topeak.GetValue())
        # print(scaling_top)
        self.GetParent().GetParent().GetParent().right_panel.plot_data(typecorr, scaling_top)
        
    # def on_close(self, event):
    #     self.Destroy()

class TabThree(wx.Panel):
    def __init__(self, parent):
        wx.Panel.__init__(self, parent)
        # Add content for Tab 2 here
        # kinetics 
        sizer=wx.BoxSizer(wx.VERTICAL)
        
        
        smoothsizer = wx.BoxSizer(wx.HORIZONTAL)
        self.smoothtypebutton = wx.Button(self, label="Smoothing type")
        self.smoothtypebutton.Bind(wx.EVT_RIGHT_DOWN, self.OnContextMenu_smoothing)
        
        
        
        smoothsizer.Add(self.smoothtypebutton, 2, wx.EXPAND)
        smoothwindowsizer = wx.BoxSizer(wx.VERTICAL)
        self.smooth_window_label = wx.StaticText(self, label = 'Smoothing window' , style = wx.ALIGN_CENTER)
        self.smooth_window_field = wx.TextCtrl(self, value = '21', style = wx.TE_CENTER)
        smoothwindowsizer.Add(self.smooth_window_label, 1 , wx.ALIGN_CENTER)
        smoothwindowsizer.Add(self.smooth_window_field, 1 , wx.ALIGN_CENTER)
        
        smoothsizer.Add(smoothwindowsizer, 1, wx.EXPAND)
        
        
        
        sizer.Add(smoothsizer, 1, wx.EXPAND | wx.HORIZONTAL, border = 2)
        
        
        
        
        
        self.corrtypebutton = wx.Button(self, label="Backgound correction type")
        self.corrtypebutton.Bind(wx.EVT_RIGHT_DOWN, self.OnContextMenu_correction)
        sizer.Add(self.corrtypebutton, 1, wx.EXPAND | wx.HORIZONTAL, border = 2)
        
        self.qualityscore_button = wx.Button(self, label='print quality of a spectrum')
        self.qualityscore_button.Bind(wx.EVT_BUTTON, self.On_qual)
        sizer.Add(self.qualityscore_button, 1, wx.EXPAND | wx.HORIZONTAL, border = 2)
        
        
        # laser_removal_sizer=wx.BoxSizer(wx.HORIZONTAL)
        self.laser_removal_button = wx.Button(self, label='remove laser dent')
        self.laser_removal_button.Bind(wx.EVT_BUTTON, self.OnLaserRemove)
        # laser_removal_fieldsizer = BoxSizer(wx.VERTICAL)   
        # self.laser_remove_label = wx.StaticText(self, label = 'Excitation wavelength', style = wx.ALIGN_CENTER_HORIZONTAL)
        # laser_removal_fieldsizer.Add(self.laser_remove_label,1 , wx.ALIGN_CENTER | wx.ALL, border = 2)
        # self.laser_remove_field = wx.TextCtrl(self, value = '450', style = wx.TE_CENTER)
        # laser_removal_fieldsizer.Add(self.laser_remove_field,2 , wx.ALIGN_CENTER | wx.ALL, border = 2)
        
        # laser_removal_sizer.Add(laser_removal_fieldsizer, 1, wx.ALIGN_CENTER | wx.ALL, border = 2)
        # laser_removal_sizer.Add(laser_removal_button, 2, wx.ALIGN_CENTER | wx.ALL, border = 2)
        # sizer.Add(laser_removal_sizer, 4, wx.EXPAND, | wx.HORIZONTAL, border = 2)
        sizer.Add(self.laser_removal_button, 1, wx.EXPAND | wx.ALL, border=2)       
        
        self.SetSizer(sizer)
        
    def OnContextMenu_smoothing(self, event):
        menu = wx.Menu()

        stavitski_golay = wx.MenuItem(menu, wx.NewId(), "Stavitski-Golay")
        rolling_average = wx.MenuItem(menu, wx.NewId(), "Rolling-Average")

        menu.Append(stavitski_golay)
        menu.Append(rolling_average)

        self.Bind(wx.EVT_MENU, self.OnStavitskiGolay, stavitski_golay)
        self.Bind(wx.EVT_MENU, self.OnRollingAverage, rolling_average)

        self.PopupMenu(menu)
        menu.Destroy()
        
    def OnStavitskiGolay(self, event):
        print("Stavitski-Golay selected")
        GenPanel.smoothing = 'savgol'

    def OnRollingAverage(self, event):
        print("Rolling-Average selected")
        GenPanel.smoothing = 'rolling'
    
    
    def OnContextMenu_correction(self, event):
        menu = wx.Menu()

        rayleigh = wx.MenuItem(menu, wx.NewId(), "Rayleigh")
        full = wx.MenuItem(menu, wx.NewId(), "full")
        custom = wx.MenuItem(menu, wx.NewId(), "1/λ^n")
        straight = wx.MenuItem(menu, wx.NewId(), 'tinker')
        lin_rayleigh = wx.MenuItem(menu, wx.NewId(), "Linear+Rayleigh")
        
        
        menu.Append(rayleigh)
        menu.Append(full)
        menu.Append(custom)
        menu.Append(straight)
        menu.Append(lin_rayleigh)

        self.Bind(wx.EVT_MENU, self.OnRayleigh, rayleigh)
        self.Bind(wx.EVT_MENU, self.OnFullCorr, full)
        self.Bind(wx.EVT_MENU, self.OnCustomCorr, custom)
        self.Bind(wx.EVT_MENU, self.OnStraight, straight)
        self.Bind(wx.EVT_MENU, self.OnLinRay, lin_rayleigh)
        self.PopupMenu(menu)
        menu.Destroy()
        
    def OnRayleigh(self, event):
        print("Only rayleigh correction has been selected")
        GenPanel.correction = 'rayleigh'

    def OnFullCorr(self, event):
        print("Rolling-Average selected")
        GenPanel.correction = 'full'
    
    def OnCustomCorr(self, event):
        print("1/λ^n selected")
        GenPanel.correction = 'custom'
        
    def OnStraight(self, event):
        print('tinker correction has been chosen')
        GenPanel.correction='straight'
        
    def OnLinRay(self, event):
        GenPanel.correction='lin_rayleigh'
        print("linear + rayleigh correction has been chosen")
        
    def On_qual(self, event):
        file_chooser = FileChooser(self, "Choose a File", 1, list(GenPanel.raw_lamp.keys()))
        if file_chooser.ShowModal() == wx.ID_OK:
            self.selection = file_chooser.check_list_box.GetCheckedStrings()[0]
            GenPanel.raw_lamp[self.selection].quality = GenPanel.raw_lamp[self.selection].I0/GenPanel.raw_lamp[self.selection].I0.max()
            self.update_right_panel('quality_plot')
    def update_right_panel(self, typecorr):
        if len(self.GetParent().GetParent().tab1.field_topeak.GetValue()) == 0:
            scaling_top=280
        else :
            scaling_top = float(self.GetParent().GetParent().tab1.field_topeak.GetValue())
        print(scaling_top)
        self.GetParent().GetParent().GetParent().right_panel.plot_data(typecorr, scaling_top)
    def OnLaserRemove(self, event):

            peak_position_first=0
            for spec in list(GenPanel.list_spec.file_name)[1:]:
                # identifying the laser peak
                lasered_data = signal.savgol_filter(np.array(GenPanel.raw_spec[spec].A[GenPanel.raw_spec[spec].wl.between(350,800)]),
                                                       window_length=23,
                                                       polyorder=3)
                peaks, _ = signal.find_peaks(-lasered_data)
                #assessing whether the dent found is indeed the laser dent, i. e. if it is close to the poisitoin of the laser dent in the first spectrum
                if peak_position_first== 0 : 
                    prominences, left_edge, right_edge = signal.peak_prominences(-lasered_data, peaks, wlen = 30)
                    peak_left = left_edge[np.argmax(prominences)]
                    peak_position = peaks[np.argmax(prominences)]
                    peak_right = right_edge[np.argmax(prominences)]
                    vars()['fig' + spec], vars()['ax' + spec] = plt.subplots()
                    vars()['ax' + spec].plot(lasered_data)
                    vars()['ax' + spec].scatter(peaks, lasered_data[peaks], color = 'red', s=50)
                    vars()['ax' + spec].scatter(np.array([peak_left, peak_position, peak_right]), lasered_data[np.array([peak_left, peak_position, peak_right])], color = 'green')
                    vars()['fig' + spec].show()
                    peak_position_first=peak_position
                    wavelength_laser = GenPanel.raw_spec[spec].wl[GenPanel.raw_spec[spec].wl.between(350,800)].iloc[peak_position]
                    laser_blue = GenPanel.raw_spec[spec].wl[GenPanel.raw_spec[spec].wl.between(350,800)].iloc[peak_left]
                    laser_red = GenPanel.raw_spec[spec].wl[GenPanel.raw_spec[spec].wl.between(350,800)].iloc[peak_right]
                    print('in ' + spec + ' Laser dent found at '+ str(GenPanel.raw_spec[spec].wl[GenPanel.raw_spec[spec].wl.between(350,800)].iloc[peak_position]))  
                    GenPanel.raw_spec[spec]=GenPanel.raw_spec[spec][~ GenPanel.raw_spec[spec].wl.between(laser_blue-5, laser_red+5)]
                    GenPanel.list_spec.laser_dent_blue[spec]=laser_blue-5
                    GenPanel.list_spec.laser_dent_red[spec]=laser_red+5
                else:
                    closepeaks=[x > peak_position_first - 10 and x < peak_position_first + 10 for x in peaks]
                    if np.array(closepeaks).any():
                        peaks=peaks[closepeaks]
                        prominences, left_edge, right_edge = signal.peak_prominences(-lasered_data, peaks, wlen = 30)
                        peak_left = left_edge[np.argmax(prominences)]
                        peak_position = peaks[np.argmax(prominences)]
                        peak_right = right_edge[np.argmax(prominences)]
                        vars()['fig' + spec], vars()['ax' + spec] = plt.subplots()
                        vars()['ax' + spec].plot(lasered_data)
                        vars()['ax' + spec].scatter(peaks, lasered_data[peaks], color = 'red',s=50)
                        vars()['ax' + spec].scatter(np.array([peak_left, peak_position, peak_right]), lasered_data[np.array([peak_left, peak_position, peak_right])], color = 'green')
                        vars()['fig' + spec].show()
                        peak_position_first=peak_position
                        wavelength_laser = GenPanel.raw_spec[spec].wl[GenPanel.raw_spec[spec].wl.between(350,800)].iloc[peak_position]
                        laser_blue = GenPanel.raw_spec[spec].wl[GenPanel.raw_spec[spec].wl.between(350,800)].iloc[peak_left]
                        laser_red = GenPanel.raw_spec[spec].wl[GenPanel.raw_spec[spec].wl.between(350,800)].iloc[peak_right]
                        print('in ' + spec + ' laser dent found at '+ str(GenPanel.raw_spec[spec].wl[GenPanel.raw_spec[spec].wl.between(350,800)].iloc[peak_position]))  
                        GenPanel.raw_spec[spec]=GenPanel.raw_spec[spec][~ GenPanel.raw_spec[spec].wl.between(laser_blue-5, laser_red+5)]
                        GenPanel.list_spec.laser_dent_blue[spec]=laser_blue-5
                        GenPanel.list_spec.laser_dent_red[spec]=laser_red+5
                    else:
                        print('in ' + spec + ' no laser dent found')
            print(GenPanel.list_spec[['laser_dent_blue','laser_dent_red']])
        

    
# Suppress GTK warning

if __name__ == "__main__":
    app = wx.App()
    frame = MainFrame()
    app.MainLoop()
