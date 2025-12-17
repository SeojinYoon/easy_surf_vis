
"""
This file contains the basic source code to visualize graph using matplotlib
"""
# Common Libraries
import numpy as np
import matplotlib.pylab as plt
from scipy.stats import cumfreq
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

# Custom Libraries
if os.getenv("easysurfvis_isRunSource"):
    sys.path.append(os.getenv("easysurfvis_source_home"))
    from cores.general_util import search_stringAcrossTarget, find_consecutive_ranges
else:
    from easysurfvis.cores.general_util import search_stringAcrossTarget, find_consecutive_ranges

# Functions
def draw_title(axis, title_info = {}):
    """
    Draw title in the axis
    
    :param title_info(dictionary): 
         -k, title(string): title
         -k, title_weight(string): weight of font
         -k, title_size(string): size of title
         -k, title_y_pos(string): y pos of title
    """
    title = title_info.get("title", "")
    title_weight = title_info.get("title_weight", "bold")
    title_size = title_info.get("title_size", 20)
    title_y_pos = title_info.get("title_y_pos", 1.0)
    
    axis.set_title(title, 
                   fontweight = title_weight,
                   size = title_size, 
                   y = title_y_pos)
        
def draw_label(axis, label_info = {}):
    """
    Draw label in the axis
    
    :param label_info(dictionary): 
        -k, color(string): label color
        -k, x_label(string): x label text
        -k, y_label(string): y label text
        -k, x_weight(string): x label weight
        -k, y_weight(string): y label weight
        -k, x_size(float): x label size
        -k, y_size(float): y label size
    """
    color = label_info.get("color", "black")
    
    x_label = label_info.get("x_label", "")
    y_label = label_info.get("y_label", "")
    
    x_weight = label_info.get("x_weight", "normal")
    y_weight = label_info.get("y_weight", "normal")

    x_label_size = label_info.get("x_size", 16)
    y_label_size = label_info.get("y_size", 16)
    
    x_label_pad = label_info.get("x_label_pad", 10)
    y_label_pad = label_info.get("y_label_pad", 10)
    
    axis.set_xlabel(x_label, 
                    color = color, 
                    weight = x_weight, 
                    size = x_label_size,
                    labelpad = x_label_pad)

    axis.set_ylabel(y_label, 
                    color = color, 
                    weight = y_weight, 
                    size = y_label_size,
                    labelpad = y_label_pad)
            
def draw_ticks(axis, tick_info = {}):
    """
    Draw ticks in the axis
    
    :param tick_info(dictionary): tick style configuration
        -k, x_data(list): x tick positions ex) [1,2,3]
        -k, x_names(list): x tick text ex) ["a", "b", "c"]
        -k, x_tick_weight(string): x tick weight
        -k, x_tick_size(float): x tick size
        -k, x_tick_rotation(int): x tick rotation
        
        -k, y_data(list): y tick positions ex) [1,2,3]
        -k, y_names(list): y tick text ex) ["a", "b", "c"] 
        -k, y_tick_precision(float): Number of decimal places to display on y-axis ticks
        -k, y_tick_weight(string): y tick weight
        -k, y_tick_size(float): y tick size
        -k, y_tick_rotation(int): y tick rotation
        -k, y_divided(int): the number of division for showing y-tick
        -k, y_need_tick(list): the y-tick which must to show
    """
    x_data = tick_info.get("x_data", [])
    x_names = tick_info.get("x_names", [])
    
    x_tick_weight = tick_info.get("x_tick_weight", "normal")
    x_tick_size = tick_info.get("x_tick_size", 14)
    x_tick_rotation = tick_info.get("x_tick_rotation", 90)
    x_tick_viewType = tick_info.get("x_tick_viewType", "remove_duplication")
    
    y_data = tick_info.get("y_data", [])
    y_names = tick_info.get("y_names", [])
    
    y_tick_weight = tick_info.get("y_tick_weight", "normal")
    y_tick_size = tick_info.get("y_tick_size", 14)
    y_tick_rotation = tick_info.get("y_tick_rotation", 0)
    y_tick_precision = tick_info.get("y_tick_precision", None)
    
    # X
    x_data = np.array(x_data)
    x_names = np.array(x_names)
    
    if x_tick_viewType == "remove_duplication":
        x_tick_duplication = np.array([int((start + end) / 2) for start, end in find_consecutive_ranges(x_names)], dtype = int)
        x_tick_data = np.array([(x_data[start] + x_data[end]) / 2 for start, end in find_consecutive_ranges(x_names)])
        
        if len(x_tick_duplication) > 0:
            x_names = x_names[x_tick_duplication]
            x_data = x_tick_data
        
    axis.set_xticks(x_data, 
                    x_names, 
                    rotation = x_tick_rotation, 
                    weight = x_tick_weight, 
                    size = x_tick_size)
    
    # Y
    axis.set_yticks(y_data, 
                    y_names,
                    rotation = y_tick_rotation,
                    weight = y_tick_weight,
                    size = y_tick_size)
    if y_tick_precision is not None:
        axis.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.{y_tick_precision}f}"))
        
def draw_spine(axis, spine_info = {}):
    """
    Draw spine in the axis
    
    :param spine_info(dictionary): 
        -k, spine_linewidth(float): spine line width ex) 2
        -k, spine_color(string): spine line color ex) "black"
        -k, invisibles(list): invisible informations ex) ["right", "top"]
    """
    spine_line_width = spine_info.get("spine_linewidth", 1)
    spine_color = spine_info.get("spine_color", "black")
    invisibles = spine_info.get("invisibles", ["right", "top"])
    
    all_spines = ["bottom", "top", "left", "right"]
    for invisible in invisibles:
        axis.spines[invisible].set_visible(False)
        all_spines.remove(invisible)
        
    for ax_name in all_spines:
        axis.spines[ax_name].set_linewidth(spine_line_width)
        axis.spines[ax_name].set_linewidth(spine_line_width)
        axis.spines[ax_name].set_color(spine_color)

def make_colorbar(vmin, 
                  vmax, 
                  figsize = (2, 6), 
                  n_inner_ticks = 3, 
                  cmap = "jet", 
                  tick_precision = 4, 
                  orientation = "horizontal",
                  fontsize = 12):
    """
    Creates a customizable colorbar using matplotlib.


    :param vmin (float): The minimum value for the colorbar.
    :param vmax (float): The maximum value for the colorbar.
    :param figsize (tuple): The size of the figure (width, height). Defaults to (2, 6).
    :param n_div (int): Number of divisions (ticks) on the colorbar. Defaults to 4.
    :param cmap (str): Colormap to use for the colorbar. Defaults to "jet".
    :param tick_precision (int): Number of decimal places to display on the tick labels. Defaults to 4.
    :param orientation (str): Orientation of the colorbar, either "horizontal" or "vertical". Defaults to "horizontal".

    return (tuple): A tuple containing the matplotlib figure and axis objects.
    """
    n_div = n_inner_ticks + 1
    
    interval = (vmax - vmin) / n_div
    ticks = np.linspace(vmin, vmax, n_div + 1)

    # Create the figure and axis for the colorbar
    fig = plt.figure(figsize = figsize)

    if orientation == "vertical":
        axis = fig.add_axes([0.05, 0.05, 0.15, 0.9])  # [left, bottom, width, height]
    else:
        axis = fig.add_axes([0.1, 0.5, 0.8, 0.3])  # [left, bottom, width, height]
    
    # Create the colorbar
    cmap = plt.get_cmap(cmap)
    norm = plt.Normalize(vmin = vmin, vmax = vmax)
    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=axis, orientation=orientation)
    
    # Set the ticks and labels
    if orientation == "vertical":
        axis.set_yticks(ticks)
        axis.set_yticklabels([f"{tick:.{tick_precision}f}" for tick in ticks], fontsize = fontsize)
        axis.get_xaxis().set_visible(False)
    else:
        axis.set_xticks(ticks)
        axis.set_xticklabels([f"{tick:.{tick_precision}f}" for tick in ticks], fontsize = fontsize)
        axis.get_yaxis().set_visible(False)
    
    return fig, axis, ticks

if __name__=="__main__":
    pass
