# File: ScrollTest.py
# coding: ASCII
"""
Interatively zoom plots together, but permit them to scroll independently.
"""
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider
from pudb import set_trace
from time import sleep
from lfp_proc import *
from lfp_view import *
import sys

def _get_limits( ax ):
    """ Return X and Y limits for the passed axis as [[xlow,xhigh],[ylow,yhigh]]
    """
    return [list(ax.get_xlim()), list(ax.get_ylim())]

def _set_limits( ax, lims ):
    """ Set X and Y limits for the passed axis
    """
    ax.set_xlim(*(lims[0]))
    ax.set_ylim(*(lims[1]))
    return

def pre_zoom( fig ):
    """ Initialize history used by the re_zoom() event handler.
        Call this after plots are configured and before plt.show().
    """
    global oxy
    oxy = [_get_limits(ax) for ax in fig.axes]
    # :TODO: Intercept the toolbar Home, Back and Forward buttons.
    return

def re_zoom(event):
    """ plt event handler to scroll all plots together
        Use with 'motion_notify_event' and 'button_release_event'.
    """
    global oxy
    for ax in event.canvas.figure.axes:
        navmode = ax.get_navigate_mode()
        if navmode is not None:
            break
    scrolling = (event.button == 1) and (navmode == "PAN")
    if scrolling:                   # Update history (independent of event type)
        changed = None
        for i, ax in enumerate(event.canvas.figure.axes): # Get the axes
            # Find the plot that changed
            nxy = _get_limits(ax)
            if (oxy[i][0] != nxy[0]):         # This plot has changed
                changed = i
                newx = nxy[0]
                break                   # No need to look at other axes
        oxy = [_get_limits(ax) for ax in event.canvas.figure.axes]
        if changed is not None:
            for i, ax in enumerate(event.canvas.figure.axes):
                if i != changed:
                    _set_limits(ax,[newx, oxy[i][1]])
            event.canvas.draw()         # re-draw the canvas (if required)
            pre_zoom(event.canvas.figure)   # Update history
        return
    if event.name != 'button_release_event':    # Nothing to do!
        return

    # We have a non-scroll 'button_release_event': Were we zooming?
    zooming = (navmode == "ZOOM") or ((event.button == 3) and (navmode == "PAN"))
    if not zooming:                 # Nothing to do!
        oxy = [_get_limits(ax) for ax in event.canvas.figure.axes]  # To be safe
        return
    # We were zooming, but did anything change?  Check for zoom activity.
    # changed = None
    # zoom = [[0.0,0.0],[0.0,0.0]]    # Zoom from each end of axis (2 values per axis)
    # for i, ax in enumerate(event.canvas.figure.axes): # Get the axes
    #     # Find the plot that changed
    #     nxy = _get_limits(ax)
    #     if (oxy[i] != nxy):         # This plot has changed
    #         changed = i
    #         # Calculate zoom factors
    #         for j in [0,1]:         # Iterate over x and y for each axis
    #             # Indexing: nxy[x/y axis][lo/hi limit]
    #             #           oxy[plot #][x/y axis][lo/hi limit]
    #             width = oxy[i][j][1] - oxy[i][j][0]
    #             # Determine new axis scale factors in a way that correctly
    #             # handles simultaneous zoom + scroll: Zoom from each end.
    #             zoom[j] = [(nxy[j][0] - oxy[i][j][0]) / width,  # lo-end zoom
    #                        (oxy[i][j][1] - nxy[j][1]) / width]  # hi-end zoom
    #         break                   # No need to look at other axes
    # if changed is not None:
    #     for i, ax in enumerate(event.canvas.figure.axes): # change the scale
    #         if i == changed:
    #             continue
    #         for j in [0,1]:
    #             width = oxy[i][j][1] - oxy[i][j][0]
    #             nxy[j] = [oxy[i][j][0] + (width*zoom[j][0]),
    #                       oxy[i][j][1] - (width*zoom[j][1])]
    #         _set_limits(ax, nxy)
        event.canvas.draw()         # re-draw the canvas (if required)
        pre_zoom(event.canvas.figure)   # Update history
    return
# End re_zoom()

def pan(event):
    while(1):
        try:
            for ax in event.canvas.figure.axes:
                xlim,ylim = _get_limits(ax)
                xlim = np.array(xlim) + 50
                lims = [xlim, ylim]
                _set_limits(ax, lims)
            event.canvas.draw()
        except:
            return

def main(argv):
    """ Test/demo code for re_zoom() event handler.
    """
    mydata,fps = readint('gt1962_2_d1_150528_115815.int')
    data,fps = downsample(mydata,fps=fps)
    filtered_data = wave_filter(mydata,fps)
    freqs = np.arange(1,250,5)
    pwr = tfwindow(data,fps,freqs,channel=3)
#    set_trace()
    fig = plt.figure()               # Create plot
    ax1 = plt.subplot(4,1,1)
    ax1.plot(data[3,:])
    plt.title('Raw dataset')
    plt.xlim( (0,1000) )
    ax2 = plt.subplot(4,1,2)
    ax2.plot(filtered_data[3,:])
    plt.title('Filtered dataset')
    plt.xlim( (0,1000) )
    ax3= plt.subplot(4,1,3)
    ax3.imshow(pwr[0,:,:], aspect='auto')
    plt.title('Time-frequency plot')
    plt.xlim( (0,1000) )
    labels = [str(item) for item in np.linspace(-50,250,7)]
    ax3.set_yticklabels(labels)
    axcolor = 'lightgoldenrodyellow'
#    ax4 = plt.subplot(4,1,4)
    axpos = plt.axes([0.25, 0.1, 0.65, 0.03], axisbg=axcolor)
    POS_SLIDER = Slider(axpos, 'Position', 500, data.shape[1], valinit=500)
    print(data.shape[1])

    def update(val):
        pos = POS_SLIDER.val
        for ax in fig.axes:
            ylim = _get_limits(ax)[1]
            xlim = [pos-500,pos+500]
            lims = [xlim, ylim]
            _set_limits(ax, lims)
        fig.canvas.draw_idle()
    POS_SLIDER.on_changed(update)

    pre_zoom( fig )                     # Prepare plot event handler
    plt.connect('motion_notify_event', re_zoom)  # for right-click pan/zoom
    plt.connect('button_release_event',re_zoom)  # for rectangle-select zoom
    plt.connect('key_press_event', pan)

    plt.show()                       # Show plot and interact with user
# End main()

if __name__ == "__main__":
    # Script is being executed from the command line (not imported)
    main(sys.argv)
