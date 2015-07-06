# coding: ASCII
"""
Find sharp wave ripples
"""
from matplotlib import pylab as plt
from matplotlib.widgets import *
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
    fig_axes = event.canvas.figure.axes[:3]
    for ax in fig_axes:
        navmode = ax.get_navigate_mode()
        if navmode is not None:
            break
    scrolling = (event.button == 1) and (navmode == "PAN")
    if scrolling:                   # Update history (independent of event type)
        changed = None
        for i, ax in enumerate(fig_axes): # Get the axes
            # Find the plot that changed
            nxy = _get_limits(ax)
            if (oxy[i][0] != nxy[0]):         # This plot has changed
                changed = i
                newx = nxy[0]
                break                   # No need to look at other axes
        oxy = [_get_limits(ax) for ax in fig_axes]
        if changed is not None:
            for i, ax in enumerate(fig_axes):
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

def onselect(eclick, erelease):
  'eclick and erelease are matplotlib events at press and release'
  print ' startposition : (%f, %f)' % (eclick.xdata, eclick.ydata)
  print ' endposition   : (%f, %f)' % (erelease.xdata, erelease.ydata)
  print ' used button   : ', eclick.button

def toggle_selector(event):
    print ' Key pressed.'
    if event.key in ['Q', 'q'] and toggle_selector.RS.active:
        print ' RectangleSelector deactivated.'
        toggle_selector.RS.set_active(False)
    if event.key in ['A', 'a'] and not toggle_selector.RS.active:
        print ' RectangleSelector activated.'
        toggle_selector.RS.set_active(True)

def pan(event):
    fig_axes = event.canvas.figure.axes[:3]
    while(1):
        try:
            for ax in fig_axes:
                xlim,ylim = _get_limits(ax)
                xlim = np.array(xlim) + 50
                lims = [xlim, ylim]
                _set_limits(ax, lims)
            event.canvas.draw()
        except:
            return

def calculate_SPWR(data, filtered_data, sfreq, tfr):
    """
    Detects SPWR in all channels
    """
    #set_trace()
    channels = xrange(filtered_data.shape[0])
    SPWRs = {}
    for ch in channels:
        SPW = find_SPW(data,channel=ch)
        freqs = np.arange(1,250,5)
        SPWR = find_SPWR(filtered_data,tfr[ch,:,:], sfreq, SPW, channel=ch)
        SPWRs[str(ch)]=SPWR
        print('Reading channel %d...' % ch)
    print('All channels read.')
    all_channels = []
    #set_trace()
    for ch in channels:
        candidates = map(lambda x:int(np.median(x)), SPWRs[str(ch)])
        all_channels.append(candidates)
    #print(all_channels)
    return all_channels

def seeknext(event):
    print("Seeking to next SPWR")
    global current_SPR
    current_SPR = (current_SPR + 1) % len(SPR_list)
    seekto = SPR_list[current_SPR]
    lb = max(seekto-500, 0)
    ub = min(seekto+500, data.shape[1])
    for i in xrange(3):
        ax = event.canvas.figure.axes[i]
        ax.set_xlim(lb,ub)
    event.canvas.draw()

def seekprev(event):
    print("Seeking to previous SPWR")
    global current_SPR
    current_SPR = (current_SPR - 1) % len(SPR_list)
    seekto = SPR_list[current_SPR]
    lb = max(seekto-500, 0)
    ub = min(seekto+500, data.shape[1])
    for i in xrange(3):
        ax = event.canvas.figure.axes[i]
        ax.set_xlim(lb,ub)
    event.canvas.draw()

def main(argv):
    """ Why so many globals?!?
    """
    if len(argv) != 1:
        argv = ['gt1962_2_d1_150528_115815.int']
    intan_data = argv[0]
    mydata,fps = readint(intan_data)
    global data
    active_channel = 0
    data,fps = downsample(mydata,fps=fps)
    filtered_data = wave_filter(mydata,fps)
    freqs = np.arange(1,250,5)
    tfspec = morlet(data,fps,freqs)
    pwr = baseline_normalize(tfspec)
 #   set_trace()
    global fig
    fig = plt.figure()               # Create plot
    ax1 = fig.add_subplot(4,1,1)
    raw_signal, = ax1.plot(data[0,:])
    plt.title('Raw dataset')
    plt.xlim( (0,1000) )
    ax2 = fig.add_subplot(4,1,2)
    filtered_signal, = ax2.plot(filtered_data[0,:])
    plt.title('Filtered dataset')
    plt.xlim( (0,1000) )
    ax3 = fig.add_subplot(4,1,3)
    tfplot = ax3.imshow(np.squeeze(pwr[0,:,:]), aspect='auto')
    plt.title('Time-frequency plot')
    plt.xlim( (0,1000) )
    labels = [str(item) for item in np.linspace(-50,250,7)]
    ax3.set_yticklabels(labels)
    global SPWRs
    SPWRs = calculate_SPWR(data, filtered_data,fps,pwr)
    # Radio buttons
    global nchans
    nchans = len(SPWRs)
    buttons = tuple(['Ch ' + str(i) for i in xrange(nchans)])
    rax = plt.axes([0.8, 0.00, 0.15, 0.05*nchans])
    radio = RadioButtons(rax, buttons)

    def get_channel_data(channel):
        raw = data[channel,:]
        filtered = filtered_data[channel,:]
        tfspec = pwr[channel,:,:]
        return raw, filtered, tfspec

    def update_data(label):
        global active_channel
        global txt
        active_channel = int(label.split()[-1])
        raw, filtered, tfspec = get_channel_data(active_channel)
        raw_signal.set_ydata(raw)
        filtered_signal.set_ydata(filtered)
        global SPR_list
        SPR_list = SPWRs[active_channel]
        tfplot.set_data(pwr[active_channel,:,:])
        txt.remove()
        info = '{} possible sharp-wave ripple events detected in channel {}'.format(len(SPR_list), active_channel)
        txt = plt.text(-0.7,1.4, info, fontsize=12)
        fig.canvas.draw()
        
    # Next and previous
    global current_SPR
    current_SPR = -1
    global SPR_list
    SPR_list = SPWRs[active_channel]
    info = '{} possible sharp-wave ripple events detected in channel 0'.format(len(SPR_list))
    global txt
    txt = plt.text(-4.5,0.5, info, fontsize=12)
    axprev = plt.axes([0.1, 0.05, 0.1, 0.075])
    axnext = plt.axes([0.21, 0.05, 0.1, 0.075])
    bnext = Button(axnext, 'Next')
    current_SPR = bnext.on_clicked(seeknext)
    bprev = Button(axprev, 'Previous')
    current_SPR = bprev.on_clicked(seekprev)
    radio.on_clicked(update_data)




#    axcolor = 'lightgoldenrodyellow'
#    ax4 = plt.subplot(4,1,4)
    # axpos = plt.axes([0.25, 0.1, 0.65, 0.03], axisbg=axcolor)
    # POS_SLIDER = Slider(axpos, 'Position', 500, data.shape[1], valinit=500)
    # print(data.shape[1])

    # def update(val):
    #     pos = POS_SLIDER.val
    #     for ax in fig.axes:
    #         ylim = _get_limits(ax)[1]
    #         xlim = [pos-500,pos+500]
    #         lims = [xlim, ylim]
    #         _set_limits(ax, lims)
    #     fig.canvas.draw_idle()
    # POS_SLIDER.on_changed(update)

    pre_zoom( fig )                     # Prepare plot event handler
    plt.connect('motion_notify_event', re_zoom)  # for right-click pan/zoom
    plt.connect('button_release_event',re_zoom)  # for rectangle-select zoom
    plt.connect('key_press_event', pan)
#    toggle_selector.RS = RectangleSelector(ax, onselect, drawtype='box')
#    plt.connect('key_press_event', toggle_selector)

    plt.show()                       # Show plot and interact with user
# End main()

if __name__ == "__main__":
    # Script is being executed from the command line (not imported)
    main(sys.argv)
