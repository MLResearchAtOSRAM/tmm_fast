import numpy as np
import matplotlib.colors as colors
import matplotlib.cm as cm

def plot_stacks(ax, indexes, thickness, labels=None, show_material=True):
    '''
    Plots material layers on top of each other with the refractive index of the layer in colorcode.
    
    Parameter: 
        ax: matplotlib axes object 
            axes where the plot will put into
            
        indexes: array_like
            array of real or complex refractive indexes 
            
        thickness: array_like or list of array_like
            if provided an array of thicknesses, plots one stack 
            if provided a list of arrays of thicknesses, plots them next to each other 
            the values should be given in meters (ie. 4e-6 for a 4 micron thick layer)
            
    Key word arguments: (optional)
        labels: str or list of str
            Labels for the stack, if provided a list of str, the length must exactly match the
            length of the list of thicknesses
            
        show_material: boolean
            If True, displays the real refractive index of the first wavelength which is computed 
            directly in the depiction of the layer. If the layer is too thin to properly display 
            the refractive index, it is suppresed.
            
    Returns:
        ax: matplotlib axes object 
            axes with the plotted stacks for further modification
            
        cmap: matplotlib colormap object 
            colormap to show 


    Example:
    fig, ax = plt.subplots(1,1)
    indexes = np.array([2, 1, 2.5, 1.6])
    thickness = np.array([5, 7, 3, 6]*1e-6)
    labels = 'this is my stack'
    ax, cmap = plot_stacks(ax, indexes, thickness, labels=labels ) 
    plt.show()
    '''
    mat={'1.4585':'Si02',
        '2.3403':'Nb205',
        '2.3991':'GaN'}
    
    if type(indexes) is not list:
        minmax = colors.Normalize(vmin=min(indexes)-1, vmax=max(indexes)+1)
        indexes = indexes.real[::-1] 
    else:
        for i in range(len(indexes)):
            indexes[i] = indexes[i].real[::-1]
#     indexes = indexes.real[::-1]
        minmax = colors.Normalize(vmin=min(indexes[0])-1, vmax=max(indexes[0])+1)
    cmap = cm.ScalarMappable(norm= minmax, cmap=cm.rainbow)
    if labels is None: # if no labels are provided, numerate the stacks
        labels = str(np.arange(len(thickness)))
    if type(thickness) is list:
        max_stack_height = np.max([np.sum(k)*1e6 for k in thickness])
        for j, thick in enumerate(thickness):
            position = j*0.4
            for i, layer in enumerate(np.cumsum(thick*1e6)[::-1]):
                ax.bar(position, layer, 0.36, color = cmap.to_rgba(indexes[j][i]) )
                if show_material and ((thick*1e6)[::-1][i] > max_stack_height/22):
                    text = mat[str(indexes[j][i])] if str(indexes[j][i]) in mat else 'n='+str(indexes[j][i])
                    ax.text(position-0.175, layer-0.008*max_stack_height, text, va='top', c='gray')
        ax.set_xticks(np.arange(0, 0.4*len(thickness), 0.401))  # funny trick to make sure the labels
                                                                # are centered beneath the stack
        if labels is not None:
            ax.set_xticklabels(labels)
        else:
            ax.set_xticklabels([i+1 for i in range(len(thickness))])
    else:      
        total_stack_height = np.sum(thickness)*1e6
        for i, layer in enumerate(np.cumsum(thickness*1e6)[::-1]):
            ax.bar(0, layer, 0.2, color = cmap.to_rgba(indexes[i]) )
            if show_material and ((thickness*1e6)[::-1][i] > total_stack_height/22):
                text = mat[str(indexes[i])] if str(indexes[i]) in mat else 'n='+str(indexes[i])
                ax.text(-0.098, layer-0.008*total_stack_height, text, va='top', c='gray')
        ax.set_ylim(0, (np.sum(thickness)*1.05*1e6))
        ax.xaxis.set_visible(False)
    ax.set_ylabel(r'Thickness in $\mu$m')
        
    return ax, cmap