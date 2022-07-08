import numpy as np
import skimage.io

from skimage.filters import threshold_otsu, gaussian, threshold_local
from skimage.morphology import remove_small_objects
from skimage.segmentation import clear_border
from skimage.util import invert
from skimage.measure import regionprops, label
from skimage.color import label2rgb
from skimage.draw import rectangle_perimeter
from skimage.transform import rescale, resize, downscale_local_mean
from skimage.transform import rotate
from skimage.measure import find_contours

import pandas as pd
import math
from scipy.spatial import distance
from scipy.signal import find_peaks
from scipy.stats import trim_mean
from scipy.stats import t
from scipy.stats import entropy
from matplotlib import pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from tqdm import tqdm
from scipy.stats import multivariate_normal

from skimage.color import rgb2gray
from skimage import color

import torch
import torch.nn as nn
import copy


def test_reload():
    print('A')
    
def Generate_Plate_Map(path):
    '''read txt file storing plate mapping and convert to 384 plate format'''
    file = open(path,'r')
    corpus=file.read().split('\n')
    plate_num=0
    plate_map=dict()
    for line in corpus:
        #print(line)
        if 'Plate' in line:
            plate_num+=1
            r=0
            plate_map[plate_num]=np.empty((32,48), dtype='<U16')
        else:
            c=0
            for gene in line.split(' ')[2:-1]: #[2:-1] gets rid of the two letters and the last letter flanking the row at each line
                plate_map[plate_num][r:r+3,c:c+3]=gene
                c+=2
            r+=2

    return plate_map


def make_mask(image, t, s, hardImageThreshold=None, hardSizeThreshold=None, local=False):
    '''
    Identifies suitable morphological components from image by thresholding.
    '''
    
    if local:
        mask = image > t*threshold_local(image, 151)

    else:
        if hardImageThreshold:
            thresh = hardImageThreshold
        else:
            thresh = t*threshold_otsu(image)
            
        mask = image>thresh
    
    #Filter small components. The default threshold is 0.00005 of the image area 
    if hardSizeThreshold:
        size_thresh = hardSizeThreshold
    else:
        size_thresh = s * np.prod(image.shape) * 0.00005
    mask = remove_small_objects(mask, min_size=size_thresh)
    
    #Fill holes?? In future
    
    #Clear border
    mask = clear_border(mask)
   # plt.imshow(mask)
    #plt.title('clear border')
   # plt.show()
    #Label connected components
    mask = label(mask)
    return mask

def make_grid_auto(im, grid):

    nrows, ncols = map(int,grid.split('-'))
    
    def find_grid_positions_1d(image, axis, n):

        #extract means across axis
        imvals = image.mean(axis=axis)
        imvals = imvals - imvals.min()
        imvals = imvals / imvals.max()

        #find peaks. Define minimum distance based on image dimension
        peaks = find_peaks(imvals, distance=(len(imvals)-0.2*len(imvals))/n)[0]

        #find distance between colonies. Use trimmed mean which is robust to outliers. Median is not precise enough (need sub-pixel resolution)
        med = trim_mean(peaks[1:] - peaks[:-1], 0.2)
        #for bad input images the distance between colonies times the number of rows/columns can exceed image dimensions. 
        #In this case, guess the distance based on image dimensions and number of colonies
        if med*(n-1) > len(imvals):
            print('Could not detect enough peaks. Guessing grid positions. Please check QC images carefully.')
            med = (len(imvals)-0.1*len(imvals))/(n-1)

        #create hypothetical, ideal grid based on mean distance
        to_fit = np.linspace(0, med*(n-1),n)

        #Find the maximum offset and all offset positions to try
        max_offset = len(imvals)-to_fit[-1]
        pos_to_try = np.linspace(0,int(max_offset),int(max_offset)+1)

        #Make a cosine function with the same period as the mean distance between colonies
        b = 2 * math.pi / med
        x = np.linspace(0,(n-1)*med,int((n-1)*med))
        y = (1+np.cos(x*b))/2#scale roughly to data
        errors = [((y - imvals[o:len(y)+o])**2).sum() for o in pos_to_try.astype(int)]

        return to_fit + np.argmin(errors), med

    cols, colmed = find_grid_positions_1d(im,0,ncols)
    rows, rowmed = find_grid_positions_1d(im,1,nrows)

    grid = {}
    for ri,r in enumerate(rows):
        for ci,c in enumerate(cols):
            grid[(ri+1, ci+1)] = (r, c)
               
    return grid, 0.5*(colmed+rowmed)

def make_grid(gd):
    '''
    Converts a grid definition to a list (x,y positions) of all vertices in the grid.
    '''
    
    rows, cols, x1, y1, x2, y2 = gd
    xpos = np.linspace(x1, x2, num=cols)
    ypos = np.linspace(y1, y2, num=rows)
    
    griddistx = xpos[1] - xpos[0]
    griddisty = ypos[1] - ypos[0]
    
    #Check if spacing similar
    if (abs(griddistx-griddisty)/(0.5*(griddistx+griddisty))) > 0.1:
        warn('Uneven spacing between rows and columns. Are you sure this is intended?')
   
    #Make dictionary of grid positions
    grid = {}
    for r in range(rows):
        for c in range(cols):
            grid[(r+1,c+1)] = (ypos[r], xpos[c])
            
    return grid, 0.5*(griddistx+griddisty)

def HSV_filter(image):
    '''Convert rgb to hsv and resturn the geometric mean of S and V'''
    convert=0
    if image.mean()>1:
        image=image/255
        convert=1
    image = color.convert_colorspace(image, 'RGB', 'HSV')
    image = np.power(image[:,:,1]*image[:,:,2],0.5) 
    return image*255 if convert == 1 else image 

def Saturation_Filter(image):
    '''Convert rgb to hsv and resturn the saturation only'''
    convert=0
    if image.mean()>1:
        image=image/255
        convert=1
    image = color.convert_colorspace(image, 'RGB', 'HSV')
    image =image[:,:,1]
    return image*255 if convert == 1 else image 

def PCA_filter(Im):
    '''Performs a PCA on Three color points and projects an image on the Second Component '''
    Brown=[0.2,0.2,0]
    Yellow=[255/255, 255/255,0/255]
    White=[1,1,1]
    X=np.stack([ Yellow,White, Brown])

    Xmean=X.mean(axis=0)
    Xstd=X.std(axis=0)
    X =( X - X.mean(axis=0))/Xstd

    #runs singular  vector decomposition on X
    U,e,Vh=np.linalg.svd(X)

    L,W,_ =Im.shape
    Im_standardised=(Im-Xmean)/Xstd
    Projection=(Im_standardised.reshape((-1,3))@Vh.T).reshape((L,W,3))
    #check:
    #plot all the image pixels projected on PC1 and PC2 with there original color
   # plt.scatter(Projection.reshape((-1,3))[:,0],Projection.reshape((-1,3))[:,1], c=Im.reshape((-1,3)))
    return Projection[:,:,1] #returns projection on second component

def HSV_filter_SC(image):
    '''take the mean of inverted H and geometric mean of S and V'''
    convert=0
    if image.mean()>1:
        image=image/255
        convert=1
    image = color.convert_colorspace(image, 'RGB', 'HSV')
    image = 0.5*(1-image[0:500,0:500,0])+0.5*np.power(image[0:500,0:500,1]*image[0:500,0:500,2],2)
    return image*255 if convert == 1 else image 



def quantify_single_image_HSV(orig_image, grid, auto, t, d, s, negate, RGB_filter, HSV_filter, reportAll=False, hardImageThreshold=None, hardSizeThreshold=None, localThresh=None):
    '''
    Process a single image to extract colony sizes.
    '''
    #all all the steps bellow work on a HxWx1 dimensional image, we extract here two different suhc image
    #1) for making the mask (RGB based)
    #2) for grading colony color (HSV based)
    
    image_RGB =  RGB_filter(orig_image)
    image_HSV =  HSV_filter(orig_image)
    
    #Prepare image
    image_RGB = check_and_negate(image_RGB, negate=negate)
    image_HSV = check_and_negate(image_HSV, negate=negate)
    
    #Create grid
    if auto:
        grid, griddist = make_grid_auto(image_RGB, grid)
    else:
        grid, griddist = make_grid(grid)
        
    #Make mask
    mask = make_mask(image_RGB, t, s, hardImageThreshold=hardImageThreshold, hardSizeThreshold=hardSizeThreshold, local=localThresh)

    #Measure regionprops
    data = {r.label : {p : r[p] for p in ['label', 'area', 'centroid', 'mean_intensity', 'perimeter']} for r in regionprops(mask, intensity_image=image_HSV)}
    data = pd.DataFrame(data).transpose()
    blob_to_pos = match_to_grid(data['label'], data['centroid'], grid, griddist, d=d, reportAll=reportAll)
    
    #Select only those blobs which have a corresponding grid position
    data = data.loc[[l in blob_to_pos for l in data['label']]]
    
    #Add grid position information to table
    data['row'] = data['label'].map(lambda x: blob_to_pos[x].split('-')[0])
    data['column'] = data['label'].map(lambda x: blob_to_pos[x].split('-')[1])
    
    #Add circularity
    data['circularity'] = (4 * math.pi * data['area']) / (data['perimeter']**2)
    
    #Make qc image
    qc = label2rgb(mask, image=orig_image, bg_label=0)
    
    return (data, qc)
    
def quantify_single_image_Fluorescence(BF_im, Fluo_im, grid, auto, t, d, s, negate, RGB_filter, HSV_filter, reportAll=False, hardImageThreshold=None, hardSizeThreshold=None, localThresh=None):
    '''
    Process a single image to extract colony sizes.
    #BF_im: Bright Field Image, a 3Channels (rgb) image>> used for mask
    #Fluo_im: Fluorescent signal 1 channel >> used for the rest
    '''
    #all all the steps bellow work on a HxWx1 dimensional image, we extract here two different suhc image
    #1) for making the mask (RGB based)
    #2) for grading colony color (HSV based)
    
    image_RGB =  RGB_filter(BF_im)
    
    #Prepare image
    image_RGB = check_and_negate(image_RGB, negate=negate)
    Fluo_im = check_and_negate(Fluo_im, negate=negate)
    
    #Create grid
    if auto:
        grid, griddist = make_grid_auto(image_RGB, grid)
    else:
        grid, griddist = make_grid(grid)
        
    #Make mask
    mask = make_mask(image_RGB, t, s, hardImageThreshold=hardImageThreshold, hardSizeThreshold=hardSizeThreshold, local=localThresh)

    #Measure regionprops
    data = {r.label : {p : r[p] for p in ['label', 'area', 'centroid', 'mean_intensity', 'perimeter']} for r in regionprops(mask, intensity_image=Fluo_im)}
    data = pd.DataFrame(data).transpose()

    blob_to_pos = match_to_grid(data['label'], data['centroid'], grid, griddist, d=d, reportAll=reportAll)
    
    #Select only those blobs which have a corresponding grid position
    data = data.loc[[l in blob_to_pos for l in data['label']]]
    
    #Add grid position information to table
    data['row'] = data['label'].map(lambda x: blob_to_pos[x].split('-')[0])
    data['column'] = data['label'].map(lambda x: blob_to_pos[x].split('-')[1])
    
    #Add circularity
    data['circularity'] = (4 * math.pi * data['area']) / (data['perimeter']**2)
    
    #Make qc image
    qc = label2rgb(mask, image=image_RGB, bg_label=0)
    
    return (data, qc)


    
def quantify_single_image_size(orig_image, grid, auto, t, d, s, negate, reportAll=False, hardImageThreshold=None, hardSizeThreshold=None, localThresh=None):
    '''
    Process a single image to extract colony sizes.
    '''
    
    #Prepare image
    image = check_and_negate(orig_image, negate=negate)
    
    #Create grid
    if auto:
        grid, griddist = make_grid_auto(image, grid)
    else:
        grid, griddist = make_grid(grid)
        
    #Make mask
    mask = make_mask(image, t, s, hardImageThreshold=hardImageThreshold, hardSizeThreshold=hardSizeThreshold, local=localThresh)

    #Measure regionprops
    data = {r.label : {p : r[p] for p in ['label', 'area', 'centroid', 'mean_intensity', 'perimeter']} for r in regionprops(mask, intensity_image=image)}
    data = pd.DataFrame(data).transpose()
    blob_to_pos = match_to_grid(data['label'], data['centroid'], grid, griddist, d=d, reportAll=reportAll)
    
    #Select only those blobs which have a corresponding grid position
    data = data.loc[[l in blob_to_pos for l in data['label']]]
    
    #Add grid position information to table
    data['row'] = data['label'].map(lambda x: blob_to_pos[x].split('-')[0])
    data['column'] = data['label'].map(lambda x: blob_to_pos[x].split('-')[1])
    
    #Add circularity
    data['circularity'] = (4 * math.pi * data['area']) / (data['perimeter']**2)
    
    #Make qc image
    qc = label2rgb(mask, image=orig_image, bg_label=0)
    
    return (data, qc)


def rank_plot(DF, metric,n, ylim=''):
    
    y=DF.groupby('gene').mean()[metric].values
    std=DF.groupby('gene').std()[metric].values
    
    i_sort=np.argsort(y)
    #subset n points to plot
    i_sort= i_sort[np.linspace(0,len(i_sort)-1,n).astype(int)]

    x=np.arange(len(i_sort))
    gene= DF.groupby('gene').mean().index
    plate= DF.groupby('gene').mean()
    df = {'x_pos': x, 'Metric': y[i_sort],'error': std[i_sort], 'error_minus':  std[i_sort], 'gene':gene[i_sort] }
    fig = px.scatter(df, x="x_pos", y="Metric",
                 error_y="error", error_y_minus="error_minus", hover_data=['gene'])

    fig.update_layout(title=metric)
    
    if ylim != '':
        fig.update_layout(yaxis_range=ylim)
    
    fig.show()
    



def hover_plot(X,Y,size,name, xname, yname, sizename, namename, xlim, ylim ):
    df = {xname: X, yname: Y, sizename:size ,namename:name , }
    fig = go.Figure(data=go.Scattergl(x=df[xname],
                                y=df[yname],
                                mode='markers',
                               # marker_color=df[col3name],
                                text=df[namename],
                                marker=dict(size=df[sizename],line_width=1)),
                                layout_xaxis_range=xlim,
                                layout_yaxis_range=ylim,
                                
                   )# hover text goes here

    fig.update_layout(title=xname+' vs '+yname,
                      xaxis_title=xname,
                      yaxis_title=yname,)

    fig.show()
    
    




def check_and_negate(orig_image, negate=True):
    '''
    Check if image is greyscale, convert if it isn't. Convert to float and invert intensities.
    '''
    image = np.copy(orig_image)
    
    #Check if images are grayscale and convert if necessary
    if len(image.shape) == 3:
        warn('Image is not in greyscale, converting before processing')
        image = image.astype(int).mean(axis=2)

    #Convert to float and re-scale to [0,1]            
    image = image.astype(float)
    image = image / 255.0
    
    #Negate images if required
    if negate:
        image = invert(image)
        
    return image




def match_to_grid(labels, centroids, grid, griddist, d=3, reportAll=False):
    '''
    From a list of grid positions and a list of centroids, construct a distance matrix between all pairs and return the best fits as a dictionary.
    '''
    
    #Construct distance matrix as pandas table
    dm = distance.cdist(np.array(list(centroids)), np.array(list(grid.values())), metric='euclidean')
    dm = pd.DataFrame(dm, index=labels, columns=map(lambda x: '-'.join(map(str,x)),grid.keys()))
            
    #Select matches
    dm[dm>(griddist/d)] = np.nan

    if not reportAll:
        #Find best match for each grid position
        dm = dm.idxmin(axis=0)
        dm = dm.dropna().astype(int)
        #Swap index and values
        #There should never be a blob associated to two blob positions since d>2 is enforced in the command line interface
        dm = pd.Series(dm.index.values, index=dm)
        dm = dm.to_dict()        
        
    else:
        #Find best match for each blob
        dm = dm.idxmin(axis=1)
        dm = dm.dropna()
        dm = dm.to_dict()
            
    return dm


def No_filter(Image):
    return Image

def To_grey(Image):
    return   0.2125*Image[:,:,0] + 0.7154*Image[:,:,1] + 0.0721*Image[:,:,2]
 
def White(Image):
    convert=0
    if Image.mean()>1:
        Image=Image/255
        convert=1
    Image = np.power(0.2125*Image[:,:,0] + 0.7154*Image[:,:,1] + 0.0721*Image[:,:,2] ,2)
    return   Image*255 if convert == 1 else Image 

            
def Helen_yellow(image):
    convert=0
    if image.mean()>1:
        image=image/255
        convert=1
    
    image = np.sqrt(np.power(image[:,:,0],1.5)*np.power(image[:,:,1], 1/1.5)) - 0.5*image[:,:,2]
   #image = np.maximum(0, image)
    return image*255 if convert == 1 else image 

def SC_orange(image):
    #parameters adjusted visually with  adjust_filter()
        RG_corr = 0.08#0.04
        R_var=0.04#0.02
        mean_red = 0.7
        mean_green = 0.1
        convert=0
        if image.mean()>1:
            image=image/255
            convert=1
        Sigma=[[R_var, RG_corr, 0],
               [RG_corr, 0.3, 0],
               [0, 0, 0.03],
              ]
        rv = multivariate_normal([mean_red, mean_green, -0.1], Sigma)
        yellowness = np.sqrt(np.power(image[:,:,0],2)*np.power(image[:,:,1], 1/2))-  0.9*image[:,:,2]
        image = np.maximum(yellowness,0.5*rv.pdf(image[:,:,:]) )
        image = np.maximum(0, image)
        return image*255 if convert == 1 else image


def YPD_yellow(image):
    convert=0
    if image.mean()>1:
        image=image/255
        convert=1
    image = np.sqrt(np.power(image[:,:,0],2)*np.power(image[:,:,1], 1/2))-  0.6*image[:,:,2]
    image = np.maximum(0, image)
    return image*255 if convert == 1 else image 


    

def prepare_yellowness_image(orig_image, filter_func):
    '''
    Prepare image for thresholding and analysis. Channels are weightedand summed. The background is estimated by gaussian blur and subtracted. The image is inverted.
    '''
    image = np.copy(orig_image).astype(float)

    #Color channelprint(image.shape) transformations and convert to grey
    image = filter_func(image)

    #2*np.sqrt(np.power(image[:,:,0],3)*np.power(image[:,:,1], 1/3)) - image[:,:,2]
   # 
    #
 
    #np.sqrt(image[:,:,0]*image[:,:,1]) #0.5*image[:,:,1] + 1*image[:,:,2]
    #Convert to float and rescale to range [0,1]
    #I don't think other fancier methods for histogram normalisation are suitable or required since simple thresholding is applied later

    #Estimate background by gaussian. Scale sigma with image area to compensate for different resolutions
    background = gaussian(image, sigma=np.prod(image.shape)/10000, truncate=4) 
    image = image - background #This may contain some negative values

    #Scale image to [0,1] in invert
    image = image.astype(float)
    image = image - np.min(image)
    image = image/np.max(image)
    #image = 1 - image
    
    return image

def Generate_Plate_Map(path):
    '''read txt file storing plate mapping and convert to 384 plate format'''
    file = open(path,'r')
    corpus=file.read().split('\n')
    plate_num=0
    plate_map=dict()
    for line in corpus:
        #print(line)
        if 'Plate' in line:
            plate_num+=1
            r=0
            plate_map[plate_num]=np.empty((32,48), dtype='<U16')
        else:
            c=0
            for gene in line.split(' ')[2:-1]: #[2:-1] gets rid of the two letters and the last letter flanking the row at each line
                plate_map[plate_num][r:r+3,c:c+3]=gene
                c+=2
            r+=2

    return plate_map


def correct_edges2del(DF, metrics ,default_cols=['gene','plate','row','column']):
   
    #Subset data set for ease:
    cols= default_cols
    [cols.append(m ) for m in metrics]
    Data_corrected = DF[cols]


    for metric in metrics: #['mean_intensity.24.YUGN','area.24.YUGN']:
    #create empty column:
        Data_corrected['corrected_' + metric]=  Data_corrected['plate'].copy()*0
        for p in range(0,13):
            plate_data = DF[DF['plate']==p][['gene','plate','row','column',metric]]
            #get rid of corners 
            corners_i =  plate_data['column'].isin([0,47]) & plate_data['row'].isin([0,31])
            plate_data[corners_i ] =0


            #this will be the corrected metric
            plate_data['corrected_' + metric] = plate_data[metric].copy()


            # to correct a group, we scale each colony by:
            # deviding the colony by the median its group and multiplying by the MLM
            
            n_edges = 5
            #compute the middle lawn median
            ML_i = plate_data['column'].isin(np.arange(n_edges,48-n_edges)) & plate_data['row'].isin(np.arange(n_edges,32-n_edges))
            #test
            #plt.imshow(ML_i.values.reshape((32,48)))
            MLM= np.median(plate_data[ML_i][metric].values )#middle lawn median

            #compute the correction needed to be applied to border colonies :
            
            for frame in range(n_edges):
                #finds the data point in the frame
                border_row_out_i = plate_data['column'].isin([0+frame,47-frame]) | plate_data['row'].isin([0+frame,31-frame])
                #gets rid of colonies which are zeroes (this could cause the median to be zero)
                group = plate_data[border_row_out_i ]['corrected_' + metric].values
                group = group[group>0]
                #compute the scale:
                scale = MLM/ np.median(group)
                plate_data.loc[border_row_out_i,'corrected_' + metric] = plate_data.loc[border_row_out_i,'corrected_' + metric]*scale

            Data_corrected.loc[Data_corrected['plate']== p,'corrected_' + metric] = plate_data['corrected_' + metric].values
    return Data_corrected


def correct_edges(DF, metrics ,default_cols=['gene','plate','row','column'], p_format=[32,48], n_edges = 5):
   
    #Subset data set for ease:
    cols= default_cols
    [cols.append(m ) for m in metrics]
    Data_corrected = DF[cols]


    for metric in metrics: #['mean_intensity.24.YUGN','area.24.YUGN']:
    #create empty column:
        Data_corrected['corrected_' + metric]=  Data_corrected['plate'].copy()*0
        for p in range(0,13):
            plate_data = DF[DF['plate']==p][['gene','plate','row','column',metric]]
            #get rid of corners 
            corners_i =  plate_data['column'].isin([0,p_format[1]-1]) & plate_data['row'].isin([0,p_format[0]-1])
            plate_data[corners_i ] =0


            #this will be the corrected metric
            plate_data['corrected_' + metric] = plate_data[metric].copy()


            # to correct a group, we scale each colony by:
            # deviding the colony by the median its group and multiplying by the MLM
            
            #compute the middle lawn median
            ML_i = plate_data['column'].isin(np.arange(n_edges,48-n_edges)) & plate_data['row'].isin(np.arange(n_edges,32-n_edges))
            #we dont want to count the empty spots values in the median so we get rid of them in the filter
            ML_i[plate_data[metric]== 0] = False
            #test
            # plt.imshow(ML_i.values.reshape((32,48)))
            #plt.show()
            MLM= np.median(plate_data[ML_i][metric].values )#middle lawn median
            #compute the correction needed to be applied to border colonies :
            
            for frame in range(n_edges):
                #finds the data point in the frame
                border_row_out_i = plate_data['column'].isin([0+frame,p_format[1]-1-frame]) | plate_data['row'].isin([0+frame,p_format[0]-1-frame])
                #gets rid of colonies which are zeroes (this could cause the median to be zero)
                group = plate_data[border_row_out_i ]['corrected_' + metric].values
                group = group[group>0]
                #compute the scale:
                scale = MLM/ np.median(group)
                plate_data.loc[border_row_out_i,'corrected_' + metric] = plate_data.loc[border_row_out_i,'corrected_' + metric]*scale
                #check:
                #plt.imshow(border_row_out_i.values.reshape((32,48))*600+plate_data['corrected'].values.reshape((32,48)), vmin=0, vmax=600 )
                #plt.show()
                #plt.hist(plate_data[border_row_out_i ]['corrected'].values , alpha=0.3)
                #plt.hist(plate_data[border_row_out_i==0 ]['corrected'].values , alpha=0.3)
                #plt.show()


            #Compute the difference: metric - neighbour_median_metric
         #   padded_plate=np.ones((34,50))*-1#a plate with a one unit margin
         #   padded_plate[1:-1,1:-1] = plate_data['corrected'].values.reshape((32,48))
         #   plate_data['corrected'] =Difference_with_neighbours(padded_plate)


            Data_corrected.loc[Data_corrected['plate']== p,'corrected_' + metric] = plate_data['corrected_' + metric].values
    return Data_corrected

def gene_loc_finder(gene , plate_map):
    '''find the plate, row, col, positions for a gene in a plate map
    counts from 0 '''
    p=[]
    r=[]
    c=[]
    for n, plate in plate_map.items():
        if gene in plate:
            #grid from pyphe counts from 1 so we substract 1
            #if genes are on the same plate we repeat the plate number in p for the number of colonies found:
            [p.append(n-1) for i in np.where(plate==gene)[0]]
            #some genes are found twice in the same plate
            [r.append(R.item()) for R in np.where(plate==gene)[0]]
            [c.append(C.item()) for C in np.where(plate==gene)[1]]
    return p ,r,c

def colony_check(plate_im, row, column, frame_opt):
    '''plot a colony on demand:
    frame_opt: nmbr of px on each side of the center of the colony, should be multiple of 2'''
    #plate=plate+1
    #row=row+1
   # column=column+1
   #
   # plate= AB+'_'+'T'+'-Run-1-Plate-'+str(plate).zfill(3)+ ' - Original.png'
   # plate= 'BTX21.6.21/H24/'+ AB +'/'+ plate
    
    #im=skimage.io.imread(plate_path)
   # im=im[230:720,260:1000, :]
    frame=frame_opt/2 #number of px between center and frame limit
    im_1d=prepare_yellowness_image(plate_im)

    
    grid, griddist = make_grid_auto(im_1d, '32-48')

    up=int(grid[(row,column)][0]-frame)
    up=max(up,0)
    down=int(grid[(row,column)][0]+frame)
    down=min(down, im.shape[0])

    #ledown=ft=int(grid[(row,column)][1]-frame)
    left=int(grid[(row,column)][1]-frame)
    left=max(left,0)
    right=int(grid[(row,column)][1]+frame)
    right=min(right, im.shape[1])
    return im[up:down,left:right,:]

def colony_picker(plate_im, row, column, grid, frame_opt):
    '''return a colony picture
    grid : the grid dictionary returned by make_grid_auto()
    frame_opt: nmbr of px on each side of the center of the colony, should be multiple of 2'''
    #the grid counts from 1:
    row+=1 #to use grid we convert to counting from 1
    column+=1
    frame=frame_opt/2 #number of px between center and frame limit
   
    up=int(grid[(row,column)][0]-frame)
    up=max(up,0)
    down=int(grid[(row,column)][0]+frame)
    down=min(down, plate_im.shape[0])
    #ledown=ft=int(grid[(row,column)][1]-frame)
    left=int(grid[(row,column)][1]-frame)
    left=max(left,0)
    right=int(grid[(row,column)][1]+frame)
    right=min(right, plate_im.shape[1])
    
    #this padding ensures that the return image is always the correct (frame_opt x frame_opt) dimension
    pad = np.ones((frame_opt,frame_opt,3))*0
    colony=  plate_im[up:down,left:right]
    pad[0:colony.shape[0],0:colony.shape[1]]= colony

    return pad

def rank_quadruplicates(DF, Metric, Plate_map, Pic_dict, Rank_opt=''):
    '''returns a list of the pictures of the quadruplicate colonies given in a DF sort by genes 
    the list is sorted in order according to the metric or the Z score of the metric if Rank_opt=Z
     genes might not be found because one 
    !! it doesnt work check if if p+1!=10: needs to be commented out
    '''
    
    n_gene=len(DF)
    frame_opt=30
    reconstituted_plate=[]
    epsilon =10e-9
    if Rank_opt == 'Z':
        Z_score= (  DF[Metric+'_mean'] - DF[Metric+'_mean'].mean())/(DF[Metric+'_std']+epsilon)
        best_genes = DF.iloc[Z_score.argsort][::-1]['gene']
    else:
        best_genes = DF.iloc[DF[Metric+'_mean'].argsort][::-1]['gene']

    gene_list=[]
    for gene in tqdm(best_genes[0:n_gene]):

        
        plates, rows, cols = gene_loc_finder(gene , Plate_map)
        quadruplicate = [np.zeros((frame_opt,frame_opt)),
                         np.zeros((frame_opt,frame_opt)),
                         np.zeros((frame_opt,frame_opt)),
                         np.zeros((frame_opt,frame_opt))  ]

        for i, p in enumerate(plates):
            
            if i <4: #some genes are present several times in the library this avoids picking more than four colonies
                original_plate_pic = Pic_dict[p+1][0]
                grid = Pic_dict[p+1][1]
                

                colony = colony_picker(original_plate_pic, rows[i], cols[i],grid, frame_opt=frame_opt)

                quadruplicate[i] = colony 
        #if len(quadruplicate)==4:
        quadruplicate=np.vstack((np.hstack((quadruplicate[0],quadruplicate[1])),
                                  np.hstack((quadruplicate[2],quadruplicate[3]))))
        reconstituted_plate.append(quadruplicate)
        gene_list.append(gene)
        #else:
          #  print('problem finding (skipped) :',gene)
       # plt.imshow(a)
       # plt.show()
        
    
    return reconstituted_plate, gene_list

def Show_Ranked_Colonies(Reconstituted_plate,Gene_list, Start, Stop):
    fig, axs= plt.subplots(8,12, figsize=(10,10))
    fig.tight_layout(pad=-1)
    [axi.set_axis_off() for axi in axs.ravel()]
    #plt.subplots_adjust(wspace=0.01, hspace=0)
    fig.tight_layout()
    axs=axs.flatten()
    for j , i in enumerate( range(Start,Stop)):
            axs[j].imshow(Reconstituted_plate[i] ,vmin=0,vmax=255)
            axs[j].set_title(Gene_list[i], fontsize = 8)
    plt.show()
    

    
def group_by_gene(DF0, metrics):
    '''group each metric by gene, calculate the mean, std, and count the number of available replicates'''
    DF=DF0.copy()
    def colonies_per_gene(x):
        return (x.isna()!=1).sum()

    cols={}
    cols['gene'] =  DF.groupby('gene').mean().index.values

    for i, metric in enumerate(metrics):
        cols[metric+ '_mean'] =  DF.groupby('gene').mean()[metric].values
        cols[metric + '_std'] = DF.groupby('gene').std()[metric].values
        cols[metric + '_count'] = DF.groupby('gene')[metric].agg(colonies_per_gene).values

    columns = np.hstack([ cols[j].reshape(-1,1) for j in cols.keys()])
    col_names = list(cols.keys())
    return pd.DataFrame(columns,columns=col_names )


## Outlier filtering functions:
## add the Zscores to the DF

def compute_Zthresh():
    '''finds the Zthresholds over which a colony is declared outlier 
    returns them as dict'''
    ## outlier explaination
    #https://www.graphpad.com/support/faqid/1598/
    # https://www.itl.nist.gov/div898/handbook/eda/section3/eda35h1.htm
    # for each gene we want to get rid of outliers in which CRISPA failed:
    # these colonies will be more white
    #we compute the G score for one sided test:
    #G = (genne_mean - gene_min)/gene_std

    #the null hypothesis: the minimal colony is not an outlier
    #the alternative hypothesis the minimal colony is an outlier

    #for a significance level alpha we reject the null iff:
    # G > ((N-1)/np.sqrt(N))*np.sqrt((cv**2)/(N-2 +(cv**2)))

    #N is the number of colonies
    #cv is the critical value for which the P(t<cv) <= alpha/N in the t distribution with df = N-2 

    #for example we set:
    alpha=0.05
    N=4
    df = N-2
    for cv in np.arange(2,25,0.001):
        P =   1- t.cdf(cv, df)
        if P < alpha/N:
            #print('proba threshold:', P, 'critical value',cv)
            break

    #we find for N=4, cv=19.1 and for N=3 cv= 6.21


    thresh = ((N-1)/np.sqrt(N))*np.sqrt((cv**2)/(N-2 +(cv**2)))

    #print('for', N ,'colonies the Zscore must be greater than', thresh)

    # however, when calculating the Z_score, the std will be boosted by the fact that colonies are not on the same plate,
    #this increases the replicate std
    #std_fold_increase =Data_corrected.groupby('gene')[metric].std().mean()/Data_corrected.groupby('gene')['Helen_mean_intensity'].std().mean()
    #print(std_fold_increase )

    #therefore an immediate drawback of randomizing colonies accross plate is that it makes harder to spot outliers

    Z_thresh ={}

    #for example we set:
    alpha=0.05
    for N in range(3,16):
        df = N-2
        for cv in np.arange(0,25,0.001):
            P =   1- t.cdf(cv, df)
            if P < alpha/N:
                break

        thresh = ((N-1)/np.sqrt(N))*np.sqrt((cv**2)/(N-2 +(cv**2)))
        Z_thresh[N]= thresh
   # print('colony',N, 'proba threshold:', P, 'critical value',cv, ' Z threshold', thresh)
    return Z_thresh

Z_thresh = compute_Zthresh() 

def colonies_per_gene(x):
    return (x.isna()!=1).sum()

def Filter_3(DF_start, metrics):
    '''return a DF where reads for genes with less than 3 colonies is NaNed'''
    DF = DF_start.copy()
    for metric in tqdm(metrics):
        colony_count = DF.groupby('gene')[metric].agg(colonies_per_gene)
        #gene reads that should be declared nan because they have less than 3 colonies:
        genes2nan = colony_count[colony_count < 3]
        DF.at[DF['gene'].isin( genes2nan.index),metric] = np.nan
    return DF


def Compute_Zscore(DF, metrics):
    '''compute the per gene Zscore for each metrics (DF column)'''
    DF=DF.copy()
    def Zscore_func(x):
        '''returns the index and value for a given DF column'''

        Z = (x.mean()-x)/(x.std())
        
        return [list(x.index),[z for z in Z]] #list(x.index)
    

    for metric in tqdm(metrics):
        #this object is a list storing [[colony rows][colony Zscores]]
        score=DF.groupby('gene')[metric].agg(Zscore_func).values

        #this convert the score object in a np.array of the same len as DF
        Zscore = np.empty(len(DF))
        Zscore[:] = np.nan

        for S in score:
            Zscore[S[0]] = S[1] 
        
        DF[metric+'_Z']=Zscore
        
    return DF


def associate_Zthresh(N):
    '''associates the 5% rejection threshold for a number of quadruplicate'''
    if N == 3 :
        thresh = 1.15
    elif N == 4 :
        thresh = 1.46
    else:
        thresh = np.nan
    return thresh


def associate_Zthresh(N, Z_tresh = Z_thresh ):
    '''associates the rejection threshold for a number of quadruplicate'''
    if N >=3 and N <16:
        thresh = Z_thresh[N]
    else:
        thresh = np.nan
    return thresh
    
def Filter_Z(DF, metrics, Z_tresh = Z_thresh):
    '''return a DF where reads for genes with a Zscore > Z is NaNed'''
    
    DF = DF.copy()
    for metric in tqdm(metrics):
        #this associate a colony number to each row of DF
        N=DF.groupby('gene')[metric].agg(colonies_per_gene) #1) find colony number for each gene
        N= N[DF['gene']].reset_index() #re-associated to each gene of DF its colony number
        Z_tresh = N[metric].agg(associate_Zthresh)
      #  print(Z)
      #  print(DF['gene'])
      #  print(abs(DF[metric+'_Z']))
        colony2nan = DF[metric+'_Z'] > Z_tresh #abs()
     #   print(metric , 'Z outliers......' ,colony2nan.sum())
        DF.at[colony2nan,metric] = np.nan
       
    return DF

def Filter_P(DF, metrics):
    '''return a DF where reads for genes with a P > 0.05 is NaNed
    see https://www.itl.nist.gov/div898/handbook/eda/section3/eda35h1.htm'''

    DF = DF.copy()
    for metric in tqdm(metrics):
        #this associate a colony number to each row of DF
        colony2nan = abs(DF[metric+'_P']) < 0.05
        print(metric , 'Z outliers......' ,colony2nan.sum())
        
        #check
        #N=DF.groupby('gene')[metric].agg(colonies_per_gene) #1) find colony number for each gene
        #N= N[DF['gene']].values#N[DF['gene']].reset_index() #re-associated to each gene of DF its colony number
        #print(N[colony2nan])
        
        DF.at[colony2nan,metric] = np.nan 
        
        #check
        #N=DF.groupby('gene')[metric].agg(colonies_per_gene) #1) find colony number for each gene
        #N= N[DF['gene']].values#N[DF['gene']].reset_index() #re-associated to each gene of DF its colony number
        #print(N[colony2nan])


    return DF


def hover_plot(X,Y,S, L, C,  xname, yname,sname, labelname, cname ):
    '''X,Y plotting dimension 1 and 2
    S marker size
    '''
    df = {xname: X, yname: Y, sname:S , labelname: L, cname:C}
    fig = go.Figure(data=go.Scattergl(x=df[xname],
                                y=df[yname],
                                mode='markers',
                               # marker_color=df[col3name],
                                text=df[labelname],
                                marker=dict(size=df[sname]*20,
                                            color=df[cname],
                                            line_width=1)))# hover text goes here

    fig.update_layout(title=xname+' vs '+yname,
                     xaxis_title=xname,
                     yaxis_title=yname)

    fig.show()
    
    



def Rotate_Plate(Im, step):
    '''Finds the best angle that minimise the entropy of the sum along dim 1 of a plate picture.
    Rationale: the highest entropy will be found when colonies are perfectly aligned
    will turn the image -3step to +3 steps and return the image rotated at best step'''
    pltshow=0
    #quick search:
    E=[]
    Angle = []

    for n, angle in enumerate(np.arange(-3*step,3*step+step,2*step)):
        rotated= rotate(Im, -angle)
        E.append(entropy(rotated.sum( axis=1)))
        Angle.append(angle)

    min_index = E.index(min(E))
    Best_Angle = Angle[min_index]
    #check
    if pltshow==1:
        plt.imshow(rotate(Im, -Best_Angle) )
        plt.show()
        plt.plot(Angle,E)
        plt.show()
    return rotate(Im, -Best_Angle), -Best_Angle


def Velber_Preprocess(im):
    '''The plates in the Velber are not horizontal nor centered
    this functions aligns and crop plate for extraction
    the plate needs to have a colony array clearly visible'''
    check=0 #plots for debugging
    
    #first quick alignment
    im, angle = Rotate_Plate(im , 0.4)

    #cropping:
    contours = find_contours(im>np.mean(im), fully_connected='low')
    ##find the longest contour:
    #list, is true if element is longest contour, otherwise false
    Im_longest = [len(C)== max(len(c) for c in contours) for C in contours] 
    #find indice of true
    i_longest=[i for i, x in enumerate(Im_longest) if x][0]
    
    contour = contours[i_longest] # I think skimage return the longest contour first
    #place x in first col and y in second:
    contour = np.vstack([contour[:,1],contour[:,0]]).T 
    
    BR=contour.max(axis=0) #bottom right 
    TL=contour.min(axis=0) #top left

    if check ==1:
        plt.imshow(im> np.mean(im))
        plt.scatter(contour[:,0],contour[:,1], s=20,c='red')
        plt.show()
        fig, ax = plt.subplots( figsize=(20,20))
        ax.imshow(im, interpolation='nearest', cmap=plt.cm.gray)
        ax.scatter(BR[0],BR[1], s=20, c='r')
        ax.scatter(TL[0],TL[1], s=20, c='cyan')
        plt.show()
    #cropping:
    top = int(TL[1]+70)
    bottom = int(BR[1]-70)
    left = int(TL[0]+80)
    right = int(BR[0]-80)
    im = im[top:bottom,left:right]
    return im,[angle,top,bottom,left,right]


