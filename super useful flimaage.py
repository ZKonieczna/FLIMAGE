#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 12:22:57 2021
@author: Mathew
"""

from os.path import dirname, join as pjoin
import scipy.io as sio
import numpy as np
from skimage.io import imread
from skimage import filters,measure
import matplotlib.pyplot as plt
import napari
from scipy.interpolate import make_interp_spline, BSpline
from scipy.spatial import distance



# Convert to wavelength- thede are from the fits to the TS Bead Data. 
m=0.5446
c=465.11

# Read the files 

mat_fname=r"\\chem-mh-store\D\Members\Zuzanna Konieczna\tisssue flim 02052024\20240418_ZK-26-Dis2Well8_h1\Lifetime_Data\LifetimeImageData.mat"
mat_contents = sio.loadmat(mat_fname,squeeze_me=True)
lifetimes=mat_contents['lifetimeImageData']


mat_fname=r"\\chem-mh-store\D\Members\Zuzanna Konieczna\tisssue flim 02052024\20240418_ZK-26-Dis2Well8_h1\Lifetime_Data\LifetimeAlphaData.mat"
mat_contents2 = sio.loadmat(mat_fname,squeeze_me=True)
intensities=mat_contents2['lifetimeAlphaData']


# This is to make an summed intensity image over all wavelengths to perfom the thresholding on
sum_int=np.sum(intensities[0:512],axis=0)
plt.imshow(sum_int)
plt.colorbar()
plt.savefig('WELL8.jpg', dpi=1200)
plt.show()


# The below just thresholds the image based on intensity value - could also use Otsu method
thresh=50000
binary_im=sum_int>thresh
plt.imshow(binary_im)
plt.show()

# Now get the stack only with the thresholded intensities or lifetimes present:
thresholded_intensities=binary_im*intensities
thresholded_lifetimes=binary_im*lifetimes

thresholded_intenisties_sum=np.sum(thresholded_intensities, axis=0)
twod_intensities=thresholded_intenisties_sum

plt.imshow(twod_intensities)
plt.colorbar()
plt.show()


#get an intensity spectrum for selected pixels

intensities_only_thresh=intensities*binary_im
intensity_wl=[]
int_sdev=[]
wl=[]
for i in range(0,512):
    wavelength_val=i*m+c
    plane=intensities_only_thresh[i]
    plane_list=plane.flatten()
    values_only=plane_list[plane_list>0]
    intensity_mean=values_only.mean()
    intensity_sdev=values_only.std()
    
    intensity_wl.append(intensity_mean)
    int_sdev.append(intensity_sdev)
    wl.append(wavelength_val)
    
    
plt.plot(wl,intensity_wl)

#smooth out the curve to get rid of noise

ave_int=np.asarray(intensity_wl) #change list to np array
int_sdev_new=np.asarray(int_sdev)
wl_int=np.asarray(wl)


wlnew=np.linspace(wl_int.min(), wl_int.max(), 100) #generate new x axis for smooth curve, changing the number changes how smooth the graph will appear - smaller number means smoother curve

spl=make_interp_spline(wl_int, ave_int, k=3) #interpolate 
ave_int_smooth=spl(wlnew) 

spl_sdev=make_interp_spline(wl_int, int_sdev_new, k=3) #interpolate stdev for graph
sdev_smooth=spl_sdev(wlnew)

plt.plot(wlnew, ave_int_smooth) #use new parameters
plt.fill_between(wlnew,ave_int_smooth-sdev_smooth, ave_int_smooth+sdev_smooth,alpha=0.1, edgecolor='#4b4b4b', facecolor='#4b4b4b', antialiased=True) #set lower and upper limits (sdev) and fill colour/intensity
plt.xlim([460,750])
plt.savefig('Spectrum_smooth.pdf', dpi=1200)
plt.show()


#normalise intensities for alpha masking

twod_intensities=twod_intensities/twod_intensities.max() 
plt.imshow(twod_intensities)
plt.colorbar() #sanity check - max val should read one
plt.show()


range_lifetimes=thresholded_lifetimes[60:300] #define wavelengths of interest in lifetime
twod_lifetime=np.mean(range_lifetimes,axis=0) #calculate mean lifetime across the wavelengths

plt.imshow(twod_lifetime, cmap='jet_r', vmin=0, vmax=1.5) #use to adjust look up table or scale bars
plt.colorbar()
plt.show()

#plot alpha-masked lifetime image

fig, ax = plt.subplots(1)
ax.set_facecolor('black')

m=ax.imshow(twod_lifetime,cmap='plasma_r',vmin=1,vmax=1.5,alpha=twod_intensities)
fig.colorbar(m)
fig.savefig('ZK26well8.jpg',dpi=1200)
plt.show()

#plot histogram of all non-zero lifetimes from the image

list_lifetimes=twod_lifetime.flatten() #generate list of lifetimes
real_lifteimes=list_lifetimes[(list_lifetimes>0)] #set condition
plt.hist(real_lifteimes, bins = 30,range=[0.6,1.6], rwidth=0.9,ec='black',color='darkmagenta',alpha=0.8)
plt.ylabel('Number of pixels')
plt.xlabel('Lifetime (ns)')
#plt.title(textstr)
plt.savefig('ZK26well8_hist.pdf', dpi=1200)
plt.show()

mask=twod_lifetime>0.7
twod_intensities=twod_intensities*mask
plt.imshow(twod_intensities)
plt.savefig('WELL8_masked_short_ones.jpg', dpi=1200)
plt.colorbar() #sanity check - max val should read one
plt.show()





#get lifetime across spectrum for selected pixels

lifetimes_only_thresh=lifetimes*binary_im
lifetime_wl=[]
lifetime_sdev=[]
wl=[]
for i in range(0,512):
    wavelength_val=i*m+c
    plane=lifetimes_only_thresh[i]
    plane_list=plane.flatten()
    values_only=plane_list[plane_list>0]
    lifetime_mean=values_only.mean()
    lifetime_stdev=values_only.std()
    
    lifetime_wl.append(lifetime_mean)
    lifetime_sdev.append(lifetime_stdev)
    wl.append(wavelength_val)

lifetime_wl_new=np.asarray(lifetime_wl) #change list to np array
lifetime_sdev_new=np.asarray(lifetime_sdev)
#wl_int=np.asarray(wl)
    
    
plt.plot(wl,lifetime_wl)
plt.fill_between(wl,lifetime_wl_new-lifetime_sdev_new, lifetime_wl_new+lifetime_sdev_new,alpha=0.1, edgecolor='#4b4b4b', facecolor='#4b4b4b', antialiased=True) #set lower and upper limits (sdev) and fill colour/intensity
plt.xlim([500,650])
plt.ylim([0,15])
#plt.savefig('Spectrum_smooth.pdf', dpi=1200)
plt.show()

