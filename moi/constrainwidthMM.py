"""
Function to constrain widths to hypsometric curve

  Based on FLaPE-Byrd functions

  by Mike, October, 2025
  ported to offline summer 26

"""

import numpy as np
import sys

def ConstrainWidth(h,w,area_fit,nt):

    hhat=np.full( (nt,) ,np.nan)
    what=np.full( (nt,) ,np.nan)
    #nt=len(hhat)
    #print('nt=',nt)
    #print(np.shape(h))

    h_break=area_fit['h_break']
    #print(h_break)
    if np.any(np.isnan(h_break)):
        constrain_mode=1
        print('bad data in height-width fits. set widths equal to average value')
    else:
        constrain_mode=0

    if constrain_mode==0:
        for j in np.arange(nt):
            if h[j] < h_break[0]:
                sd=0
            elif h[j] >= h_break[0] and h[j]<h_break[1]:
                sd=1
            else:
                sd=2
            p0=area_fit['fit_coeffs'][1,sd,0]  #intercept
            p1=area_fit['fit_coeffs'][0,sd,0]  #slope

            hhat[j]=h[j]
            what[j] = p0 + p1 * hhat[j]
    if any(what<0):
        print('hypsometric constraint configured incorrectly and some widths were negative. setting all widths to average value')
        constrain_mode=1
    if constrain_mode ==1:
        what=np.full( (nt,), np.nanmean(w) )
    print('widths constrained')
    return hhat,what
