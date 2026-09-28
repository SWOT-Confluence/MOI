#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 04:50:53 2021

@author: mtd
"""

from numpy import inf,sqrt,std,zeros_like,log

class FlowLaws:
    
    def __init__(self,dA,W,S,H,name='No Name'):
        self.dA=dA
        self.W=W
        self.S=S       
        self.H=H       #plan is to switch these to Obs                
        
        self.params=[]
        self.init_params=[]                

        self.name=name
        
class MWACN(FlowLaws):
    # this flow law is Manning's equation, wide-river approximation, area-formulation, 
    #   constant friction coefficient, no channel shape assumption: MWACN
    def __init__(self,dA,W,S,H):
        super().__init__(dA,W,S,H)        
        
    def CalcQ(self,params):
        Q=1/params[0]*(params[1]+self.dA)**(5/3)*self.W**(-2/3)*self.S**(1/2)
        return Q
    def GetInitParams(self):
        #etc
        init_params=[.03, -min(self.dA)+1+std(self.dA)]
        return init_params
        #etc
    def GetParamBounds(self):
        param_bounds=( (.001, 1)  , (-min(self.dA)+1,inf) )
        return param_bounds
    def Jacobian(self,params,Qt):
        dydp=zeros_like(params)                  
        dydp[0]=2*sum(((self.dA+params[1])**(5/3)*(Qt*sqrt(self.S)*self.W**(2/3)*params[0]-self.S*(self.dA+params[1])**(5/3)))/(self.W**(4/3)*params[0]**3))
        dydp[1]=(10/3)*sum(((self.dA+params[1])**(2/3)*(self.S*(self.dA+params[1])**(5/3)-Qt*sqrt(self.S)*self.W**(2/3)*params[0])/(self.W**(4/3)*params[0]**2)))
        return dydp

class MWAPN(FlowLaws):
    # this flow law is Manning's equation, wide-river approximation, area-formulation, 
    #   powerlaw  friction coefficient, no channel shape assumption: MWAPN
    def __init__(self,dA,W,S,H):
        super().__init__(dA,W,S,H)     
    def CalcQ(self,params):
        n=params[0]*((params[1]+self.dA)/self.W)**params[2]
        Q=1/n*(params[1]+self.dA)**(5/3)*self.W**(-2/3)*self.S**(1/2)        
        
        return Q
    def GetInitParams(self):
        min_dA=min(self.dA)
        if min_dA < 0:
            Abar_min=-min(self.dA)+1
        else:
            Abar_min=1
        #etc
        # init_params=[.03, -min(self.dA)+1+std(self.dA),1]
        #init_params=[.03, -min(self.dA)+1+std(self.dA),-1]
        init_params=[.03, Abar_min+std(self.dA),1]
        return init_params       
    def GetParamBounds(self):
        min_dA=min(self.dA)
        if min_dA < 0:
            Abar_min=-min(self.dA)+1
        else:
            Abar_min=1        
        #etc
        param_bounds=( (.001, 1)  , (Abar_min,inf), (-inf,inf) )
        return param_bounds
    def Jacobian(self,params,Qt):
        dydp=zeros_like(params)                  
        dydp[0]=sum((2*(self.dA+params[1])**(5/3)*sqrt(self.S)*(Qt-((self.dA+params[1])**(5/3)*sqrt(self.S))/(((self.dA+params[1])/self.W)**params[2]*self.W**(2/3)*params[0])))/(((self.dA+ params[1])/self.W)**params[2]*self.W**(2/3)*params[0]**2))
        dydp[1]=sum(2*((sqrt(self.S)*params[2]*(params[1]+self.dA)**(2/3))/(self.W**(2/3)*params[0]*((params[1]+self.dA)/self.W)**params[2])-(5*sqrt(self.S)*(params[1]+self.dA)**(2/3))/(3*self.W**(2/3)*params[0]*((params[1]+self.dA)/self.W)**params[2]))*(Qt-(sqrt(self.S)*(params[1]+self.dA)**(5/3))/(self.W**(2/3)*params[0]*((params[1]+self.dA)/self.W)**params[2])))
        dydp[2]=sum((2*(self.dA+params[1])**(5/3)*sqrt(self.S)*log((self.dA+params[1])/self.W)*(Qt-((self.dA+params[1])**(5/3)*sqrt(self.S))/(((self.dA+params[1])/self.W)**params[2]*self.W**(2/3)*params[0])))/(((self.dA+params[1])/self.W)**params[2]*self.W**(2/3)*params[0]))
        return dydp

class MWHFN(FlowLaws):
    # this flow law is Manning's equation, height only, fixed n: MWHCN
    # params= H0
    def __init__(self,dA,W,S,H):
        super().__init__(dA,W,S,H)     
    def CalcQ(self,params):
        Q=1/0.03*(self.H-params[0])**(5/3)*self.W*self.S**(1/2)
        return Q
    def GetInitParams(self):
        #etc
        H0max=min(self.H)-0.1
        init_params=[H0max-1.0] 
        return init_params       
    def GetParamBounds(self):
        #etc
        param_bounds=( (-inf,min(self.H)-0.1) )
        return param_bounds               

