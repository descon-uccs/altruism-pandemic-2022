# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 15:36:49 2020

@author: Philip Brown
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import lambertw

def Rinf(xi,R0=2,eta=.001,C=0.01) :
    w = np.real(lambertw(-R0*(1-eta)*xi*np.exp(-R0*xi)))
    return xi+w/R0

def deriv(func,xi,window=0.000001,R0=2,eta=0.001,C=0.01) :
    return (func(xi+window,R0=R0,eta=eta,C=C) -
             func(xi,R0=R0,eta=eta,C=C))/window
            
def Rderiv(xi,R0=2,eta=0.001,C=0.01,window=0.0000001) :
    return deriv(Rinf,xi,window,R0=R0,eta=eta,C=C)
            
def Rdderiv(xi,R0=2,eta=0.001,C=0.01,window=0.0000001) :
    return deriv(Rderiv,xi,window,R0=R0,eta=eta,C=C)

def Rprob(xi,R0=2,eta=0.001,C=0.01) :
    return Rinf(xi,R0,eta)/xi

P = Rprob

def Pdown(xi,R0=2,eta=0.001,C=0.01) :
    return 2*(1-P(xi,R0,eta,C))

## numerical derivatives:

def Pderiv(xi,R0=2,eta=0.001,C=0.01,window=0.0000001) :
    return deriv(P,xi,window,R0=R0,eta=eta,C=C)

def Pdderiv(xi,R0=2,eta=0.001,C=0.01,window=0.000001) :
    return deriv(Pderiv,xi,window,R0=R0,eta=eta,C=C)
    
    

## derivatives from implicit stuff in onenote:
    
def PderivImp(xi,R0=2,eta=0.001,C=0.01) :
    # based on Full Bore on Derivatives Page; red star
    RinfinityHere = Rinf(xi,R0,eta,C=C)
    num = (1-eta)*R0*P(xi,R0,eta,C)
    denom = (np.exp(R0*RinfinityHere)-(1-eta)*R0*xi)
    return num/denom

def RderivImp(xi,R0=2,eta=0.001,C=0.01) :
    # based on Full Bore on Derivatives Page; double red star
    RinfinityHere = Rinf(xi,R0,eta,C)
    num = 1-(1-eta)*np.exp(-R0*RinfinityHere)
    denom = 1-(1-eta)*xi*R0*np.exp(-R0*RinfinityHere)
    return num/denom
    
def PdderivImp(xi,R0=2,eta=0.001,C=0.01) :
    # based on Full Bore on Derivatives Page; triple red star
    RinfinityHere = Rinf(xi,R0,eta,C)
    RinfPrimeHere = RderivImp(xi,R0,eta,C)
    Phere = P(xi,R0,eta,C)
    num = (1-eta)*R0*R0*Phere
    denom = (np.exp(R0*RinfinityHere)-(1-eta)*R0*xi)**2
    parens = 2*(1-eta) - RinfPrimeHere*np.exp(R0*RinfinityHere)
    return num/denom*parens

def PdderivImpEarly(xi,R0=2,eta=0.001,C=0.01) :
    # based on Full Bore on Derivatives Page; circled in RAINBOW
    RinfinityHere = Rinf(xi,R0,eta,C)
    RinfPrimeHere = RderivImp(xi,R0,eta,C)
    Phere = P(xi,R0,eta,C)
    PPrimehere = PderivImp(xi,R0,eta,C)
    exp = np.exp(R0*RinfinityHere)
    
    num1 = (1-eta)*R0*PPrimehere
    denom1 = exp - (1-eta)*R0*xi
    
    num2 = (1-eta)*R0*Phere
    denom2 = denom1**2
    
    parens = R0*RinfPrimeHere*exp-(1-eta)*R0
    
    return num1/denom1 - num2/denom2*parens
    
def RinfPrimeExp(xi,R0=2,eta=0.001,C=0.01) :
    RinfinityHere = Rinf(xi,R0,eta,C)
    RinfPrimeHere = RderivImp(xi,R0,eta,C)
    return RinfPrimeHere*np.exp(R0*RinfinityHere)

def S(xi,R0=2,eta=0.001,C=0.01) :
    return 2*(1-eta)-RinfPrimeExp(xi,R0,eta,C)

#def PdderivImp()
    
def f(xi,R0=2,eta=0.001,C=0.01) :
    return C/xi;

def cost(xi,R0=2,eta=.001,C=0.01) :
    # this is Js in paper
    return Rprob(xi,R0,eta,C) + f(xi,R0,eta,C)

def costs(xlist,R0=2,eta=0.001,C=0.01) :
    return [cost(x,R0=2,eta=eta,C=C) for x in xlist]
            
def costderiv(xi,R0=2,eta=0.001,C=0.01,window=0.0000001) :
    return deriv(cost,xi,window,R0=R0,eta=eta,C=C)

def Cost(numInUse,R0=2,eta=.001,C=0.01) :
    # assuming all used locations have equal xi
    return cost(1/numInUse,R0=R0,eta=eta,C=C) 

def Cost2(xlist,R0=2,eta=0.001,C=0.01) :
    # xlist contains list of location densities
    return C*len(xlist) + sum([Rinf(x,R0=R0,eta=eta,C=C) for x in xlist])

def findOptimal(R0=2,eta=0.001,C=0.01) :
    # idea here: starting with 1, walk up until you're at twice R0 and then return the best you've seen
    # Note: fails when C is very small
    argmin = 1
    min_cost = Cost(1,R0,eta,C)
    i = 2
    while True :
        new_cost = Cost(i,R0,eta,C) # use the Cost(numLocations) method
        if new_cost >= min_cost :
            if i > R0*2 :
                return argmin, min_cost
        else :
            min_cost = new_cost
            argmin = i
        i += 1
        
def isESS(numLoc,R0=2,eta=0.001,C=0.01) :
    # is the allocation with 1/numLoc people at each of numLoc locations 
    # an ESS? To see, inspect the slope of the location cost function.
    # if nonincreasing at this allocation, this is not an ESS.
    return numLoc == 1 or cost(1/numLoc,R0,eta,C) <= cost(1/numLoc*1.0001,R0,eta,C)
         
def altPoS(R0=2,eta=0.001,C=0.01) :
    # first check if optimal is an altruistic ESS; if so, ret 1
    optLoc, optcost = findOptimal(R0,eta,C)
    if isAltESS(optLoc,R0,eta,C) :
        return 1.0
    else:
        # next walk up from 1 to R0, return the worst AltESS
        # locations = [i for i in range(1,1+int(np.ceil(R0*10))) if i is not optLoc]
        bestESS = np.inf
        numLoc = 1
        while True :
            if isAltESS(numLoc,R0,eta,C) :
                bestESS = min(bestESS,Cost(numLoc,R0,eta,C))
            if numLoc>R0 : # after R0, if we start increasing, we're done
                if Cost(numLoc,R0,eta,C) > bestESS :
                    return bestESS/optcost
            numLoc += 1

def isAltESS(numLoc,R0=2,eta=0.001,C=0.01) :
    # 2 conditions: nobody wants to be alone, and Rdderiv>0
    xi = 1/numLoc
    happyTogether = Rderiv(xi,R0=R0,eta=eta,C=C) <= eta+C
    if happyTogether :
        return deriv(Rderiv,xi,R0=R0,eta=eta,C=C) > 0
    return False # if we aren't happy together, we're not ESS
        

def derivRinf(xi,R0=2,eta=0.001,C=0.01) :
    # note: this function may not be correct at all
    lambertInput = -R0*(1-eta)*xi*np.exp(-R0*xi)
    w = np.real(lambertw(lambertInput))
    expTerm = (1-eta)*(1-R0*xi)*np.exp(-R0*xi)*w
    return 1-expTerm+w/(lambertInput*R0)/(1+w)
    

def plotRinf(R0,eta=0.001,C=0.01) :
    xx = np.arange(0.0,1.01,0.01)
    RRinf = np.empty_like(xx)
    for i in range(len(xx)) :
        RRinf[i] = Rinf(xx[i],R0,eta)
    plt.plot(xx,RRinf)
    
def plotRprob(R0,eta=0.001) :
    xx = np.arange(0.01,1.01,0.01)
    Rprob = np.empty_like(xx)
    for i in range(len(xx)) :
        Rprob[i] = Rprob(xx[i],R0,eta)
    plt.plot(xx,Rprob)
    
    
def plotit(func,R0=2,eta=0.001,C=0.01,numPoints=100,ax=None,xrange=(0,1),ylim=(0,2),label=None,lw=5,ls='-') :
    if ax is None:
        fig,ax = plt.subplots()
    xx = np.linspace(xrange[0]+0.001,xrange[1],numPoints)
    toplot = np.empty_like(xx)
    for i in range(len(xx)) :
        toplot[i] = func(xx[i],R0,eta,C)
    ax.plot(xx,toplot,label=label,lw=lw,ls=ls)
    plt.ylim(ylim)
    plt.xlim((0,1))
    return ax
    
def plotCost(R0=2,eta=0.001,C=0.01,numPoints=10,fignum=13,integer=True) :
    if integer:
        numnum = np.arange(1.,numPoints+1)
    else:
        numnum = np.linspace(1,10,100)
    toplot = np.empty_like(numnum)
    for i in range(len(numnum)) :
        toplot[i] = Cost(numnum[i],R0,eta,C)
    plt.figure(fignum)
    plt.plot(numnum,toplot)
    return numnum, toplot
    
    

if __name__=="__main__" :
    plt.close('all')
    numPoints = 1000
    lw = 3
    eta2 = 0.05
    max_locations = 12
    locations_to_check = np.arange(1,max_locations)
    ax = plotit(cost,R0=5,C=0.2,eta=eta2,numPoints=numPoints,lw=lw,ls='--',label='R0=5, eta='+str(eta2)+', C=0.2')
    plotit(cost,R0=5,C=0.2,numPoints=numPoints,lw=lw,ax=ax,label='R0=5, eta=0.001, C=0.2')
    plotit(cost,eta=eta2,numPoints=numPoints,lw=lw,ls='--',ax=ax,label='R0=2, eta='+str(eta2)+', C=0.01')
    plotit(cost,numPoints=numPoints,lw=lw,label='R0=2, eta=0.001, C=0.01',ax=ax)
    ax.legend(handlelength=3.5,fontsize=16)
    plt.xlabel('Population Density',fontsize=16)
    plt.xticks(fontsize=16)
    plt.ylabel('Cost',fontsize=16)
    plt.yticks(fontsize=16)
    
    ESS1x = []
    ESS1y = []
    aESS1x = []
    aESS1y = []
    for n in locations_to_check :
        if isESS(n,R0=5,C=0.2,eta=eta2) :
            ESS1x.append(1/n)
            ESS1y.append(cost(1/n,R0=5,C=0.2,eta=eta2))
        if isAltESS(n,R0=5,C=0.2,eta=eta2) :
            aESS1x.append(1/n)
            aESS1y.append(cost(1/n,R0=5,C=0.2,eta=eta2))
    plt.scatter(ESS1x,ESS1y,s=80,marker='o',color='#1f77b4')
    plt.scatter(aESS1x,aESS1y,s=160,marker="*",color='#1f77b4')
    
    ESS2x = []
    ESS2y = []
    aESS2x = []
    aESS2y = []
    for n in locations_to_check :
        if isESS(n,R0=5,C=0.2) :
            ESS2x.append(1/n)
            ESS2y.append(cost(1/n,R0=5,C=0.2))
        if isAltESS(n,R0=5,C=0.2) :
            aESS2x.append(1/n)
            aESS2y.append(cost(1/n,R0=5,C=0.2))
    plt.scatter(ESS2x,ESS2y,s=80,marker='o',color='#ff7f0e')
    plt.scatter(aESS2x,aESS2y,s=160,marker="*",color='#ff7f0e')
    
    
    max_locations = 200
    locations_to_check = [1,2,3,4,5] + [int(1/i) for i in np.arange(.2,.001,-.01)]
    
    ESS3x = []
    ESS3y = []
    aESS3x = []
    aESS3y = []
    for n in locations_to_check :
        if isESS(n,R0=2,eta=eta2) :
            ESS3x.append(1/n)
            ESS3y.append(cost(1/n,R0=2,eta=eta2))
        if isAltESS(n,R0=2,eta=eta2) :
            aESS3x.append(1/n)
            aESS3y.append(cost(1/n,R0=2,eta=eta2))
    plt.scatter(ESS3x,ESS3y,s=80,marker='o',color='#2ca02c')
    plt.scatter(aESS3x,aESS3y,s=160,marker="*",color='#2ca02c')
    
    ESS4x = []
    ESS4y = []
    aESS4x = []
    aESS4y = []
    for n in locations_to_check :
        if isESS(n) :
            ESS4x.append(1/n)
            ESS4y.append(cost(1/n))
        if isAltESS(n) :
            aESS4x.append(1/n)
            aESS4y.append(cost(1/n))
    plt.scatter(ESS4x,ESS4y,s=80,marker='o',color='#d62728')
    plt.scatter(aESS4x,aESS4y,s=120,marker="*",color='#d62728')