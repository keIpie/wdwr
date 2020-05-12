#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function, absolute_import, division
from builtins import map, range, object, zip, sorted
import os, sys, time, math
import pandas as pd
import numpy as nm

import rpy2
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
from single import *

from amplpy import AMPL, DataFrame

def bi(modfile, solver, verbose):
    if(verbose):
        print("---------------------------------------------------------------")
        print("   WDWR project 20204 - model dwukryterialny ")
        print("---------------------------------------------------------------")
    ampl = AMPL()
    ampl.read(os.path.join('models/', modfile))
    R = tstudnet(verbose)
    #cost = singleR(R, modfile, solver, False)
    #print("cost:", cost)
    #R_min = [R[0]-1,R[1]-4,R[2]-2,R[3]-5,R[4]-3,R[5]-6]
    #R_max = [R[0]+1,R[1]+4,R[2]+2,R[3]+5,R[4]+3,R[5]+6]
    #prob = tstudentDensity(R,verbose)
    #prob_min = tstudentDensity(R_min,verbose)
    #prob_max = tstudentDensity(R_max,verbose)
    #if(verbose):
    #    print("prob:", prob)
    #    print("prob_min:", prob_min)
    #    print("prob_max:", prob_max)
    R_t = []
    prob_t = []
    for i0 in [-1,0,1]:
        for i1 in [-4,0,4]:
            for i2 in [-2,0,2]:
                for i3 in [-5,0,5]:
                    for i4 in [-3,0,3]:
                        for i5 in [-6,0,6]:
                            R_tmp = [R[0]+i0,R[1]+i1,R[2]+i2,R[3]+i3,R[4]+i4,R[5]+i5]
                            prob_t.append(tstudentDensity(R_tmp,verbose))
                            R_t.append([[R[0]+i0,R[1]+i1,R[2]+i2],[R[3]+i3,R[4]+i4,R[5]+i5]])
    if(verbose):
        print(R_t)
        print(prob_t)
    # building a model
    PRODUCTS = ['A','B']
    MONTHS = ['M1','M2','M3']
    RESOURCES = ['Z1','Z2']
    SCENARIOS = []
    for i in xrange(3**6):
        SCENARIOS.append(str(i+1))
    print("assert", len(SCENARIOS), len(prob_t))
    # variabels for params
    promised = [1100,1200]
    cost = [[R[0],R[1],R[2]],[R[3],R[4],R[5]]]
    costS = R_t
    prob = prob_t
    maxno = 150 # 150 sztuk mozna przechowywac z miesiaca na miesiac
    percent = 15/100 # 15% kosztow wytwarzania ksoztuje przechowywanie
    lmbd = 10**12
    demand = [[0.2, 0.7],[0.8, 0.3]]
    delivery = [[600,700,550],[1400,900,1200]]
    chronology = [[1,0,0],[1,1,0],[1,1,1]]
    # create params
    ampl.param['maxno'] = maxno #ustawianie zwyklego parametru
    ampl.param['percent'] = percent #ustawianie zwyklego parametru
    ampl.param['lmbd'] = lmbd # ustawianie zwyklego parametru
    # create tables
    # data frame for products
    dp = DataFrame('PRODUCTS')
    dp.setColumn('PRODUCTS', PRODUCTS)
    dp.addColumn('promised', promised)
    ampl.setData(dp, 'PRODUCTS')
    if(verbose):
        print(dp)
    # data frame for months
    dm = DataFrame('MONTHS')
    dm.setColumn('MONTHS', MONTHS)
    ampl.setData(dm, 'MONTHS')
    if(verbose):
        print(dm)
    # data frame for resources
    dr = DataFrame('RESOURCES')
    dr.setColumn('RESOURCES', RESOURCES)
    ampl.setData(dr, 'RESOURCES')
    if(verbose):
        print(dr)
    # data frame for resources
    ds = DataFrame('SCENARIOS')
    ds.setColumn('SCENARIOS', SCENARIOS)
    ds.addColumn('prob', prob)
    ampl.setData(ds, 'SCENARIOS')
    if(verbose):
        print(ds)
    # data frame with demands
    ddd = DataFrame(('PRODUCTS','RESOURCES'),'demand')
    ddd.setValues({
        (product, res): demand[i][j]
        for i, res in enumerate(RESOURCES)
        for j, product in enumerate(PRODUCTS)
    })
    ampl.setData(ddd)
    if(verbose):
        print(ddd)
    # data frame with deliveries
    ddy = DataFrame(('MONTHS','RESOURCES'),'delivery')
    ddy.setValues({
        (month, res): delivery[i][j]
        for i, res in enumerate(RESOURCES)
        for j, month in enumerate(MONTHS)
    })
    ampl.setData(ddy)
    if(verbose):
        print(ddy)
    # data frame with costs
    dc = DataFrame(('PRODUCTS','MONTHS'),'cost')
    dc.setValues({
        (product, month): cost[i][j]
        for i, product in enumerate(PRODUCTS)
        for j, month in enumerate(MONTHS)
    })
    ampl.setData(dc)
    if(verbose):
        print(dc)
    # data frame with costs for scenarios
    dcS = DataFrame(('PRODUCTS','MONTHS','SCENARIOS'),'costS')
    dcS.setValues({
        (product, month, scene): costS[k][i][j]
        for i, product in enumerate(PRODUCTS)
        for j, month in enumerate(MONTHS)
        for k, scene in enumerate(SCENARIOS)
    })
    ampl.setData(dcS)
    if(verbose):
        print(dcS)
    # data frame with chronology
    dch = DataFrame(('MONTHS','MONTHS2'),'chronology')
    dch.setValues({
        (mon, month): chronology[i][j]
        for i, mon in enumerate(MONTHS)
        for j, month in enumerate(MONTHS)
    })
    ampl.setData(dch)
    if(verbose):
        print(dch)
    # solve
    ampl.option['solver'] = solver
    ampl.solve()
    totalcost = ampl.getObjective('Total_Cost')
    if verbose:
        print("Total cost:", totalcost.value())
    if(verbose):
        ampl.display('Make')
        ampl.display('Made')
        ampl.display('cond')
    dfM = ampl.var['Make'].getValues().toPandas()
    Make = dfM['Make.val'].astype(float)
    dfMd = ampl.var['Made'].getValues().toPandas()
    Made = dfMd['Made.val'].astype(float)
    dfC = ampl.var['cond'].getValues().toPandas()
    cond = dfC['cond.val'].astype(float)
    pc = Make[0]*cost[0][0] + Make[1]*cost[0][1] + Make[2]*cost[0][2] + Make[3]*cost[1][0] + Make[4]*cost[1][1] + Make[5]*cost[1][2]
    sc = cond[0]*(Made[0]-maxno)*percent*cost[0][0] + cond[1]*(Made[1]-maxno)*percent*cost[0][1] + cond[2]*(Made[2]-maxno)*percent*cost[0][2] + cond[3]*(Made[3]-maxno)*percent*cost[1][0] + cond[4]*(Made[4]-maxno)*percent*cost[1][1] + cond[5]*(Made[5]-maxno)*percent*cost[1][2]
    print('producing cost = ', pc)
    print('storage cost = ', sc)
    print('total cost = ', pc + sc)
    risk = 0
    for s in xrange(len(SCENARIOS)):
        for p in xrange(len(PRODUCTS)*len(MONTHS)):
                n = 0
                if (p>2):
                    n = 1
                risk = risk + abs(cost[n][p%3]-costS[s][n][p%3])*(Make[p] + cond[p]*(Made[p]-maxno)*percent)*prob[s]
    print('risk = ', risk*lmbd)
    print('minimized = ', (pc+sc) + risk*lmbd)

def tstudentDensity(R,verbose):
    mvtnorm = importr('mvtnorm')
    r_dmvt = robjects.r['dmvt']
    r_c = robjects.r['c']
    r_matrix = robjects.r['matrix']
    x = r_c(R[0],R[1],R[2],R[3],R[4],R[5])
    mi = r_c(55,40,50,35,45,30)
    mat = r_c(1,1,0,2,-1,-1,1,16,-6,-6,-2,12,0,-6,4,2,-2,-5,2,-6,2,25,0,-17,-1,-2,-2,0,9,-5,-1,12,-5,-17,-5,36)
    sigma = r_matrix(mat,6)
    degrees = 5
    p = r_dmvt(x,mi,sigma,degrees,log=False)[0]
    return p

def tstudnet(verbose):
    mi = [55,40,50,35,45,30]
    sigma = [[1,1,0,2,-1,-1],[1,16,-6,-6,-2,12],[0,-6,4,2,-2,-5],
            [2,-6,2,25,0,-17],[-1,-2,-2,0,9,-5],[-1,12,-5,-17,-5,36]]
    alpha = 20
    beta = 60
    degrees = 5
    return ER(mi,sigma,alpha,beta,degrees)

def tstudnet_test(verbose):
    print("test rozkałdu t-studenta z 4 stopniami swobody")
    mi = [45,35,40]
    sigma = [[1,-2,-1],[-2,36,-8],[-1,-8,9]]
    alpha = 20
    beta = 50
    degrees = 4
    return ER(mi,sigma,alpha,beta,degrees)

def ER(mi,sigma,alpha,beta,degrees):
    vect = []
    for i in xrange(len(mi)):
        E = ERi(mi[i],sigma[i][i],alpha,beta,degrees)
        vect.append(E)
        #print("ER"+str(i+1)+":", E)
    return vect

def ERi(mi,sigma,alpha,beta,v):
    sigma = math.sqrt(abs(sigma))
    a = (alpha-mi)/sigma
    b = (beta-mi)/sigma
    #print("a,b", (a,b))
    r_gamma = robjects.r['gamma']
    g = r_gamma(v)[0]
    numerator = g*((v+a**2)**(-(v-1)/2)-(v+b**2)**(-(v-1)/2))*v**(v/2)
    r_pt = robjects.r['pt']
    fb = r_pt(b,v)[0]
    fa = r_pt(a,v)[0]
    f = fb - fa
    g = r_gamma(v/2)[0]*r_gamma(1/2)[0]
    denominator = 1*f*g
    frac = numerator/denominator
    return mi + sigma*frac
