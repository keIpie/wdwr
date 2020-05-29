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

from amplpy import AMPL, DataFrame

def single(modfile, solver, verbose):
    if(verbose):
        print("---------------------------------------------------------------")
        print("   WDWR project 20204 - model jednokryterialny ")
        print("---------------------------------------------------------------")
    ampl = AMPL()
    ampl.read(os.path.join('models/', modfile))
    #utils = importr('utils')
    #data = robjects.r('read.table(file = "http://personality-project.org/r/datasets/R.appendix3.data", header = T)')
    # create sets
    PRODUCTS = ['A','B']
    MONTHS = ['M1','M2','M3']
    RESOURCES = ['Z1','Z2']
    # variabels for params
    promised = [1100,1200]
    R = tstudnet(verbose)
    cost = [[R[0],R[1],R[2]],[R[3],R[4],R[5]]]
    maxno = 150 # 150 sztuk mozna przechowywac z miesiaca na miesiac
    percent = 15/100 # 15% kosztow wytwarzania ksoztuje przechowywanie
    demand = [[0.2, 0.7],[0.8, 0.3]]
    delivery = [[600,700,550],[1400,900,1200]]
    chronology = [[1,0,0],[1,1,0],[1,1,1]]
    # create params
    ampl.param['maxno'] = maxno #ustawianie zwyklego parametru
    ampl.param['percent'] = percent #ustawianie zwyklego parametru
    # create tables
    # data frame for products
    dp = DataFrame('PRODUCTS')
    dp.setColumn('PRODUCTS', PRODUCTS)
    dp.addColumn('promised', promised)
    ampl.setData(dp, 'PRODUCTS')
    print(dp)
    # data frame for months
    dm = DataFrame('MONTHS')
    dm.setColumn('MONTHS', MONTHS)
    ampl.setData(dm, 'MONTHS')
    print(dm)
    # data frame for resources
    dr = DataFrame('RESOURCES')
    dr.setColumn('RESOURCES', RESOURCES)
    ampl.setData(dr, 'RESOURCES')
    print(dr)
    # data frame with demands
    ddd = DataFrame(('PRODUCTS','RESOURCES'),'demand')
    ddd.setValues({
        (product, res): demand[i][j]
        for i, res in enumerate(RESOURCES)
        for j, product in enumerate(PRODUCTS)
    })
    ampl.setData(ddd)
    print(ddd)
    # data frame with deliveries
    ddy = DataFrame(('MONTHS','RESOURCES'),'delivery')
    ddy.setValues({
        (month, res): delivery[i][j]
        for i, res in enumerate(RESOURCES)
        for j, month in enumerate(MONTHS)
    })
    ampl.setData(ddy)
    print(ddy)
    # data frame with costs
    dc = DataFrame(('PRODUCTS','MONTHS'),'cost')
    dc.setValues({
        (product, month): cost[i][j]
        for i, product in enumerate(PRODUCTS)
        for j, month in enumerate(MONTHS)
    })
    ampl.setData(dc)
    print(dc)
    # data frame with chronology
    dch = DataFrame(('MONTHS','MONTHS2'),'chronology')
    dch.setValues({
        (mon, month): chronology[i][j]
        for i, mon in enumerate(MONTHS)
        for j, month in enumerate(MONTHS)
    })
    ampl.setData(dch)
    print(dch)
    # solve
    ampl.option['solver'] = solver
    ampl.solve()
    # result
    totalcost = ampl.getObjective('Total_Cost')
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
        #print('Make:', Make, type(Make))
        #print('time:', nm.dot(invert(a),X))
        print("---------------------------------------------------------------")
        print("   PRODUCING COSTS")
        print("---------------------------------------------------------------")
        print('producing cost (A,M1) = ', Make[0]*cost[0][0])
        print('producing cost (A,M2) = ', Make[1]*cost[0][1])
        print('producing cost (A,M3) = ', Make[2]*cost[0][2])
        print('producing cost (B,M1) = ', Make[3]*cost[1][0])
        print('producing cost (B,M2) = ', Make[4]*cost[1][1])
        print('producing cost (B,M3) = ', Make[5]*cost[1][2])
        pc = Make[0]*cost[0][0] + Make[1]*cost[0][1] + Make[2]*cost[0][2] + Make[3]*cost[1][0] + Make[4]*cost[1][1] + Make[5]*cost[1][2]
        print('producing cost = ', pc)
        print("---------------------------------------------------------------")
        print("   STORAGE COSTS")
        print("---------------------------------------------------------------")
        print('storage cost (A,M1) = ', cond[0]*(Made[0]-maxno)*percent*cost[0][0])
        print('storage cost (A,M2) = ', cond[1]*(Made[1]-maxno)*percent*cost[0][1])
        print('storage cost (A,M3) = ', cond[2]*(Made[2]-maxno)*percent*cost[0][2])
        print('storage cost (B,M1) = ', cond[3]*(Made[3]-maxno)*percent*cost[1][0])
        print('storage cost (B,M2) = ', cond[4]*(Made[4]-maxno)*percent*cost[1][1])
        print('storage cost (B,M3) = ', cond[5]*(Made[5]-maxno)*percent*cost[1][2])
        sc = cond[0]*(Made[0]-maxno)*percent*cost[0][0] + cond[1]*(Made[1]-maxno)*percent*cost[0][1] + cond[2]*(Made[2]-maxno)*percent*cost[0][2] + cond[3]*(Made[3]-maxno)*percent*cost[1][0] + cond[4]*(Made[4]-maxno)*percent*cost[1][1] + cond[5]*(Made[5]-maxno)*percent*cost[1][2]
        print('storage cost = ', sc)
        print("---------------------------------------------------------------")
        print("   TOTAL COST")
        print("---------------------------------------------------------------")
        print('total cost = ', pc + sc)
        print("---------------------------------------------------------------")
        print("   ARE CONDITIONS SATISFIED?")
        print("---------------------------------------------------------------")
        print('(Z1, M1) : ', (Make[0]*demand[0][0] + Make[3]*demand[0][1]), "<=", delivery[0][0], (Make[0]*demand[0][0] + Make[3]*demand[0][1]) <= delivery[0][0])
        print('(Z2, M1) : ', (Make[0]*demand[1][0] + Make[3]*demand[1][1]), "<=", delivery[1][0], (Make[0]*demand[1][0] + Make[3]*demand[1][1]) <= delivery[1][0])
        print('(Z1, M2) : ', (Make[1]*demand[0][0] + Make[4]*demand[0][1]), "<=", delivery[0][1], (Make[1]*demand[0][0] + Make[4]*demand[0][1]) <= delivery[0][1])
        print('(Z2, M2) : ', (Make[1]*demand[1][0] + Make[4]*demand[1][1]), "<=", delivery[1][1], (Make[1]*demand[1][0] + Make[4]*demand[1][1]) <= delivery[1][1])
        print('(Z1, M3) : ', (Make[2]*demand[0][0] + Make[5]*demand[0][1]), "<=", delivery[0][2], (Make[2]*demand[0][0] + Make[5]*demand[0][1]) <= delivery[0][2])
        print('(Z2, M3) ; ', (Make[2]*demand[1][0] + Make[5]*demand[1][1]), "<=", delivery[1][2], (Make[2]*demand[1][0] + Make[5]*demand[1][1]) <= delivery[1][2])

def singleR(R, modfile, solver, verbose):
    if(verbose):
        print("---------------------------------------------------------------")
        print("   WDWR project 20204 - model jednokryterialny ")
        print("---------------------------------------------------------------")
    assert len(R)==6, "long length of R"
    ampl = AMPL()
    ampl.read(os.path.join('models/', modfile))
    #utils = importr('utils')
    #data = robjects.r('read.table(file = "http://personality-project.org/r/datasets/R.appendix3.data", header = T)')
    # create sets
    PRODUCTS = ['A','B']
    MONTHS = ['M1','M2','M3']
    RESOURCES = ['Z1','Z2']
    # variabels for params
    promised = [1100,1200]
    cost = [[R[0],R[1],R[2]],[R[3],R[4],R[5]]]
    maxno = 150 # 150 sztuk mozna przechowywac z miesiaca na miesiac
    percent = 15/100 # 15% kosztow wytwarzania ksoztuje przechowywanie
    demand = [[0.2, 0.7],[0.8, 0.3]]
    delivery = [[600,700,550],[1400,900,1200]]
    chronology = [[1,0,0],[1,1,0],[1,1,1]]
    # create params
    ampl.param['maxno'] = maxno #ustawianie zwyklego parametru
    ampl.param['percent'] = percent #ustawianie zwyklego parametru
    # create tables
    # data frame for products
    dp = DataFrame('PRODUCTS')
    dp.setColumn('PRODUCTS', PRODUCTS)
    dp.addColumn('promised', promised)
    ampl.setData(dp, 'PRODUCTS')
    if verbose:
        print(dp)
    # data frame for months
    dm = DataFrame('MONTHS')
    dm.setColumn('MONTHS', MONTHS)
    ampl.setData(dm, 'MONTHS')
    if verbose:
        print(dm)
    # data frame for resources
    dr = DataFrame('RESOURCES')
    dr.setColumn('RESOURCES', RESOURCES)
    ampl.setData(dr, 'RESOURCES')
    if verbose:
        print(dr)
    # data frame with demands
    ddd = DataFrame(('PRODUCTS','RESOURCES'),'demand')
    ddd.setValues({
        (product, res): demand[i][j]
        for i, res in enumerate(RESOURCES)
        for j, product in enumerate(PRODUCTS)
    })
    ampl.setData(ddd)
    if verbose:
        print(ddd)
    # data frame with deliveries
    ddy = DataFrame(('MONTHS','RESOURCES'),'delivery')
    ddy.setValues({
        (month, res): delivery[i][j]
        for i, res in enumerate(RESOURCES)
        for j, month in enumerate(MONTHS)
    })
    ampl.setData(ddy)
    if verbose:
        print(ddy)
    # data frame with costs
    dc = DataFrame(('PRODUCTS','MONTHS'),'cost')
    dc.setValues({
        (product, month): cost[i][j]
        for i, product in enumerate(PRODUCTS)
        for j, month in enumerate(MONTHS)
    })
    ampl.setData(dc)
    if verbose:
        print(dc)
    # data frame with chronology
    dch = DataFrame(('MONTHS','MONTHS2'),'chronology')
    dch.setValues({
        (mon, month): chronology[i][j]
        for i, mon in enumerate(MONTHS)
        for j, month in enumerate(MONTHS)
    })
    ampl.setData(dch)
    if verbose:
        print(dch)
    # solve
    ampl.option['solver'] = solver
    ampl.solve()
    # result
    totalcost = ampl.getObjective('Total_Cost')
    if verbose:
        print("Total cost:", totalcost.value())
    ampl.display('Make')
    ampl.display('Made')
    ampl.display('cond')
    dfM = ampl.var['Make'].getValues().toPandas()
    Make = dfM['Make.val'].astype(float)
    dfMd = ampl.var['Made'].getValues().toPandas()
    Made = dfMd['Made.val'].astype(float)
    dfC = ampl.var['cond'].getValues().toPandas()
    cond = dfC['cond.val'].astype(float)
    if(verbose):
        #print('Make:', Make, type(Make))
        #print('time:', nm.dot(invert(a),X))
        print("---------------------------------------------------------------")
        print("   PRODUCING COSTS")
        print("---------------------------------------------------------------")
        print('producing cost (A,M1) = ', Make[0]*cost[0][0])
        print('producing cost (A,M2) = ', Make[1]*cost[0][1])
        print('producing cost (A,M3) = ', Make[2]*cost[0][2])
        print('producing cost (B,M1) = ', Make[3]*cost[1][0])
        print('producing cost (B,M2) = ', Make[4]*cost[1][1])
        print('producing cost (B,M3) = ', Make[5]*cost[1][2])
        print('producing cost = ', Make[0]*cost[0][0] + Make[1]*cost[0][1] + Make[2]*cost[0][2] + Make[3]*cost[1][0] + Make[4]*cost[1][1] + Make[5]*cost[1][2])
        print("---------------------------------------------------------------")
        print("   STORAGE COSTS")
        print("---------------------------------------------------------------")
        print('storage cost (A,M1) = ', cond[0]*(Made[0]-maxno)*percent*cost[0][0])
        print('storage cost (A,M2) = ', cond[1]*(Made[1]-maxno)*percent*cost[0][1])
        print('storage cost (A,M3) = ', cond[2]*(Made[2]-maxno)*percent*cost[0][2])
        print('storage cost (B,M1) = ', cond[3]*(Made[3]-maxno)*percent*cost[1][0])
        print('storage cost (B,M2) = ', cond[4]*(Made[4]-maxno)*percent*cost[1][1])
        print('storage cost (B,M3) = ', cond[5]*(Made[5]-maxno)*percent*cost[1][2])
        print('storage cost = ', cond[0]*(Made[0]-maxno)*percent*cost[0][0] + cond[1]*(Made[1]-maxno)*percent*cost[0][1] + cond[2]*(Made[2]-maxno)*percent*cost[0][2] +
                                cond[3]*(Made[3]-maxno)*percent*cost[1][0] + cond[4]*(Made[4]-maxno)*percent*cost[1][1] + cond[5]*(Made[5]-maxno)*percent*cost[1][2])
        print("---------------------------------------------------------------")
        print("   ARE CONDITIONS SATISFIED?")
        print("---------------------------------------------------------------")
        print('(Z1, M1) : ', (Make[0]*demand[0][0] + Make[3]*demand[0][1]), "<=", delivery[0][0], (Make[0]*demand[0][0] + Make[3]*demand[0][1]) <= delivery[0][0])
        print('(Z2, M1) : ', (Make[0]*demand[1][0] + Make[3]*demand[1][1]), "<=", delivery[1][0], (Make[0]*demand[1][0] + Make[3]*demand[1][1]) <= delivery[1][0])
        print('(Z1, M2) : ', (Make[1]*demand[0][0] + Make[4]*demand[0][1]), "<=", delivery[0][1], (Make[1]*demand[0][0] + Make[4]*demand[0][1]) <= delivery[0][1])
        print('(Z2, M2) : ', (Make[1]*demand[1][0] + Make[4]*demand[1][1]), "<=", delivery[1][1], (Make[1]*demand[1][0] + Make[4]*demand[1][1]) <= delivery[1][1])
        print('(Z1, M3) : ', (Make[2]*demand[0][0] + Make[5]*demand[0][1]), "<=", delivery[0][2], (Make[2]*demand[0][0] + Make[5]*demand[0][1]) <= delivery[0][2])
        print('(Z2, M3) ; ', (Make[2]*demand[1][0] + Make[5]*demand[1][1]), "<=", delivery[1][2], (Make[2]*demand[1][0] + Make[5]*demand[1][1]) <= delivery[1][2])
    return Make, totalcost.value()

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
        print("ER"+str(i+1)+":", E)
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
