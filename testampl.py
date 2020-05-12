#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function, absolute_import, division
from builtins import map, range, object, zip, sorted
import os, sys, time
import pandas as pd
import numpy as nm

from amplpy import AMPL, DataFrame


def testmodel_init(solver, verbose):
    if(verbose):
        print("-----------------------------------------------------------------")
        print("   testing AMPL 2 variables model (mill)")
        print("-----------------------------------------------------------------")
    ampl = AMPL()
    ampl.eval('var XB;')
    ampl.eval('var XC;')
    ampl.eval('maximize Profit: 25*XB + 30*XC;')
    ampl.eval('subject to Time: (1/200)*XB + (1/140)*XC <= 40;')
    ampl.eval('subject to B_limit: 0 <= XB <= 6000;')
    ampl.eval('subject to C_limit: 0 <= XC <= 4000;')
    # solve
    ampl.option['solver'] = solver
    ampl.solve()
    XB = ampl.getVariable('XB')
    XC = ampl.getVariable('XC')
    if(verbose):
        print('XB, XC:', XB.value(), XC.value())
        print('time:', (1/200)*XB.value() + (1/140)*XC.value())
        print('profit:', 25*XB.value() + 30*XC.value())

def testmodel(solver, verbose):
    if(verbose):
        print("-----------------------------------------------------------------")
        print("   testing AMPL generic model (mill)")
        print("-----------------------------------------------------------------")
    ampl = AMPL()
    # creating ampl model
    ampl.eval('set PROD;') # products that can be made
    ampl.eval('set STAGE;') # stages of production
    ampl.eval('param rate {PROD, STAGE} > 0;') # nr of product per hour
    ampl.eval('param avail {STAGE} >=0 ;') # total number of hours
    ampl.eval('param profit {PROD};') # profit from product
    ampl.eval('param market {PROD};') # max nr of product needed
    ampl.eval('param commit {PROD};') # min nr of product needed
    ampl.eval('var Make {p in PROD} >= commit[p], <= market[p];') # nr of products to be made
    ampl.eval('maximize Total_Profit: sum {p in PROD} profit[p]*Make[p];')
    ampl.eval('subject to Time {s in STAGE}: sum {p in PROD} (1/rate[p,s])*Make[p] <= avail[s];')
    # data for model
    PROD = ['bands','coils','plate']
    STAGE = ['reheat','roll']
    reheat = [200,200,200]
    roll = [200,140,160]
    rate = [reheat, roll]
    avail = [35,40]
    profit = [25,30,29]
    commit = [1000,500,750]
    market = [6000,4000,3500]
    # data frame for products
    dfP = DataFrame('PROD')
    dfP.setColumn('PROD', PROD)
    dfP.addColumn('profit', profit)
    dfP.addColumn('market', market)
    dfP.addColumn('commit', commit)
    ampl.setData(dfP, 'PROD')
    print(dfP)
    # data frame for stages
    dfS = DataFrame('STAGE')
    dfS.setColumn('STAGE', STAGE)
    dfS.addColumn('avail', avail)
    ampl.setData(dfS, 'STAGE')
    print(dfS)
    # data frame with avaliable
    df = DataFrame(('STAGE','PROD'),'rate')
    df.setValues({
        (product, stg): rate[i][j]
        for i, stg in enumerate(STAGE)
        for j, product in enumerate(PROD)
    })
    ampl.setData(df)
    print(df)
    #ampl.param['b'] = b #ustawianie zwyklego parametru
    # solve
    ampl.option['solver'] = solver
    ampl.solve()
    # result
    totalcost = ampl.getObjective('Total_Profit')
    print("Total cost:", totalcost.value())
    if(verbose):
        # ampl.display('X')
        dfX = ampl.var['X'].getValues().toPandas()
        X = dfX['X.val'].astype(float)
        print('X:', X)
        print('time:', nm.dot(invert(a),X))
        print('profit:', nm.dot(c,X))

def testread(solver, modfile, datfile, verbose):
    if(verbose):
        print("-----------------------------------------------------------------")
        print("   testing AMPL reading models (mill)")
        print("-----------------------------------------------------------------")
    ampl = AMPL()
    # read model and data
    ampl.read(os.path.join('models/', modfile))
    ampl.readData(os.path.join('data/', datfile))
    #solve
    ampl.option['solver'] = solver
    ampl.solve()
    totalcost = ampl.getObjective('Total_Profit')
    print("Total cost:", totalcost.value())

def invert(a):
    ainv = []
    for i in xrange(len(a)):
        ainv.append(1/a[i])
    return ainv
