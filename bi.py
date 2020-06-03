#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function, absolute_import, division
from builtins import map, range, object, zip, sorted
import os, sys, time, math
import pandas as pd
import numpy as nm
import matplotlib.pyplot as plt

import rpy2
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
from single import *
from amplpy import AMPL, DataFrame

def bi(modfile, solver, verbose):
    """
    Evaluates bi-criterion model for the project data
    Parameters:
        modfile (str): path to file defining model
        solver (str):  name of solver to be used
        verbose (bool): if True prints comments
    Returns:
        None
    """
    if(verbose):
        print("---------------------------------------------------------------")
        print("   WDWR project 20204 - model dwukryterialny ")
        print("---------------------------------------------------------------")
    risk(modfile, solver, verbose)

def risk(modfile, solver, verbose):
    """
    Counts risks and costs for bi-criterion model for the project data
    Parameters:
        modfile (str): path to file defining model
        solver (str):  name of solver to be used
        verbose (bool): if True prints comments
    Returns:
        None
    """
    f = open("wyniki.txt","w+")
    ampl = AMPL()
    ampl.read(os.path.join('models/', modfile))
    R = tstudnet(verbose)
    make, total = singleR(R, modfile, solver, verbose)
    R_t = []
    resize = 5 # how many scenarios in single dimension -> (60-20)/resize
    intervals_prob = tstudentProbability(resize,verbose)
    prob_t = []
    for i0 in range(0,40/resize,1):
        for i1 in range(0,40/resize,1):
            for i2 in range(0,40/resize,1):
                for i3 in range(0,40/resize,1):
                    for i4 in range(0,40/resize,1):
                        for i5 in range(0,40/resize,1):
                            #prob_t.append(tstudentDensity(R_tmp,verbose))
                            prob_t.append(intervals_prob[i0,i1,i2,i3,i4,i5])
                            R_tmp = [20+resize*i0,20+resize*i1,20+resize*i2,20+resize*i3,20+resize*i4,20+resize*i5]
                            R_t.append(R_tmp)
    #print(prob_t)
    sum_prob = 0
    for i in xrange(len(prob_t)):
        sum_prob = sum_prob + prob_t[i]
    print(sum_prob)
    for i in xrange(len(prob_t)):
        prob_t[i] = prob_t[i]/sum_prob
    # prob_t_n0 = []
    # for i in range(0, len(prob_t), 1):
    #     if(prob_t[i] != 0):
    #         prob_t_n0.append(prob_t[i])
    # print(prob_t_n0)
    # #plt.plot([i for i in xrange(len(prob_t))], prob_t, 'ro')
    # #plt.show()
    plt_cost = []
    plt_risk = []
    for i in xrange(len(R_t)):
        if (prob_t[i] == 0):
        #if (prob_t[i] < 1/1000):
            print("przerywam")
        else:
            make_t, total_t = singleR(R_t[i], modfile, solver, verbose)
            #print(R_t[i], make_t, total_t)
            risk_t = 0
            for j in xrange(len(R_t)):
                if (prob_t[j]!=0):
                    total_j = countCost(make_t, R_t[j])
                    total_mi = countCost(make_t, R)
                    risk_t = risk_t + abs(total_mi - total_j)*prob_t[j]
            #print(prob_t[j], total_t, risk_t)
            plt_cost.append(total_t)
            plt_risk.append(risk_t)
    print(plt_cost)
    print(plt_risk)
    print(sum_prob)
    print("koszt:", countCost(make, R), total)
    f.write("koszt:")
    f.write(str(countCost(make, R)))
    f.write(str(total))
    plt.xlabel('koszt wytworzenia produktow dla decyzji optymalnej dla scenariusza')
    plt.ylabel('ryzyko dla decyzji optymalnej dla scenariusza')
    plt.plot(plt_cost, plt_risk, 'ro')
    plt.show()
    ef1, ef2, ef3 = find_minimal(plt_cost, plt_risk)
    fsd1 = fsdSolution(ef1, R_t, prob_t, modfile, solver, verbose)
    fsd2 = fsdSolution(ef2, R_t, prob_t, modfile, solver, verbose)
    fsd3 = fsdSolution(ef3, R_t, prob_t, modfile, solver, verbose)
    f.write("fsd1")
    f.write(str(fsd1))
    f.write("fsd2")
    f.write(str(fsd2))
    f.write("fsd3")
    f.write(str(fsd3))
    print("Czy 1 dominuje 2?", checkFSD(fsd1, fsd2))
    print("Czy 2 dominuje 1?", checkFSD(fsd2, fsd1))
    print("Czy 1 dominuje 3?", checkFSD(fsd1, fsd3))
    print("Czy 3 dominuje 1?", checkFSD(fsd3, fsd1))
    print("Czy 2 dominuje 3?", checkFSD(fsd2, fsd3))
    print("Czy 3 dominuje 2?", checkFSD(fsd3, fsd2))
    f.write("Czy 1 dominuje 2?")
    f.write(str(checkFSD(fsd1, fsd2)))
    f.write("Czy 2 dominuje 1?")
    f.write(str(checkFSD(fsd2, fsd1)))
    f.write("Czy 1 dominuje 3?")
    f.write(str(checkFSD(fsd1, fsd3)))
    f.write("Czy 3 dominuje 1?")
    f.write(str(checkFSD(fsd3, fsd1)))
    f.write("Czy 2 dominuje 3?")
    f.write(str(checkFSD(fsd2, fsd3)))
    f.write("Czy 3 dominuje 2?")
    f.write(str(checkFSD(fsd3, fsd2)))
    f.close()

def find_effective(tab1, tab2):
    """
    Finds effective solutions
    Parameters:
        tab1 ([float]): list of costs of solutions
        tab2 ([float]): list of risks of solutions
    Returns:
        effective_cost ([float]): list of costs of effective solutions
        effective_risk ([float]): list of risks of effective solutions
    """
    effective_cost = []
    effective_risk = []
    effective_id = []
    for i in range(0, len(tab1), 1):
        for j in range(0, len(tab1),1):
            if(tab1[i]>tab1[j] and tab2[i]>=tab2[j]):
                break
            if(tab1[i]>=tab1[j] and tab2[i]>tab2[j]):
                break
            if (j == len(tab1)-1):
                effective_cost.append(tab1[i])
                effective_risk.append(tab2[i])
                effective_id.append(i)
                print(tab1[i], tab2[i])
    plt.xlabel('koszt wytworzenia produktow dla decyzji optymalnej dla scenariusza')
    plt.ylabel('ryzyko dla decyzji optymalnej dla scenariusza')
    plt.plot(tab1, tab2, 'ro')
    plt.plot(effective_cost, effective_risk, 'bo')
    plt.show()
    print(effective_cost)
    print(effective_risk)
    print(effective_id)
    return (effective_cost, effective_risk, effective_id)

def find_minimal(tab1, tab2):
    """
    Finds minimal solutions
    Parameters:
        tab1 ([float]): list of costs of solutions
        tab2 ([float]): list of risks of solutions
    Returns:
        None
    """
    fm = open("minimal.txt","w+")
    eff_cost, eff_risk, eff_id = find_effective(tab1, tab2)
    minimal_cost = eff_cost[0]
    minimal_risk = eff_risk[0]
    minimal_cost_id = eff_id[0]
    minimal_risk_id = eff_id[0]
    for i in range(1, len(eff_cost), 1):
        print(minimal_cost, minimal_risk)
        if (eff_cost[i]<minimal_cost):
            minimal_cost = eff_cost[i]
            minimal_cost_id = eff_id[i]
        if (eff_risk[i]<minimal_risk):
            minimal_risk = eff_risk[i]
            minimal_risk_id = eff_id[i]
    plt.xlabel('koszt wytworzenia produktow dla decyzji optymalnej dla scenariusza')
    plt.ylabel('ryzyko dla decyzji optymalnej dla scenariusza')
    plt.plot(tab1, tab2, 'ro')
    plt.plot(eff_cost, eff_risk, 'bo')
    plt.plot(tab1[minimal_cost_id], tab2[minimal_cost_id], 'ko')
    plt.plot(tab1[minimal_risk_id], tab2[minimal_risk_id], 'ko')
    plt.show()
    print("minimal_cost:", minimal_cost_id, tab1[minimal_cost_id], tab2[minimal_cost_id])
    print("minimal_risk:", minimal_risk_id, tab1[minimal_risk_id], tab2[minimal_risk_id])
    fm.write("minimal_cost:")
    fm.write(str(minimal_cost_id))
    fm.write(str(tab1[minimal_cost_id]))
    fm.write(str(tab2[minimal_cost_id]))
    fm.write("minimal_risk:")
    fm.write(str(minimal_risk_id))
    fm.write(str(tab1[minimal_risk_id]))
    fm.write(str(tab2[minimal_risk_id]))
    eff_id_3 = eff_id[0]
    if (minimal_cost_id == eff_id_3):
        eff_id_3 = eff_id[1]
        if (minimal_risk_id == eff_id_3):
            eff_id_3 = eff_id[2]
    elif (minimal_risk_id == eff_id_3):
        eff_id_3 = eff_id[1]
        if (minimal_cost_id == eff_id_3):
            eff_id_3 = eff_id[2]
    print("effective points id:", minimal_cost_id, minimal_risk_id, eff_id_3)
    fm.write("eff_id_3:")
    fm.write(str(eff_id_3))
    fm.write(str(tab1[eff_id_3]))
    fm.write(str(tab2[eff_id_3]))
    fm.close()
    return minimal_cost_id, minimal_risk_id, eff_id_3

def fsd(sol1id, sol2id, modfile, solver, verbose):
    """
    Checks if exists first stochastic domination (FSD) between solutions
    """
    ampl = AMPL()
    ampl.read(os.path.join('models/', modfile))
    R = tstudnet(verbose)
    make, total = singleR(R, modfile, solver, verbose)
    R_t = []
    resize = 5 # how many scenarios in single dimension -> (60-20)/resize
    intervals_prob = tstudentProbability(resize,verbose)
    prob_t = []
    for i0 in range(0,40/resize,1):
        for i1 in range(0,40/resize,1):
            for i2 in range(0,40/resize,1):
                for i3 in range(0,40/resize,1):
                    for i4 in range(0,40/resize,1):
                        for i5 in range(0,40/resize,1):
                            #prob_t.append(tstudentDensity(R_tmp,verbose))
                            p = intervals_prob[i0,i1,i2,i3,i4,i5]
                            if (p != 0):
                                prob_t.append(p)
                                R_tmp = [20+resize*i0,20+resize*i1,20+resize*i2,20+resize*i3,20+resize*i4,20+resize*i5]
                                R_t.append(R_tmp)
    sum_prob = 0
    for i in xrange(len(prob_t)):
        sum_prob = sum_prob + prob_t[i]
    print(sum_prob)
    for i in xrange(len(prob_t)):
        prob_t[i] = prob_t[i]/sum_prob
    sum_prob = 0
    for i in xrange(len(prob_t)):
        sum_prob = sum_prob + prob_t[i]
    print(sum_prob)
    #counts fsd for 1st solution
    fsd1 = fsdSolution(sol1id, R_t, prob_t, modfile, solver, verbose)
    fsd2 = fsdSolution(sol2id, R_t, prob_t, modfile, solver, verbose)
    print("Dystrybuanta 1")
    #print(fsd1)
    #print("Dystrybuanta 2")
    #print(fsd2)
    print("Czy 1 domiuje 2?", checkFSD(fsd1, fsd2))

def fsdSolution(sol1id, R_t, prob_t, modfile, solver, verbose):
    make1, total1 = singleR(R_t[sol1id], modfile, solver, verbose)
    #print("1st solution:", total1, prob_t[sol1id])
    dict1 = {}
    for i in range(0,len(R_t),1):
        dict1[countCost(make1, R_t[i])] = 0
    for i in range(0,len(R_t),1):
        dict1[countCost(make1, R_t[i])] = dict1[countCost(make1, R_t[i])] + prob_t[i]
    #print(dict1)
    dict1s = sorted(dict1)
    fsd1 = []
    sum1 = 0
    for i in range(len(dict1s)-1,-1,-1):
        sum1 = sum1 + dict1[dict1s[i]]
        fsd1.append(sum1)
    #print("cumulative distribution of 1st solution", fsd1)
    return fsd1

def checkFSD(tab1, tab2):
    dominate = False
    for i in range(0,len(tab1),1):
        if(tab1[i]<tab2[i]):
            break
        if (i == len(tab1)-1):
            dominate = True
    return dominate

def countCost(Make, R):
    """
    Counts cost of given decision for given scenario
    Parameters:
        Make ([int]): list of numbers of products to be made during each month (decision)
        R ([int]): list of costs of production per month, specific for given scenario
    Returns:
        total cost for given scenrio (float)
    """
    percent = 15/100
    maxno = 150
    cond = []
    Made = []
    Made.append(Make[0])
    Made.append(Make[0] + Make[1])
    Made.append(Make[0] + Make[1] + Make[2])
    Made.append(Make[3])
    Made.append(Make[3] + Make[4])
    Made.append(Make[3] + Make[4] + Make[5])
    for i in xrange(len(Made)):
        if (Made[i]>150):
            cond.append(1)
        else:
            cond.append(0)
    pc = Make[0]*R[0] + Make[1]*R[1] + Make[2]*R[2] + Make[3]*R[3] + Make[4]*R[4] + Make[5]*R[5]
    sc = cond[0]*(Made[0]-maxno)*percent*R[0] + cond[1]*(Made[1]-maxno)*percent*R[1] + cond[2]*(Made[2]-maxno)*percent*R[2] + cond[3]*(Made[3]-maxno)*percent*R[3] + cond[4]*(Made[4]-maxno)*percent*R[4] + cond[5]*(Made[5]-maxno)*percent*R[5]
    return pc+sc

def tstudentProbability(resize,verbose):
    """
    Parameters:
        resize (int): defines on how many intervals to split distribution
        verbose (str): if True print comments
    Returns:
        intervals: six dimension array of probabilities (one for every scenario)
    """
    num = 2**18
    mvtnorm = importr('LaplacesDemon')
    r_rmvt = robjects.r['rmvt']
    r_c = robjects.r['c']
    r_matrix = robjects.r['matrix']
    r_sum = robjects.r['sum']
    r_max = robjects.r['max']
    r_array = robjects.r['array']
    degrees = 5
    scl = (degrees-2)/degrees
    mi = r_c(55,40,50,35,45,30)
    mat = r_c(1*scl,1*scl,0,2*scl,-1*scl,-1*scl,1*scl,16*scl,-6*scl,-6*scl,-2*scl,12*scl,0,-6*scl,4*scl,2*scl,-2*scl,-5*scl,2*scl,-6*scl,2*scl,25*scl,0,
            -17*scl,-1*scl,-2*scl,-2*scl,0,9*scl,-5*scl,-1*scl,12*scl,-5*scl,-17*scl,-5*scl,36*scl)
    print(type(mat))
    sigma = r_matrix(mat,6)
    # generating random points
    p = r_rmvt(num,mi,sigma,degrees)
    # dividing different dimensions
    p1v = p[0:num]
    p2v = p[num:2*num]
    p3v = p[2*num:3*num]
    p4v = p[3*num:4*num]
    p5v = p[4*num:5*num]
    p6v = p[5*num:6*num]
    # testing if expected values are correct
    p1exp = r_sum(p1v)[0]/num
    p2exp = r_sum(p2v)[0]/num
    p3exp = r_sum(p3v)[0]/num
    p4exp = r_sum(p4v)[0]/num
    p5exp = r_sum(p5v)[0]/num
    p6exp = r_sum(p6v)[0]/num
    if verbose:
        print("wartości oczekiwane:", mi)
        print("wyniki eksperymentalne:", p1exp, p2exp, p3exp, p4exp, p5exp, p6exp)
    # counting points in intervals
    lw_lim = 20
    up_lim = 60
    resize = 5
    dim = int((up_lim - lw_lim)/resize)
    intervals = nm.ndarray([dim,dim,dim,dim,dim,dim])
    for i in range(0,num,1):
        v1, v2, v3, v4, v5, v6 = -2, -2, -2, -2, -2, -2
        if (p1v[i] > lw_lim and p1v[i] < up_lim):
            v1 = math.floor((p1v[i]-lw_lim)/resize)
        if (p2v[i] > lw_lim and p2v[i] < up_lim):
            v2 = math.floor((p2v[i]-lw_lim)/resize)
        if (p3v[i] > lw_lim and p3v[i] < up_lim):
            v3 = math.floor((p3v[i]-lw_lim)/resize)
        if (p4v[i] > lw_lim and p4v[i] < up_lim):
            v4 = math.floor((p4v[i]-lw_lim)/resize)
        if (p5v[i] > lw_lim and p5v[i] < up_lim):
            v5 = math.floor((p5v[i]-lw_lim)/resize)
        if (p6v[i] > lw_lim and p6v[i] < up_lim):
            v6 = math.floor((p6v[i]-lw_lim)/resize)
        if (v1>-1 and v2>-1 and v3>-1 and v4>-1 and v5>-1 and v6>-1):
            if verbose:
                print(v1,v2,v3,v4,v5,v6)
                print(intervals[int(v1),int(v2),int(v3),int(v4),int(v5),int(v6)])
            intervals[int(v1),int(v2),int(v3),int(v4),int(v5),int(v6)] += 1
    #print(intervals[3,2,3,0,2,1])
    #intervals2 = intervals[7,4,5,0:8,4,1]
    #plt.plot(intervals2, 'ro')
    #plt.show()
    psum = intervals.sum()
    print("psum", psum)
    return intervals

def tstudentDensity(R,verbose):
    """
    Counts density of given scenario
    Parameters:
        R ([int]): vector of costs in given scenario
        verbose (str): if True print comments
    Returns:
        density of given scenario (float)
    """
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
    """
    Counts vector of expected values for multidimensional t-Student distribution for project data
    Returns:
        vector of expected values ([float])
    """
    mi = [55,40,50,35,45,30]
    sigma = [[1,1,0,2,-1,-1],[1,16,-6,-6,-2,12],[0,-6,4,2,-2,-5],
            [2,-6,2,25,0,-17],[-1,-2,-2,0,9,-5],[-1,12,-5,-17,-5,36]]
    alpha = 20
    beta = 60
    degrees = 5
    return ER(mi,sigma,alpha,beta,degrees)

def ER(mi,sigma,alpha,beta,degrees):
    """
    Counts expected value of variable from multidimensional t-Studnet distribution
    Parameters:
        mi ([int]): vector of average values
        sigma ([int]): covariance matrix
        alpha (int): lower limit of the narrowed range
        beta (int): upper limit of the narrowed range
        degree (int): number of degrees of freedom
    Returns:
        vect ([float]): vector of expected values
    """
    vect = []
    for i in xrange(len(mi)):
        E = ERi(mi[i],sigma[i][i],alpha,beta,degrees)
        vect.append(E)
        #print("ER"+str(i+1)+":", E)
    return vect

def ERi(mi,sigma,alpha,beta,v):
    """
    Counts expected value of variable from multidimensional t-Studnet distribution
    Parameters:
        mi (int): average value in i-th dimension
        sigma (int): standard deviation in i-th dimension
        alpha (int): lower limit of the narrowed range
        beta (int): upper limit of the narrowed range
        v (int): number of degrees of freedom
    Returns:
        expected value (float)
    """
    sigma = math.sqrt(abs(sigma))
    a = (alpha-mi)/sigma
    b = (beta-mi)/sigma
    #print("a,b", (a,b))
    r_gamma = robjects.r['gamma']
    g = r_gamma((v-1)/2)[0]
    numerator = g*((v+a**2)**(-(v-1)/2)-(v+b**2)**(-(v-1)/2))*v**(v/2)
    r_pt = robjects.r['pt']
    fb = r_pt(b,v)[0]
    fa = r_pt(a,v)[0]
    f = fb - fa
    g = r_gamma(v/2)[0]*r_gamma(1/2)[0]
    denominator = 1*f*g
    frac = numerator/denominator
    return mi + sigma*frac

########################### tests of correctness ###############################

def tstudnet_test(verbose):
    """
    Tests counting vector of expected values for multidimensional t-Student distribution for example from lecture
    Returns:
        vector of expected values ([float])
    """
    if verbose:
        print("test rozkałdu t-studenta z 4 stopniami swobody")
    mi = [45,35,40]
    sigma = [[1,-2,-1],[-2,36,-8],[-1,-8,9]]
    alpha = 20
    beta = 50
    degrees = 4
    return ER(mi,sigma,alpha,beta,degrees)

#################### variant with single2.mod ##################################

def risk2(modfile, solver, verbose):
    """
    Counts risks and costs for bi-criterion model for the project data
    Parameters:
        modfile (str): path to file defining model
        solver (str):  name of solver to be used
        verbose (bool): if True prints comments
    Returns:
        None
    """
    ampl = AMPL()
    ampl.read(os.path.join('models/', modfile))
    R = tstudnet(verbose)
    make, total = single2R(R, modfile, solver, verbose)
    R_t = []
    resize = 5 # how many scenarios in single dimension -> (60-20)/resize
    intervals_prob = tstudentProbability(resize,verbose)
    prob_t = []
    for i0 in range(0,40/resize,1):
        for i1 in range(0,40/resize,1):
            for i2 in range(0,40/resize,1):
                for i3 in range(0,40/resize,1):
                    for i4 in range(0,40/resize,1):
                        for i5 in range(0,40/resize,1):
                            #prob_t.append(tstudentDensity(R_tmp,verbose))
                            prob_t.append(intervals_prob[i0,i1,i2,i3,i4,i5])
                            R_tmp = [20+resize*i0,20+resize*i1,20+resize*i2,20+resize*i3,20+resize*i4,20+resize*i5]
                            R_t.append(R_tmp)
    #print(prob_t)
    sum_prob = 0
    for i in xrange(len(prob_t)):
        sum_prob = sum_prob + prob_t[i]
    print(sum_prob)
    for i in xrange(len(prob_t)):
        prob_t[i] = prob_t[i]/sum_prob
    # prob_t_n0 = []
    # for i in range(0, len(prob_t), 1):
    #     if(prob_t[i] != 0):
    #         prob_t_n0.append(prob_t[i])
    # print(prob_t_n0)
    # #plt.plot([i for i in xrange(len(prob_t))], prob_t, 'ro')
    # #plt.show()
    plt_cost = []
    plt_risk = []
    for i in xrange(len(R_t)):
        #if (prob_t[i] == 0):
        if (prob_t[i] < 1/1000):
            print("przerywam")
        else:
            make_t, total_t = single2R(R_t[i], modfile, solver, verbose)
            #print(R_t[i], make_t, total_t)
            risk_t = 0
            for j in xrange(len(R_t)):
                if (prob_t[j]!=0):
                    total_j = countCost2(make_t, R_t[j])
                    total_mi = countCost2(make_t, R)
                    risk_t = risk_t + abs(total_mi - total_j)*prob_t[j]
            #print(prob_t[j], total_t, risk_t)
            plt_cost.append(total_t)
            plt_risk.append(risk_t)
    print(plt_cost)
    print(plt_risk)
    print(sum_prob)
    print("koszt:", countCost2(make, R), total)
    plt.xlabel('koszt wytworzenia produktow dla decyzji optymalnej dla scenariusza')
    plt.ylabel('ryzyko dla decyzji optymalnej dla scenariusza')
    plt.plot(plt_cost, plt_risk, 'ro')
    plt.show()
    ef1, ef2, ef3 = find_minimal(plt_cost, plt_risk)
    fsd1 = fsdSolution2(ef1, R_t, prob_t, modfile, solver, verbose)
    fsd2 = fsdSolution2(ef2, R_t, prob_t, modfile, solver, verbose)
    fsd3 = fsdSolution2(ef3, R_t, prob_t, modfile, solver, verbose)
    print("Czy 1 dominuje 2?", checkFSD(fsd1, fsd2))
    print("Czy 2 dominuje 1?", checkFSD(fsd2, fsd1))
    print("Czy 1 dominuje 3?", checkFSD(fsd1, fsd3))
    print("Czy 3 dominuje 1?", checkFSD(fsd3, fsd1))
    print("Czy 2 dominuje 3?", checkFSD(fsd2, fsd3))
    print("Czy 3 dominuje 2?", checkFSD(fsd3, fsd2))

def fsdSolution2(sol1id, R_t, prob_t, modfile, solver, verbose):
    make1, total1 = single2R(R_t[sol1id], modfile, solver, verbose)
    #print("1st solution:", total1, prob_t[sol1id])
    dict1 = {}
    for i in range(0,len(R_t),1):
        dict1[countCost2(make1, R_t[i])] = 0
    for i in range(0,len(R_t),1):
        dict1[countCost2(make1, R_t[i])] = dict1[countCost2(make1, R_t[i])] + prob_t[i]
    #print(dict1)
    dict1s = sorted(dict1)
    fsd1 = []
    sum1 = 0
    for i in range(len(dict1s)-1,-1,-1):
        sum1 = sum1 + dict1[dict1s[i]]
        fsd1.append(sum1)
    #print("cumulative distribution of 1st solution", fsd1)
    return fsd1

def countCost2(Make, R):
    """
    Counts cost of given decision for given scenario
    Parameters:
        Make ([int]): list of numbers of products to be made during each month (decision)
        R ([int]): list of costs of production per month, specific for given scenario
    Returns:
        total cost for given scenrio (float)
    """
    percent = 15/100
    maxno = 150
    cond = []
    for i in xrange(len(Make)):
        if (Make[i]>150):
            cond.append(1)
        else:
            cond.append(0)
    pc = Make[0]*R[0] + Make[1]*R[1] + Make[2]*R[2] + Make[3]*R[3] + Make[4]*R[4] + Make[5]*R[5]
    sc = cond[0]*(Make[0]-maxno)*percent*R[0] + cond[1]*(Make[1]-maxno)*percent*R[1] + cond[2]*(Make[2]-maxno)*percent*R[2] + cond[3]*(Make[3]-maxno)*percent*R[3] + cond[4]*(Make[4]-maxno)*percent*R[4] + cond[5]*(Make[5]-maxno)*percent*R[5]
    return pc+sc

######################### variant with different idea #########################

def risk_variant2(modfile, solver, verbose):
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
    #pc = Make[0]*cost[0][0] + Make[1]*cost[0][1] + Make[2]*cost[0][2] + Make[3]*cost[1][0] + Make[4]*cost[1][1] + Make[5]*cost[1][2]
    #sc = cond[0]*(Made[0]-maxno)*percent*cost[0][0] + cond[1]*(Made[1]-maxno)*percent*cost[0][1] + cond[2]*(Made[2]-maxno)*percent*cost[0][2] + cond[3]*(Made[3]-maxno)*percent*cost[1][0] + cond[4]*(Made[4]-maxno)*percent*cost[1][1] + cond[5]*(Made[5]-maxno)*percent*cost[1][2]
    #print('producing cost = ', pc)
    #print('storage cost = ', sc)
    #print('total cost = ', pc + sc)
    risk = 0
    for s in xrange(len(SCENARIOS)):
        for p in xrange(len(PRODUCTS)*len(MONTHS)):
                n = 0
                if (p>2):
                    n = 1
                risk = risk + abs(cost[n][p%3]-costS[s][n][p%3])*(Make[p] + cond[p]*(Made[p]-maxno)*percent)*prob[s]
    print('risk = ', risk)
    return risk
    #print('minimized = ', (pc+sc) + risk*lmbd)
