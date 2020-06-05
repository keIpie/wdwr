#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse, os, sys, time
from testampl import *
from single import *
from bi import *

def main():
    parser = argparse.ArgumentParser(description='AMPL modeling.')
    parser.add_argument("--solver", type=str, help="specify solver", default="cplex")
    parser.add_argument("--fsd", help="check first stochastic domination", action="store_true")
    parser.add_argument("--msingle", help="single criterion model", action="store_true")
    parser.add_argument("--mbi", help="bi-criteria model", action="store_true")
    parser.add_argument("--testampl", help="test AMPL", action="store_true")
    parser.add_argument("--testexp", help="test counting expected value", action="store_true")
    parser.add_argument("--verbose", help="increase output verbosity", action="store_true")
    parser.add_argument('--mod', metavar='FILE', type=str, help='specify model file',  action='store', default="mill.mod")
    parser.add_argument('--dat', metavar='FILE', type=str, help='specify dat file',  action='store', default="mill.dat")
    args = parser.parse_args()
    if args.testampl:
        testmodel(args.solver, args.verbose)
    if args.testexp:
        tstudnet_test(args.verbose)
    if args.msingle:
        single(args.mod, args.solver, args.verbose)
    if args.mbi:
        bi(args.mod, args.solver, args.verbose)
    if args.fsd:
        fsd(86, 3397, args.mod, args.solver, args.verbose)

if __name__ == "__main__":
    main()
