# wdwr

wspomaganie decyzji w warunkach ryzyka - projekt

-----------------------------------------------------------------------------------------

To test building simple model run:

`./main.py --testampl`

To test reading models and data from file run:

`./main.py --testread --mod "steel.mod" --dat "steel.dat"`

To solve single criterion model run:

`./main.py --msingle --mod "single.mod" --verbose`

To solve bi-criteria model run (TBD):

`./main.py --mbi --mod "bi.mod" --verbose`

Arguments for commands:
- soler - specifies solver to be used (default cplex)
- verbose - to print intermediate results

For help run:

`./main.py -h`

-----------------------------------------------------------------------------------------
