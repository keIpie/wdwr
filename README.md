# wdwr

wspomaganie decyzji w warunkach ryzyka - projekt

-----------------------------------------------------------------------------------------

To test building simple model run:

`./main.py --testampl`

To solve single criterion model run:

`./main.py --msingle --mod "singleLIN.mod" --verbose`

To solve bi-criteria model run:

`./main.py --mbi --mod "singleLIN.mod" --verbose`

Arguments for commands:
- solver - specifies solver to be used (default cplex)
- verbose - to print intermediate results

-----------------------------------------------------------------------------------------

To test counting of expected value run:

`./main.py --testexp --verbose`

-----------------------------------------------------------------------------------------

For help run:

`./main.py -h`

-----------------------------------------------------------------------------------------
