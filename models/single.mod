
set PRODUCTS; # A, B
set MONTHS; # M1, M2, M3
set RESOURCES; # Z1, Z2

param promised {PRODUCTS};
param maxno;
param percent;
param cost {PRODUCTS, MONTHS}; #cost per unit
param demand {PRODUCTS, RESOURCES}; # number of resources needed for product
param delivery {MONTHS, RESOURCES}; # number of resources that can be delivered
param chronology {MONTHS, MONTHS}; # number of resources that can be delivered

var cond {PRODUCTS, MONTHS} binary; # conditional binary variables (if more than 150 entities)
var Made {PRODUCTS, MONTHS} >= 0 ; # number of products already made month
var Make {PRODUCTS, MONTHS} >= 0 ; # number of products made per month

minimize Total_Cost: sum {p in PRODUCTS, m in MONTHS} (cost[p,m]*Make[p,m] + cond[p,m]*(Made[p,m]-maxno)*percent*cost[p,m]);

subject to Promised {p in PRODUCTS}: sum {m in MONTHS} Make[p,m] >= promised[p];
subject to Needs {r in RESOURCES, m in MONTHS}: sum {p in PRODUCTS} Make[p,m]*demand[p,r] <= delivery[m,r];
subject to Storage {p in PRODUCTS, m in MONTHS}: Made[p,m] <= maxno + cond[p,m]*(promised[p]-maxno);
subject to Storage2 {p in PRODUCTS, m in MONTHS}: Made[p,m] >= maxno*cond[p,m];
subject to Summing {p in PRODUCTS, m in MONTHS}: Made[p,m] = sum {n in MONTHS} Make[p,n]*chronology[m,n];
