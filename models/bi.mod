
set PRODUCTS; # A, B
set MONTHS; # M1, M2, M3
set RESOURCES; # Z1, Z2
set SCENARIOS; #

param promised {PRODUCTS};
param maxno;
param percent;
param cost {PRODUCTS, MONTHS}; #cost per unit
param costS {PRODUCTS, MONTHS, SCENARIOS}; #cost per unit in scenario
param prob; # probability of a scenario
param demand {PRODUCTS, RESOURCES}; # number of resources needed for product
param delivery {MONTHS, RESOURCES}; # number of resources that can be delivered
param chronology {MONTHS, MONTHS}; # table with months chronology
param lmbd; # importance of the risk
param min_diff; # minimal difference of costs for scenarios (for negative difference)
param max_diff; # maximal difference of costs for scenarios (for positive difference)

var cond {PRODUCTS, MONTHS} binary; # conditional binary variables (if more than 150 entities)
var cond2 {SCENARIOS} binary; # conditional binary variables (connected with absolute value of median absolute deviation)
var Made {PRODUCTS, MONTHS} >= 0 ; # number of products already made month
var Make {PRODUCTS, MONTHS} integer >= 0 ; # number of products made per month
var extra_stored {PRODUCTS, MONTHS} integer >= 0 ; # number of products that have to be extra paid per month
var expected_cost >= 0 ; # expected cost of production
var risk {SCENARIOS} >= 0 ; # risk calculated as median absolute deviation
var risk_sum >= 0 ; # sum of risks over scenarios
var cost_difference {SCENARIOS}; # difference between expected cost and cost of a scenario
var cost_difference_cond2 {SCENARIOS} <= 0; # for linear presentation of cost_difference*cond2

minimize Total_Cost: expected_cost + lmbd*prob*risk_sum;

subject to Risk: risk_sum = sum {s in SCENARIOS} risk[s];

subject to Cost: expected_cost = sum {p in PRODUCTS, m in MONTHS} cost[p,m]*(Make[p,m] + extra_stored[p,m]*percent);
subject to Cost2 {s in SCENARIOS}: cost_difference[s] = sum {p in PRODUCTS, m in MONTHS} (cost[p,m]-costS[p,m,s])*(Make[p,m] + extra_stored[p,m]*percent);

subject to Promised {p in PRODUCTS}: sum {m in MONTHS} Make[p,m] >= promised[p];
subject to Needs {r in RESOURCES, m in MONTHS}: sum {p in PRODUCTS} Make[p,m]*demand[p,r] <= delivery[m,r];
subject to Summing {p in PRODUCTS, m in MONTHS}: Made[p,m] = sum {n in MONTHS} Make[p,n]*chronology[m,n];

subject to Storage {p in PRODUCTS, m in MONTHS}: Made[p,m] <= maxno + cond[p,m]*(promised[p]-maxno);
subject to Storage2 {p in PRODUCTS, m in MONTHS}: Made[p,m] >= maxno*cond[p,m];

subject to Conditional {p in PRODUCTS, m in MONTHS}: 0 <= extra_stored[p,m] <= promised[p]-maxno;
subject to Conditional2 {p in PRODUCTS, m in MONTHS}: (Made[p,m]-maxno) - extra_stored[p,m] + cond[p,m]*(promised[p]-maxno) <= promised[p]-maxno;

subject to Risk_Conditional {s in SCENARIOS}: risk[s] <= max_diff;
subject to Risk_Conditional2 {s in SCENARIOS}: risk[s] = cost_difference[s] - 2*cost_difference_cond2[s]; # risk[s] = cost_difference[s]*(1-2*cond2[s]);

subject to Risk_Conditional3 {s in SCENARIOS}: cost_difference_cond2[s] >=  min_diff*cond2[s];
subject to Risk_Conditional4 {s in SCENARIOS}: cost_difference_cond2[s] <= cost_difference[s];
subject to Risk_Conditional5 {s in SCENARIOS}: cost_difference[s]-cost_difference_cond2[s] + max_diff*cond2[s] <= max_diff;
