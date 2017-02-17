Smoothing using hurdle gamme derived from labeled nodes:
--------------------------------------------------------

2908 correct, 1092 incorrect, 4000 total.
0.727 percent accurate.

Smoothing using a two smoothing values (value changes from 0.03 at 20 million to 0.0001 at greater values)
------------------------------------------------------------------------------------------------------------

3133 correct, 867 incorrect, 4000 total.
0.78325 percent accurate.

Smoothing using a single smoothing value:
-------------------------------------------

2310 correct, 1690 incorrect, 4000 total.
0.5775 percent accurate

Typical ECDF
-------------

608 correct, 392 incorrect, 1000 total.
0.608 percent accurate.


599 correct, 401 incorrect, 1000 total.
0.599 percent accurate.


ECDF with zero special cased
------------------------------

681 correct, 319 incorrect, 1000 total.
0.681 percent accurate.

Two tailed ecdf, zero special case.
-------------------------------------

703 correct, 297 incorrect, 1000 total.
0.703 percent accurate.

Ignoring segments with less than 5 megabases
---------------------------------------------

571 correct, 429 incorrect, 1000 total.
0.571 percent accurate.

Relationship distance stats: DescribeResult(nobs=420, minmax=(1, 9), mean=5.5333333333333332, variance=5.2661893396976929, skewness=-0.4172305708966148, kurtosis=-1.1632814028756717)

No common ancestor occured 9 times.

**Trained also ignoring segments less than 5 Mb**

529 correct, 471 incorrect, 1000 total.
0.529 percent accurate.

Relationship distance stats: DescribeResult(nobs=456, minmax=(1, 9), mean=5.5789473684210522, variance=5.101445922498554, skewness=-0.4475874186317736, kurtosis=-1.0791944620709029)

No common ancestor occured 15 times.

Ignoring segments with less than 10 megabases
---------------------------------------------

414 correct, 586 incorrect, 1000 total.
0.414 percent accurate.

Relationship distance stats: DescribeResult(nobs=562, minmax=(1, 9), mean=5.827402135231317, variance=4.8667700661629913, skewness=-0.6232183666946493, kurtosis=-0.800250148509253)

No common ancestor occured 24 time

5% Non-paternity
----------------


1379 correct, 2621 incorrect, 4000 total.
0.34475 percent accurate.

Relationship distance stats: DescribeResult(nobs=2543, minmax=(1, 9), mean=5.841918993314982, variance=4.6461256629868695, skewness=-0.6206490960353643, kurtosis=-0.750046314830997)

No common ancestor occured 78 times.

Perturbation with 5% Non-paternity
-----------------------------------

61 correct, 139 incorrect, 200 total.
0.305 percent accurate.


Relationship distance stats: DescribeResult(nobs=133, minmax=(1, 9), mean=6.0375939849624061, variance=5.0213032581453634, skewness=-0.6472872903398937, kurtosis=-0.7229647571526283)

No common ancestor occured 6 times.

10% missing mother and father link (independent)
------------------------------------------------

549 correct, 3451 incorrect, 4000 total.
0.13725 percent accurate.

Relationship distance stats: DescribeResult(nobs=3349, minmax=(1, 9), mean=6.0830098536876678, variance=4.0068468520534131, skewness=-0.7307080943114279, kurtosis=-0.39759762448243974)

No common ancestor occured 102 times.

10% missing mother and father link; with 250 labeled nodes.
-----------------------------------

89 correct, 211 incorrect, 300 total.
0.2966666666666667 percent accurate.

No common ancestor occured 6 times.


10% missing mother and father link; ecdf for each labeled node
-----------------------------------

151 correct, 249 incorrect, 400 total.
0.3775 percent accurate.


Relationship distance stats: DescribeResult(nobs=246, minmax=(1, 9), mean=5.1300813008130079, variance=5.2891322382611587, skewness=-0.14533186121524308, kurtosis=-1.373677255119877)
No common ancestor occured 3 times.


10% missing mother and father link; replacing 0 in ecdf with min possible value; 300 labeled nodes.
-----------------------------------

370 correct, 630 incorrect, 1000 total.
0.37 percent accurate.

Relationship distance stats: DescribeResult(nobs=616, minmax=(1, 9), mean=5.2808441558441555, variance=5.0510796114454655, skewness=-0.21042520119758587, kurtosis=-1.266861223296635)
No common ancestor occured 14 times.
Error from correct node to rca stats: DescribeResult(nobs=95465, minmax=(0, 8), mean=2.3874089980621171, variance=4.278681656076329, skewness=0.5665300216286012, kurtosis=-0.7389988159390182)
Error from labeled node to rca stats: DescribeResult(nobs=89097, minmax=(0, 8), mean=2.6821778511060979, variance=4.1640615763869473, skewness=0.4223465504133544, kurtosis=-0.8122822227230837)
Fraction there is error on one side of path: 0.31675716320466196, both sides: 0.17522105978555574
Error from correct node to rca when correct stats: DescribeResult(nobs=44956, minmax=(0, 8), mean=2.9420099652994041, variance=3.9702328991110125, skewness=0.144100369293417, kurtosis=-0.9090301920527599)
Error from labeled node to rca when correct stats: DescribeResult(nobs=51473, minmax=(0, 8), mean=2.728848133973151, variance=4.268855560418543, skewness=0.41216377215533406, kurtosis=-0.8134281045417495)


10% missing mother and father link; using ECDF created with no 0 entries, 0 is special case; 300 labeled nodes.
-----------------------------------
21 correct, 379 incorrect, 400 total.
0.0525 percent accurate.


Relationship distance stats: DescribeResult(nobs=356, minmax=(1, 9), mean=6.4943820224719104, variance=3.7830669409716724, skewness=-0.8644443348511213, kurtosis=-0.032510706499715436)
No common ancestor occured 23 times.
Error from correct node to rca stats: DescribeResult(nobs=184424, minmax=(0, 8), mean=2.5479547130525311, variance=4.3114139153888793, skewness=0.47810198207676974, kurtosis=-0.8277122104407169)
Error from labeled node to rca stats: DescribeResult(nobs=179798, minmax=(0, 8), mean=2.5897896528326232, variance=4.1891129209312172, skewness=0.44136766869415445, kurtosis=-0.8092671662515682)
Fraction there is error on one side of path: 0.3143398847946276, both sides: 0.172292456653255
Error from correct node to rca when correct stats: DescribeResult(nobs=9621, minmax=(0, 7), mean=1.9104043238748571, variance=3.2225331109334849, skewness=0.6325043711027413, kurtosis=-0.6616980189629644)
Error from labeled node to rca when correct stats: DescribeResult(nobs=9403, minmax=(0, 8), mean=2.7356162926725514, variance=4.6122890984454337, skewness=0.42323811590505633, kurtosis=-0.8760906082584978)


