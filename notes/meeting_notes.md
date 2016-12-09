12-22-15
========
* Disambiguate degree of relationship vs nth cousin
* Don't use naive bayes to determine relatedness at first. Instead use total length of shared segments. Only use bayes later if it gives better results.
* Use naive bayes / likelikhood maximizaion on vector of relatedness to determine the individual.

03-01-16
========

* See if it is possible to determine distribution automatically.
* Sanity check the distributions against other results or data.
* See if it is possible to do on demand determination of distributions.
* Try to understand if the different types of relationships that share a vector have different distributions. eg for (2, 2) is the distribution of siblings vs siblings who have had a kid with eachother (compared to the grandparent) the same?

10-12-16
========

* Play with smoothing parameter, what gives better accuracy in the case without full genealogy?
* When running on the full genealogy, how many labeled nodes are necessary to get 50% accuracy.
* How many labeled nodes in our model necessary wrt to the actual node to get 50 accuracy?

10-14-16
========

* Remind Arvind to look into theory of deanon for our project. Conversation about error lead into this discussion.

10-17-16
========

* Search for best smoothing parameter automatically, then generate graph with # labeled nodes x axis, % accuracy y axis
* Understand theory behind inference problem.

10-19-16
========

* There will be some distribution for IBD between the founders. Find this in the literature and adapt it for use in our algorithm.
* For now make up a gamma function and see how it performs. Maybe exponential.

10-24-16
========

* Think about better organizational structures to manage personal task parallelism
* Mess with parameters in parallel with thinking about theory for what those parameters should be and mean

10-28-16
=========

* Fix simulation next week
* Determine some parameter to describe background level of sharing in any two founders.

11-14-16
========

* Only draw test nodes from last generation, see if that makes a difference with respect to distance to the cutoff.

11-15-16
========

* Try more values for two point smoothing (eg 10,000,000 bases)
* Try empirical distribution rather than fitted distribution, we have enough data
* Test correctness of existing code.
* Try single tailed test.
* Try ignoring zero

12-02-16
========

* Try paternity error
* Paper notes:
     * Structure paper so its clear what and why I am talking about
     * Defend our choices more than just stating them
     * What will reviewers be skeptical of?
         * Lots of questions about model, why it wont work in the real world?
         
12-07-16
=========

* Have an approach for each type of error by next meeting

Command to generate section headers:
date +"%m-%d-%y"
