"""
Construct models of a few arms - like a class per arm
this can be botorch models if we're using KG
We can use readily available functions available at botorch.test_functions.synthetic

construct an optimization routine to see how things work out

the original weight scheme does not work that well - see notes on this

currently implementing algorithms focused on pure discrete case
The nested algorithms will be compared with a straightforward application of classical algorithms
for R&S / Bandits or discrete BO to see if the nested treatment offers performance improvements
different budget scenarios should be considered, N < M, N=M, N >> M etc
"""