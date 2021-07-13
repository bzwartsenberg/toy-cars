# Quick notes


## Reading
### Local Exchangeability (Trevor, Tamara Broderick)
Reading Trevor's paper. In the intro, he mentions partial exchangeability, which
seems very close to what I'm interested in. This is an older concept, and it
says that you can have a sequence that is partially exchangeable, if you have a
covariate t that indicates some equivalence class in which you are
permutationally invariant.

In the case of the cars there is such a variable, which is related to the
density at the sampled position. You can exchange the order of two cars which
take out the same amount of density, hence a variable t exists.

What is not clear is how that works out in the finite sequence setting.

More interesting, Trevor talks about local exchangeability, in which sequences
are approximately exchangeable. 

From the intro: it sounds like maybe dependent DP mixture models might be
something to look into (we have dependence between the components)

Section two, the main meat, gives t, the similarity covariate, and the a
symmetric mapping d of mcTxmcT -> [0,1], that is 0 when t = t'.  




July 13th updates
Trying to sample all the cars at once, and then after do the rejection: this way
the sequences actually become exchangeable, but there are still some
difficulties. 
 - Pyprob does not have a MV normal distribution. This means that if you learn a
 proposal, all the cars will be iid by definition, and you still have to "guess"
 the permutation at inference. So this will not work. Somehow they need to know
 about each other
    - You can generate a sequence of iid first, and then reject. This sequence
      is exchagneable, but you still have the same problem as you did first,
      which is that you have to learn the perm. inv.
  - Idea would be to train a MV gaussian proposal with ML.
  - or a neural ODE flow with ML
  - Other than that, create a sequence, a trace, but sort the values to change
    the trace after having created it, that way you should learn an ordered
    distribution in the inference net, which is fine since the sequence is
    exchagneable.
    
    
    TODO:
     - make a simple thing that learns a distribution sampling non-iid cars. Test on  neural ODE and MV gaussian with GD
     - make a model that samples n cars, but sorts the trace previous to
       ingesting it in the net.
     - do this on some more computing power

1. organize sacred, wandb
2. make a sort/train algorithm
3. make a flow or other dist that does ML on accepted cars
