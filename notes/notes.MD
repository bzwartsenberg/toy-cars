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
