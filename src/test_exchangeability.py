#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt


#TODO:
# Make a program, that samples a distribution of acceptable cars, in 1D
# with fixed size. Sample by rejection sampling. Then, calcualte the likelihood.
# For uniform, this should be doable, you can calculate the available space, and
# calculate the likelihood. Check if they are the same. It sounds like it isn't.
# In particular, if you sample close to the boundary, you are more likely to do
# it later, because you block off a lot of probability mass.
#
#
#Then, do this for a distribution with support on the real line, a Gaussian,
# also here you can calculate the likelihood by renormalizing the gaussian,
# dividing by the unavailable likelihood. Again, it seems that if you sample in
# the tails first, the renorm you get is LESS than if you sample it the other
# way around.
#
# I guess another way to see this, is that if you have a sequence, where one is
# in the center, the other is in the tail, it is more likely that you sampled
# the one in the center first, because after that you could only sample in the
# tail. Damn.
#
#
a = 1.
d = 1.

class Obj():
    def __init__(self, x, d):
        self.x = x
        self.d = d

    def overlaps(self, obj):
        return np.abs(self.x - obj.x) < (self.d + obj.d)/2

    def overlaps_with_any(self, objs):
        for obj in objs:
            if self.overlaps(obj):
                return True
        return False

def sample_sequence_uniform():

    n = 2
    o_samples = []
    for n in range(n):
        while True:
            x = np.random.uniform(d/2, a-d/2)
            o = Obj(x,d)

            if not o.overlaps_with_any(o_samples):
                o_samples.append(o)
                break
    return o_samples

def sample_sequence_normal():

    n = 2
    o_samples = []
    for n in range(n):
        while True:
            x = np.random.normal(0, a)
            o = Obj(x,d)

            if not o.overlaps_with_any(o_samples):
                o_samples.append(o)
                break
    return o_samples

def print_sequence(o_samples):

    for i,o in enumerate(o_samples):
        print('Object {} with with {} and position {}'.format(i, o.d, o.x))


def calc_likelihood(o_samples):
    """fix to length two sequences for now"""

    o0 = o_samples[0]
    o1 = o_samples[1]
    p0 = 1/(a - o0.d)

    left_size = max(0.,(o0.x - o0.d/2 - o1.d))
    right_size = max(0.,(a - o0.x - o0.d/2 - o1.d))

    p1 = 1/(left_size + right_size)

    return p0 * p1

if __name__ == '__main__':

    # for i in range(10):
    #     print('')
    #     o_samples = sample_sequence_uniform()
    #     print_sequence(o_samples)
    #     print('likelihood normal order: ', calc_likelihood(o_samples))
    #     print('likelihood reverse order: ',
    #           calc_likelihood(list(reversed(o_samples))))


    nsamples = 1000000
    x0s = []
    x1s = []
    for i in range(nsamples):
        if i % 10000 == 0:
            print(i)
        o_samples = sample_sequence_normal()

        x0s.append(o_samples[0].x)
        x1s.append(o_samples[1].x)

    x0s = np.array(x0s)

    x1s = np.array(x1s)

    fig,ax = plt.subplots(figsize=(8,4))
    ax.hist(x0s, bins=50)
    plt.savefig('img/x0s.png')

    fig,ax = plt.subplots(figsize=(8,4))
    ax.hist(x1s, bins=50)
    plt.savefig('img/x1s.png')
