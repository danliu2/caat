"""
Tests for the C implementation of the sequence transducer.

From outside the package directory, run
`python -m transducer.test.`
"""
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import time
import torch
import torch.autograd as autograd
import torch.nn as nn

from warprnnt_pytorch import RNNTLoss
from transducer_np import RNNTLoss as rnntloss

parser = argparse.ArgumentParser(description='MXNet RNN Transducer Test.')
parser.add_argument('--np', default=False, action='store_true', help='numpy loss')
args = parser.parse_args()

fn = rnntloss() if args.np else RNNTLoss(reduction='sum')

gpu = 1
def wrap_and_call(acts, labels):
    acts = torch.FloatTensor(acts)
    if use_cuda:
        acts = acts.cuda(gpu)
    #acts = autograd.Variable(acts, requires_grad=True)
    acts.requires_grad = True

    lengths = [acts.shape[1]] * acts.shape[0]
    label_lengths = [len(l) for l in labels]
    labels = torch.IntTensor(labels)
    lengths = torch.IntTensor(lengths)
    label_lengths = torch.IntTensor(label_lengths)
    if use_cuda:
        labels = labels.cuda(gpu)
        lengths = lengths.cuda(gpu)
        label_lengths = label_lengths.cuda(gpu)

    costs = fn(acts, labels, lengths, label_lengths)
    cost = torch.sum(costs)
    cost.backward()
    # print(repr(acts.grad.data.cpu().numpy()))
    return costs.data.cpu().numpy(), acts.grad.data.cpu().numpy()


def small_test():
    acts = np.array([[[[0.1, 0.6, 0.1, 0.1, 0.1],
                      [0.1, 0.1, 0.6, 0.1, 0.1],
                      [0.1, 0.1, 0.2, 0.8, 0.1]],
                     [[0.1, 0.6, 0.1, 0.1, 0.1],
                      [0.1, 0.1, 0.2, 0.1, 0.1],
                      [0.7, 0.1, 0.2, 0.1, 0.1]]]])
    labels = [[1, 2]]

    cost, grads = wrap_and_call(acts, labels)
    expected_cost = 4.495666
    expected_grads = np.array([[[[-0.13116688, -0.3999269 ,  0.17703125,  0.17703125,
                                0.17703125],
                                [-0.18572757,  0.12247056, -0.18168412,  0.12247056,
                                0.12247056],
                                [-0.32091254,  0.06269141,  0.06928472,  0.12624499,
                                0.06269141]],

                                [[ 0.05456069, -0.21824276,  0.05456069,  0.05456069,
                                0.05456069],
                                [ 0.12073959,  0.12073959, -0.48295835,  0.12073959,
                                0.12073959],
                                [-0.6925882 ,  0.16871116,  0.18645467,  0.16871116,
                                0.16871116]]]])
    assert np.allclose(cost, expected_cost, rtol=1e-6), \
        "small_test costs mismatch."
    assert np.allclose(grads, expected_grads), \
        "small_test gradient mismatch."

def big_test():

    # minibatch x T x U x alphabet_size
    activations = [
            [[[0.06535690384862791, 0.7875301411923206, 0.08159176605666074],
              [0.5297155426466327, 0.7506749639230854, 0.7541348379087998],
              [0.6097641124736383, 0.8681404965673826, 0.6225318186056529]],

             [[0.6685222872103057, 0.8580392805336061, 0.16453892311765583],
              [0.989779515236694, 0.944298460961015, 0.6031678586829663],
              [0.9467833543605416, 0.666202507295747, 0.28688179752461884]],

             [[0.09418426230195986, 0.3666735970751962, 0.736168049462793],
              [0.1666804425271342, 0.7141542198635192, 0.3993997272216727],
              [0.5359823524146038, 0.29182076440286386, 0.6126422611507932]],

             [[0.3242405528768486, 0.8007644367291621, 0.5241057606558068],
              [0.779194617063042, 0.18331417220174862, 0.113745182072432],
              [0.24022162381327106, 0.3394695622533106, 0.1341595066017014]]],


            [[[0.5055615569388828, 0.051597282072282646, 0.6402903936686337],
              [0.43073311517251, 0.8294731834714112, 0.1774668847323424],
              [0.3207001991262245, 0.04288308912457006, 0.30280282975568984]],

             [[0.6751777088333762, 0.569537369330242, 0.5584738347504452],
              [0.08313242153985256, 0.06016544344162322, 0.10795752845152584],
              [0.7486153608562472, 0.943918041459349, 0.4863558118797222]],

             [[0.4181986264486809, 0.6524078485043804, 0.024242983423721887],
              [0.13458171554507403, 0.3663418070512402, 0.2958297395361563],
              [0.9236695822497084, 0.6899291482654177, 0.7418981733448822]],

             [[0.25000547599982104, 0.6034295486281007, 0.9872887878887768],
              [0.5926057265215715, 0.8846724004467684, 0.5434495396894328],
              [0.6607698886038497, 0.3771277082495921, 0.3580209022231813]]]]

    expected_costs = [4.2806528590890736, 3.9384369822503591]
    expected_grads = [[[[-1.86843902e-01, -6.25548810e-02,  2.49398798e-01],
                        [-2.03376666e-01,  2.02399328e-01,  9.77333169e-04],
                        [-1.41016081e-01,  7.91234672e-02,  6.18926100e-02]],

                        [[-1.15517676e-02, -8.12802389e-02,  9.28319991e-02],
                        [-1.54257029e-01,  2.29432687e-01, -7.51756504e-02],
                        [-2.46593088e-01,  1.46404594e-01,  1.00188486e-01]],

                        [[-1.29182907e-02, -6.15932420e-02,  7.45115355e-02],
                        [-5.59857301e-02,  2.19830811e-01, -1.63845062e-01],
                        [-4.97626871e-01,  2.09239945e-01,  2.88386941e-01]],

                        [[ 1.36048580e-02, -3.02196294e-02,  1.66147724e-02],
                        [ 1.13924511e-01,  6.27811998e-02, -1.76705718e-01],
                        [-6.67078257e-01,  3.67658824e-01,  2.99419403e-01]]],


                    [[[-3.56343776e-01, -5.53474613e-02,  4.11691219e-01],
                        [-9.69219357e-02,  2.94591039e-02,  6.74628317e-02],
                        [-6.35175705e-02,  2.76544970e-02,  3.58630717e-02]],

                        [[-1.54499024e-01, -7.39420280e-02,  2.28441030e-01],
                        [-1.66789949e-01, -8.78955179e-05,  1.66877866e-01],
                        [-1.72369644e-01,  1.05565332e-01,  6.68043196e-02]],

                        [[ 2.38748826e-02, -1.18255816e-01,  9.43809375e-02],
                        [-1.04707085e-01, -1.08934477e-01,  2.13641584e-01],
                        [-3.69844258e-01,  1.80118099e-01,  1.89726159e-01]],

                        [[ 2.57137045e-02, -7.94617534e-02,  5.37480488e-02],
                        [ 1.22328237e-01, -2.38788679e-01,  1.16460443e-01],
                        [-5.98686993e-01,  3.02203178e-01,  2.96483815e-01]]]]

    activations = np.array(activations)
    labels = [[1, 2],
              [1, 1]]

    costs, grads = wrap_and_call(activations, labels)

    assert np.allclose(costs, sum(expected_costs)), \
        "big_test average costs mismatch."

    assert np.allclose(grads, expected_grads, rtol=1e-3), \
        "big_test grads for average cost mismatch."

if __name__ == "__main__":
    use_cuda = False
    small_test()
    big_test()
    print("CPU Tests passed!")
    if torch.cuda.is_available():
        use_cuda = True
        small_test()
        big_test()
        print("GPU Tests passed!")
