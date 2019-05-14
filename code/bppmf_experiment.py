
from argparse import ArgumentParser

import numpy as np
import numpy.random as rn
from scriptine import path

#pylint: disable=E0611
from bppmf import BPPMF


def get_args():
    p = ArgumentParser()
    p.add_argument('-o', '--out', type=path, required=True)
    p.add_argument('-k', '--n_topics', type=int, default=50)
    p.add_argument('-v', '--verbose', action="store_true", default=False)
    p.add_argument('--rho', type=float, default=0.)
    p.add_argument('--shp', type=float, default=0.1)
    p.add_argument('--rte', type=float, default=1.0)
    p.add_argument('--trunc', action="store_true", default=False)
    p.add_argument('--seed', type=int, default=None)
    p.add_argument('--n_burnin_itns', type=int, default=2000)
    p.add_argument('--n_epoch_itns', type=int, default=100)
    p.add_argument('--n_epochs', type=int, default=25)
    p.add_argument('--data_file', required=True)
    p.add_argument('--privacy_interval', type=int, default=1)
    args = p.parse_args()
    return args


def save_state_file(
    theta_DK,
    phi_KV,
    initialization={},
    out_file='out.npz'
):
    output_dict = initialization.copy()
    
    # Save parameters
    output_dict['theta_DK'] = theta_DK
    output_dict['phi_KV'] = phi_KV

    np.savez_compressed(out_file, **output_dict)


def make_scheduler(burnin=0, interval=1):
    return lambda iter_idx: (iter_idx >= burnin) and (iter_idx % interval == 0)


def two_sided_geometric(p, size=()):
    """Generate 2-sided geometric noise"""
    if p == 0:
        return np.zeros(size)

    rte = (1-p) / p
    Lambda_2 = rn.gamma(1, 1. / rte, size=(2,) + size)
    G_2 = rn.poisson(Lambda_2)
    return G_2[0] - G_2[1]


def main():
    args = get_args()
    data_file = np.load(args.data_file)
    Y_DV = data_file['Y_DV']
    Y_DV = Y_DV.astype(np.int16)

    schedule = {}
    if args.privacy_interval > 1:
        schedule['Lambda_2DV'] = make_scheduler(interval=args.privacy_interval)
        schedule['G_2DV'] = make_scheduler(interval=args.privacy_interval)

    D, V = Y_DV.shape
    K = args.n_topics
    rn.seed(args.seed)
    out_dir = args.out

    noised_data_DV = Y_DV + two_sided_geometric(args.rho, size=Y_DV.shape)
    noised_data_DV = noised_data_DV.astype(np.int16)

    verbose = args.verbose
    if verbose:
        print('Starting inference...')

    priv_DV = np.ones((D, V)) * args.rho
    model = BPPMF(D=D,
                  V=V,
                  K=K,
                  a=args.shp,
                  b=args.rte,
                  trunc=args.trunc,
                  debug=0,
                  seed=args.seed)

    model.fit(data=noised_data_DV,
              priv=priv_DV,
              mask=None,
              num_itns=args.n_burnin_itns,
              verbose=verbose,
              initialize=True,
              schedule=schedule)

    if verbose:
        print('Burned in...')

    avg_theta_DK = np.zeros((D, K))
    avg_phi_KV = np.zeros((K, V))

    for epoch in range(args.n_epochs):
        model.fit(data=noised_data_DV,
                  priv=priv_DV,
                  mask=None,
                  num_itns=args.n_epoch_itns,
                  verbose=verbose,
                  initialize=False,
                  schedule=schedule)

        if verbose:
            print('Epoch %d' % epoch)

        state = dict(model.get_state())
        Theta_DK = state['Theta_DK'].copy()
        Phi_KV = state['Phi_KV'].copy()

        avg_theta_DK += Theta_DK
        avg_phi_KV += Phi_KV

        save_state_file(
            Theta_DK,
            Phi_KV,
            None,
            Y_DV,
            out_file=out_dir.joinpath('sample_{}.npz'.format(epoch))
        )

    avg_theta_DK /= float(args.n_epochs)
    avg_phi_KV /= float(args.n_epochs)

    initialization = {
        'seed': args.seed,
        'rho': args.rho,
        'K': K,
        'D': D,
        'V': V,
        'a': args.shp,
        'b': args.rte,
        'noisy_data': noised_data_DV
    }
    save_state_file(
        avg_theta_DK,
        avg_phi_KV,
        None,
        Y_DV,
        initialization=initialization,
        out_file=out_dir.joinpath('avg_state.npz')
    )


if __name__ == '__main__':
    main()
