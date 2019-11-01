import torch
import argparse

from torch import distributions

from src.trainer import GanTrainer
from src.variables import LinearDualVariable, LinearGenerator
from src.optim import SGD


def main(args):
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    # Define true distribution and noise distrubution
    true_mean = torch.tensor([0., 0.], device=device)
    true_covariance = torch.tensor([[.1, .09], [.09, .1]], device=device)
    noise_mean = torch.tensor([0., 0.], device=args.device)
    noise_covariance = torch.tensor([[1., 0.], [0., 1.]], device=device)
    data = distributions.MultivariateNormal(true_mean, true_covariance)
    noise = distributions.MultivariateNormal(noise_mean, noise_covariance)

    # Initialize generator and dual variable
    f = LinearDualVariable(input_dim=2)
    g = LinearGenerator(noise_dim=2, output_dim=2)

    # Initialize optimizers
    f_optim = SGD(f.parameters(), lr=-args.lr)  # maximization
    g_optim = SGD(g.parameters(), lr=args.lr)  # minimization

    # Initialize trainer
    trainer = GanTrainer(
            args.batch_size, data=data, noise=noise, make_gif=args.make_gif)

    # train and save GIF
    if args.training_mode == 'simultaneous':
        trainer.simultaneous(
                n_iter=args.n_iter, f=f, g=g, f_optim=f_optim, g_optim=g_optim,
                n_checkpoints=4)
    elif args.training_mode == 'alternating':
        trainer.alternating(
                n_iter=args.n_iter, f=f, g=g, f_optim=f_optim, g_optim=g_optim,
                n_checkpoints=4)
    else:
        raise ValueError(
                'training_mode should be "simultaneous" or "alternating"')

    trainer.render_gif(
            'movie' + '_' + args.training_mode + '.gif', duration=0.1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # training parameters
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--n_iter', type=int, default=400)
    parser.add_argument('--training_mode', type=str, default='simultaneous')
    parser.add_argument('--lr', type=float, default=1e-1)

    # miscelaneous parameters
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--make_gif', type=bool, default=True)
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()
    main(args)

