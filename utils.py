import sys
import argparse
import torch
import numpy as np


# Parse arguments for simulation
def get_sim_options():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--manual-control',
                        action='store_true', help='Start manual control')
    parser.add_argument('-e', '--evidence',
                        action='store_true', help='Start evidence')
    parser.add_argument('-u', '--urgency',
                        action='store_true', help='Start urgency')
    parser.add_argument('-d', '--modulation',
                        action='store_true', help='Start modulation')
    parser.add_argument('-t', '--commitment',
                        action='store_true', help='Start commitment')
    parser.add_argument('-a', '--accuracy',
                        action='store_true', help='Start accuracy')
    parser.add_argument('-s', '--cost',
                        action='store_true', help='Start cost')
    parser.add_argument('-l', '--learning',
                        action='store_true', help='Start learning')
    parser.add_argument('-k', '--likelihood',
                        action='store_true', help='Start likelihood')
    parser.add_argument('-c', '--custom',
                        action='store_true', help='Start custom')

    args = parser.parse_args()

    return args


# Parse arguments for plots
def get_plot_options():
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--evidence',
                        action='store_true', help='Plot evidence')
    parser.add_argument('-u', '--urgency',
                        action='store_true', help='Plot urgency')
    parser.add_argument('-a', '--accuracy',
                        action='store_true', help='Plot accuracy')
    parser.add_argument('-t', '--commitment',
                        action='store_true', help='Plot commitment')
    parser.add_argument('-s', '--cost',
                        action='store_true', help='Plot cost')
    parser.add_argument('-l', '--learning',
                        action='store_true', help='Plot learning')
    parser.add_argument('-k', '--likelihood',
                        action='store_true', help='Plot likelihood')
    parser.add_argument('-v', '--video',
                        action='store_true', help='Record video')

    args = parser.parse_args()
    return args


# Print simulation info
def print_info(trial, n_trials, step, n_steps, success, react):
    sys.stdout.write('\rTrial: {:4d}({:4d})/{:4d}\tStep: {:4d}/{:4d}'
                     '\t\tReact time: {:4.1f}'.format(
        trial + 1, np.count_nonzero(success), n_trials, step + 1,
        n_steps, np.nansum(react) // (trial + 1)))
    sys.stdout.flush()


# Print score
def print_score(log):
    pass


# Add Gaussian noise to array
def add_gaussian_noise(array, noise):
    sigma = noise ** 0.5
    return array + np.random.normal(0, sigma, np.shape(array))


# Normalize data
def normalize(x, limits, pyt=False, rng=True):
    limits = np.array(limits)
    if pyt:
        limits = torch.tensor(limits, dtype=torch.float32)

    x_norm = (x - limits[0]) / (limits[1] - limits[0])
    if rng:
        x_norm = x_norm * 2 - 1
    return x_norm


# Denormalize data
def denormalize(x, limits, pyt=False, rng=True):
    limits = np.array(limits)
    if pyt:
        limits = torch.tensor(limits, dtype=torch.float32)

    x_denorm = (x + 1) / 2 if rng else x
    x_denorm = x_denorm * (limits[1] - limits[0]) + limits[0]
    return x_denorm


# Compute forward kinematics
def kinematics(angles_norm, lengths, limits):
    angles = denormalize(angles_norm, limits, pyt=True)

    z = torch.tensor(0.0)
    i = torch.tensor(1.0)

    T_abs = torch.eye(3)

    for angle, length in zip(angles, lengths):
        l = torch.tensor(length, dtype=torch.float32)
        c = torch.cos(torch.deg2rad(angle))
        s = torch.sin(torch.deg2rad(angle))

        T_rel = torch.stack([torch.stack([c, -s, l * c], -1),
                            torch.stack([s, c, l * s], -1),
                            torch.stack([z, z, i], -1)], -2)

        T_abs = T_abs.matmul(T_rel)

    return T_abs[:2, 2]


# Shift (D) operator
def shift(array):
    if len(array) > 1:
        return torch.stack((*array[1:], torch.zeros_like(array[0])))
    else:
        return torch.zeros((1, len(array)))


# Normalize categorical distribution
def norm_dist(dist):
    if dist.ndim == 3:
        new_dist = np.zeros_like(dist)
        for c in range(dist.shape[2]):
            new_dist[:, :, c] = np.divide(dist[:, :, c],
                                          dist[:, :, c].sum(axis=0))
        return new_dist
    else:
        return np.divide(dist, dist.sum(axis=0))


# Sample from probability
def sample(probs):
    sample_onehot = np.random.multinomial(1, probs.squeeze())
    return np.where(sample_onehot == 1)[0][0]


# Compute stable logarithm
def log_stable(arr):
    return np.log(arr + 1e-16)


# Compute sigmoid function
def sigmoid(x, slope, bias):
    return 1 / (1 + np.exp(-slope * (x - bias)))


# Compute softmax function
def softmax(dist, precision=1.0):
    output = dist - dist.max(axis=0)
    output = np.exp(precision * output)
    output = output / np.sum(output, axis=0)
    return output


# Normalize counts
def norm_counts(counts):
    counts_normalized = np.zeros_like(counts.T)
    for c, count in enumerate(counts.T):
        counts_normalized[c] = count / np.sum(count)

    return counts_normalized.T


# Transform angle to cos/sin
def to_cos_sin(angles):
    angles_rad = np.radians(angles)

    return np.array([np.cos(angles_rad), np.sin(angles_rad)]).T


# Transform cos/sin to angle
def to_angle(cos_sin):
    if isinstance(cos_sin[0], np.ndarray):
        angles_rad = np.arctan2(cos_sin[:, 1], cos_sin[:, 0])
    else:
        angles_rad = np.arctan2(cos_sin[1], cos_sin[0])

    return np.degrees(angles_rad)


# Accumulate log evidence
def acc_log_evidence(eta, Eta_m, mu, pi, pi_m, p):
    eta = eta.detach().numpy()
    Eta_m = [eta_m.detach().numpy() for eta_m in Eta_m]
    mu = mu.detach().numpy()

    pi = pi.detach().numpy()
    pi_m = pi_m.detach().numpy()
    p = p.detach().numpy()
    p_m = p - pi + pi_m

    L = np.zeros(len(Eta_m))

    for m, eta_m in enumerate(Eta_m):
        mu_m = (p * mu - pi * eta + pi_m * eta_m) / p_m

        L[m] = np.sum(p_m * mu_m ** 2 - p * mu ** 2 +
                      pi * eta ** 2 - pi_m * eta_m ** 2)

    return L / 2


# Perform Bayesian model comparison
def bmc(probs, log_evidence, weight, gain_prior, gain_evidence):
    E = - log_stable(probs) * gain_prior - log_evidence * gain_evidence

    return softmax(-E, weight)
