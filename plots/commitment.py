import numpy as np
import matplotlib.pyplot as plt
import config as c


def plot_commitment(log, width):
    trial = 3

    # Load variables
    states = log['states']

    causes = log['causes']

    vel = log['vel']
    est_vel = log['est_vel']
    F_m = log['F_m']

    # Initialize plots
    fig, axs = plt.subplots(3, figsize=(16, 20))
    axs[0].set_ylim(-0.05, 1.05)
    axs[1].set_ylim(-0.05, 1.05)
    axs[2].set_ylim(-10, 250)
    axs[0].tick_params(labelbottom=False)
    axs[1].set_xlabel(r'$\tau$')
    axs[2].set_xlabel('t')
    axs[0].set_ylabel('Prob.')
    axs[1].set_ylabel('Prob.')
    axs[2].set_ylabel('Vel. (px/t)')

    axs[0].plot(states[trial, :, 0][::c.n_tau], lw=width,
                label=r'$s_{t1}$', color='r')
    axs[0].plot(states[trial, :, 1][::c.n_tau], lw=width,
                label=r'$s_{t2}$', color='g')

    axs[1].plot(causes[trial, :, 0][::c.n_tau], lw=width,
                label=r'$o_{h,t1}$', color='r')
    axs[1].plot(causes[trial, :, 1][::c.n_tau], lw=width,
                label=r'$o_{h,t2}$', color='g')
    axs[1].plot(causes[trial, :, 2][::c.n_tau], lw=width,
                label=r'$o_{h,s}$', color='purple')

    axs[2].plot(np.linalg.norm(F_m[trial, :, 0], axis=1), lw=width,
                label=r'$f_{t1}$', color='r')
    axs[2].plot(np.linalg.norm(F_m[trial, :, 1], axis=1), lw=width,
                label=r'$f_{t2}$', color='g')
    axs[2].plot(np.linalg.norm(F_m[trial, :, 2], axis=1), lw=width,
                label=r'$f_{s}$', color='purple')
    axs[2].plot(np.linalg.norm(est_vel[trial], axis=1), lw=width,
                label=r'$\mu_h^\prime$', color='navy', ls='--')

    axs[0].legend()
    axs[1].legend()
    axs[2].legend()

    plt.tight_layout()
    fig.savefig('plots/commitment_' + c.log_name, bbox_inches='tight')
