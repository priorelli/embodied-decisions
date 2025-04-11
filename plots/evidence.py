import numpy as np
import scipy
from numpy.linalg import norm as norm
import matplotlib.pyplot as plt
import config as c


def plot_evidence(log, width):
    trial = 203

    # Load variables
    pos = log['pos']

    t1_pos = log['t1_pos']
    t2_pos = log['t2_pos']

    states = log['states']
    cues = log['cues']

    vel = log['vel']
    est_vel = log['est_vel']
    F_m = log['F_m']

    # Initialize plots
    fig, axs = plt.subplots(3, figsize=(16, 16))
    axs[0].set_ylim(-0.05, 1.05)
    axs[1].set_ylim(-0.1, 15.1)
    axs[2].set_ylim(-0.1, 600)
    axs[0].tick_params(labelbottom=False)
    axs[1].tick_params(labelbottom=False)
    axs[0].set_ylabel('Prob.')
    axs[1].set_ylabel('Cues')
    axs[2].set_ylabel('Dist. (px)')
    axs[2].set_xlabel('t')

    axs[0].plot(states[trial, :, 0], lw=width, label=r'$s_{t1}$', color='r')
    axs[0].plot(states[trial, :, 1], lw=width, label=r'$s_{t2}$', color='g')

    axs[1].plot(np.repeat(np.cumsum(cues[trial, :, 0]), c.n_tau), lw=width,
                label=r'$o_{c,t1}$', color='r')
    axs[1].plot(np.repeat(np.cumsum(cues[trial, :, 1]), c.n_tau), lw=width,
                label=r'$o_{c,t2}$', color='g')

    t1_dist = np.linalg.norm(pos[trial, :, -1] - t1_pos[trial], axis=1)
    t2_dist = np.linalg.norm(pos[trial, :, -1] - t2_pos[trial], axis=1)

    axs[2].plot(t1_dist, lw=width, label=r'$||x_h - t_1||$', color='r')
    axs[2].plot(t2_dist, lw=width, label=r'$||x_h - t_2||$', color='g')

    # axs[0].set_ylim(-0.05, 1.05)
    # axs[1].set_ylim(-2, 80)
    # axs[2].set_ylim(-20, 600)
    # axs[0].tick_params(labelbottom=False)
    # axs[1].tick_params(labelbottom=False)
    # axs[2].set_xlabel('t')
    # axs[0].set_ylabel('Prob.')
    # axs[1].set_ylabel('Dist. (px)')
    # axs[2].set_ylabel('Dist. (px)')
    #
    # axs[0].plot(states[trial, :, 0], lw=width, label=r'$s_{t1}$', color='r')
    # axs[0].plot(states[trial, :, 1], lw=width, label=r'$s_{t2}$', color='g')
    #
    # axs[1].plot(np.linalg.norm(est_vel[trial], axis=1), lw=width,
    #             label=r'$\mu_h^\prime$', color='navy')
    # axs[1].plot(np.linalg.norm(vel[trial], axis=1), lw=width,
    #             label=r'$x_h^\prime$', color='b', ls='--')
    #
    # t1_dist = np.linalg.norm(pos[trial, :, -1] - t1_pos[trial], axis=1)
    # t2_dist = np.linalg.norm(pos[trial, :, -1] - t2_pos[trial], axis=1)
    #
    # axs[2].plot(t1_dist, lw=width, label=r'$||x_h - t_1||$', color='r')
    # axs[2].plot(t2_dist, lw=width, label=r'$||x_h - t_2||$', color='g')

    axs[0].legend()
    axs[1].legend()
    axs[2].legend()

    plt.tight_layout()
    fig.savefig('plots/evidence_' + c.log_name, bbox_inches='tight')

    # Print maximum deviations
    # maximum_deviation(pos, t1_pos)


def maximum_deviation(pos, t1_pos):
    max_devs = np.zeros(len(pos))

    for trial in range(len(pos)):
        max_devs[trial] = np.max([norm(np.cross(
            pos[0, 0] - t1_pos[0, 0], pos[0, 0] - pos[trial, step])) / norm(
            t1_pos[0, 0] - pos[0, 0]) for step in range(c.n_steps)])

    print('===\nCongruent tials\n===')
    print('Mean: {:.2f}'.format(np.mean(max_devs[:100])))
    print('Std: {:.2f}'.format(np.std(max_devs[:100])))
    print('===\nIncongruent tials\n===')
    print('Mean: {:.2f}'.format(np.mean(max_devs[100:200])))
    print('Std: {:.2f}'.format(np.std(max_devs[100:200])))
    print('===\nNeutral tials\n===')
    print('Mean: {:.2f}'.format(np.mean(max_devs[200:])))
    print('Std: {:.2f}'.format(np.std(max_devs[200:])))

    print('\n Congruent-incongruent')
    print(scipy.stats.ttest_ind(max_devs[:100], max_devs[100:200]))
    print('\n Congruent-neutral')
    print(scipy.stats.ttest_ind(max_devs[:100], max_devs[200:]))
    print('\n Incongruent-neutral')
    print(scipy.stats.ttest_ind(max_devs[100:200], max_devs[200:]))
