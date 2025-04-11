import numpy as np
import matplotlib.pyplot as plt
import config as c


def plot_urgency(log, width):
    # Load variables
    states = log['states']

    causes = log['causes']

    est_vel = log['est_vel']

    # Initialize plots
    fig, axs = plt.subplots(3, figsize=(20, 30))
    axs[0].set_ylim(-0.05, 1.05)
    axs[1].set_ylim(-0.05, 1.05)
    # axs[2].set_ylim(-0.05, 1.05)
    axs[2].set_ylim(-0.1, 80)
    axs[0].tick_params(labelbottom=False)
    # axs[1].tick_params(labelbottom=False)
    axs[1].set_xlabel(r'$\tau$')
    axs[2].set_xlabel('t')
    axs[0].set_ylabel('Prob.')
    axs[1].set_ylabel('Prob.')
    # axs[2].set_ylabel('Prob.')
    axs[2].set_ylabel('Vel. (px/t)')

    # go_signals = [np.where(np.linalg.norm(vel, axis=1) > 15)[0][0]
    #              for vel in est_vel]
    #
    # axs[0].plot(states[0, :, 0][::c.n_tau], lw=width,
    #             label=r'$s_{t1}$' + ' (risky)')
    # axs[0].plot(states[1, :, 0][::c.n_tau], lw=width,
    #             label=r'$s_{t1}$' + ' (medium)')
    # axs[0].plot(states[2, :, 0][::c.n_tau], lw=width,
    #             label=r'$s_{t1}$' + ' (conservative)')
    #
    # axs[1].plot(causes[0, :, 0][::c.n_tau], lw=width,
    #             label=r'$o_{h,t1}$' + ' (risky)')
    # axs[1].plot(causes[1, :, 0][::c.n_tau], lw=width,
    #             label=r'$o_{h,t1}$' + ' (medium)')
    # axs[1].plot(causes[2, :, 0][::c.n_tau], lw=width,
    #             label=r'$o_{h,t1}$' + ' (conservative)')
    #
    # axs[2].plot(causes[0, :, 2][::c.n_tau], lw=width,
    #             label=r'$o_{h,s}$' + ' (risky)')
    # axs[2].plot(causes[1, :, 2][::c.n_tau], lw=width,
    #             label=r'$o_{h,s}$' + ' (medium)')
    # axs[2].plot(causes[2, :, 2][::c.n_tau], lw=width,
    #             label=r'$o_{h,s}$' + ' (conservative)')
    #
    # axs[3].plot(np.linalg.norm(est_vel[0], axis=1), lw=width,
    #             label=r'$\mu_h^\prime$' + ' (risky)')
    # axs[3].plot(np.linalg.norm(est_vel[1], axis=1), lw=width,
    #             label=r'$\mu_h^\prime$' + ' (medium)')
    # axs[3].plot(np.linalg.norm(est_vel[2], axis=1), lw=width,
    #             label=r'$\mu_h^\prime$' + ' (conservative)')
    #
    # for go, color in zip(go_signals, ['b', 'orange', 'g']):
    #     for ax in axs[:3]:
    #         ax.axvline(go // c.n_tau, lw=width - 2, ls='--', color=color)
    #     axs[3].axvline(go, lw=width - 2, ls='--', color=color)

    axs[0].plot(states[0, :, 0][::c.n_tau], lw=width,
                label=r'$s_{t1}$' + ' (high evidence, low urgency)')
    axs[0].plot(states[1, :, 0][::c.n_tau], lw=width,
                label=r'$s_{t1}$' + ' (high urgency, low evidence)')

    axs[1].plot(causes[0, :, 0][::c.n_tau], lw=width,
                label=r'$o_{h,t1}$' + ' (high evidence, low urgency)')
    axs[1].plot(causes[1, :, 0][::c.n_tau], lw=width,
                label=r'$o_{h,t1}$' + ' (high urgency, low evidence)')

    axs[2].plot(np.linalg.norm(est_vel[0], axis=1), lw=width,
                label=r'$\mu_h^\prime$' + ' (high evidence, low urgency)')
    axs[2].plot(np.linalg.norm(est_vel[1], axis=1), lw=width,
                label=r'$\mu_h^\prime$' + ' (high urgency, low evidence)')

    axs[0].legend()
    axs[1].legend()
    axs[2].legend()
    # axs[3].legend()

    plt.tight_layout()
    fig.savefig('plots/urgency_' + c.log_name, bbox_inches='tight')


def correlation(states, vel_true):
    return np.corrcoef(states, vel_true)
