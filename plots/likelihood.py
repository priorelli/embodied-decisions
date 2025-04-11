import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import config as c


def plot_cue(log, width):
    # Load variables
    states = log['states']
    causes = log['causes']

    est_vel = log['est_vel']

    a_cue = log['a_cue']
    A_cue = log['A_cue']

    # Initialize mappings
    fig, axs = plt.subplots(2, 4, figsize=(35, 15))

    x_labels = r'$s_{t1}$', r'$s_{t2}$'
    y_labels = r'$o_{c,t1}$', r'$o_{c,t2}$'
    trials = np.concatenate([np.linspace(0, 30, 4, dtype=int),
                             np.linspace(31, 59, 4, dtype=int)])
    for t, trial in enumerate(trials):
        sns.heatmap(A_cue[trial], ax=axs[t // 4, t % 4], annot=True,
                    xticklabels=x_labels, yticklabels=y_labels,
                    vmin=np.min(A_cue), vmax=np.max(A_cue), fmt='.2f')

    plt.tight_layout()
    plt.savefig('plots/likelihood_cue_' + c.log_name, bbox_inches='tight')

    # Initialize plots
    fig, axs = plt.subplots(2, 3, figsize=(35, 15))
    axs[0, 0].tick_params(labelbottom=False)
    axs[1, 0].set_ylim(-0.05, 1.05)
    axs[1, 0].set_xlabel('# of trials')
    axs[0, 0].set_ylabel('Counts (a.u.)')
    axs[1, 0].set_ylabel('Prob.')
    axs[0, 1].tick_params(labelbottom=False)
    axs[0, 1].set_ylim(-0.05, 1.05)
    axs[1, 1].set_ylim(-0.05, 1.05)
    axs[1, 1].set_xlabel(r'$\tau$')
    axs[0, 1].set_ylabel('Prob.')
    axs[1, 1].set_ylabel('Prob.')
    # axs[0, 2].tick_params(labelbottom=False)
    # axs[0, 2].set_ylim(-0.05, 1.05)
    # axs[1, 2].set_ylim(-0.05, 1.05)
    # axs[1, 2].set_xlabel(r'$\tau$')
    # axs[0, 2].set_ylabel('Prob.')
    # axs[1, 2].set_ylabel('Prob.')
    axs[0, 2].tick_params(labelbottom=False)
    axs[0, 2].set_ylim(-1, 50)
    axs[1, 2].set_ylim(-1, 50)
    axs[1, 2].set_xlabel('t')
    axs[0, 2].set_ylabel('Vel. (px/t)')
    axs[1, 2].set_ylabel('Vel. (px/t)')
    axs[0, 1].set_title(r'$s_{t1}$')
    # axs[0, 2].set_title(r'$o_{h, t1}$')
    axs[0, 2].set_title(r'$||\mu_{h}||$')

    axs[0, 0].plot(a_cue[:, 0, 0], lw=width, label=r'$a_{c,00}$', color='r')
    axs[0, 0].plot(a_cue[:, 1, 0], lw=width, label=r'$a_{c,10}$', color='g')

    axs[1, 0].plot(A_cue[:, 0, 0], lw=width, label=r'$A_{c,00}$', color='r')
    axs[1, 0].plot(A_cue[:, 1, 0], lw=width, label=r'$A_{c,10}$', color='g')

    ranges = (2, 29), (31, 59)
    for ax, rng in zip([axs[0, 1], axs[1, 1]], ranges):
        arr = states[np.arange(*rng, 6)]
        sns.lineplot(data=arr[:, ::c.n_tau, 0].T, ax=ax,
                     lw=width, palette='coolwarm', legend=False)

    # for ax, rng in zip([axs[0, 2], axs[1, 2]], ranges):
    #     arr = causes[np.arange(*rng, 6)]
    #     sns.lineplot(data=arr[:, ::c.n_tau, 0].T, ax=ax,
    #                  lw=width, palette='coolwarm', legend=False)

    for ax, rng in zip([axs[0, 2], axs[1, 2]], ranges):
        arr = est_vel[np.arange(*rng, 6)]
        sns.lineplot(data=np.linalg.norm(arr, axis=2).T, ax=ax,
                     lw=width, palette='coolwarm', legend=False)

    for ax in (axs[0, 0], axs[1, 0]):
        ax.axvline(30, lw=width - 2, ls='--')

    axs[0, 0].legend()
    axs[1, 0].legend()

    plt.tight_layout()
    fig.savefig('plots/learning_cue_' + c.log_name, bbox_inches='tight')


def plot_ext(log, width):
    # Load variables
    states = log['states']
    causes = log['causes']

    est_vel = log['est_vel']

    a_ext = log['a_ext']
    A_ext = log['A_ext']

    # Initialize mappings
    fig, axs = plt.subplots(2, 4, figsize=(35, 16))

    x_labels = r'$s_{t1}$', r'$s_{t2}$'
    y_labels = r'$o_{h,t1}$', r'$o_{h,t2}$', r'$o_{h,s}$'
    trials = np.concatenate([np.linspace(0, 30, 4, dtype=int),
                             np.linspace(31, 59, 4, dtype=int)])

    for t, trial in enumerate(trials):
        sns.heatmap(A_ext[trial], ax=axs[t // 4, t % 4], annot=True,
                    xticklabels=x_labels, yticklabels=y_labels,
                    vmin=np.min(A_ext), vmax=np.max(A_ext),
                    cmap='crest', fmt='.2f')

    plt.tight_layout()
    plt.savefig('plots/likelihood_ext_' + c.log_name, bbox_inches='tight')

    # Initialize plots
    fig, axs = plt.subplots(2, 3, figsize=(35, 15))
    axs[0, 0].tick_params(labelbottom=False)
    axs[1, 0].set_ylim(-0.05, 1.05)
    axs[1, 0].set_xlabel('# of trials')
    axs[0, 0].set_ylabel('Counts (a.u.)')
    axs[1, 0].set_ylabel('Prob.')
    axs[0, 1].tick_params(labelbottom=False)
    axs[0, 1].set_ylim(-0.05, 1.05)
    axs[1, 1].set_ylim(-0.05, 1.05)
    axs[1, 1].set_xlabel(r'$\tau$')
    axs[0, 1].set_ylabel('Prob.')
    axs[1, 1].set_ylabel('Prob.')
    # axs[0, 2].tick_params(labelbottom=False)
    # axs[0, 2].set_ylim(-0.05, 1.05)
    # axs[1, 2].set_ylim(-0.05, 1.05)
    # axs[1, 2].set_xlabel(r'$\tau$')
    # axs[0, 2].set_ylabel('Prob.')
    # axs[1, 2].set_ylabel('Prob.')
    axs[0, 2].tick_params(labelbottom=False)
    axs[0, 2].set_ylim(-1, 50)
    axs[1, 2].set_ylim(-1, 50)
    axs[1, 2].set_xlabel('t')
    axs[0, 2].set_ylabel('Vel. (px/t)')
    axs[1, 2].set_ylabel('Vel. (px/t)')
    axs[0, 1].set_title(r'$s_{t1}$')
    # axs[0, 2].set_title(r'$o_{h, t1}$')
    axs[0, 2].set_title(r'$||\mu_{h}||$')

    axs[0, 0].plot(a_ext[:, 0, 0], lw=width, label=r'$a_{h,00}$', color='r')
    axs[0, 0].plot(a_ext[:, 1, 0], lw=width, label=r'$a_{h,10}$',
                   color='purple')
    axs[0, 0].plot(a_ext[:, 2, 0], lw=width, label=r'$a_{h,20}$', color='g')

    axs[1, 0].plot(A_ext[:, 0, 0], lw=width, label=r'$A_{h,00}$', color='r')
    axs[1, 0].plot(A_ext[:, 1, 0], lw=width, label=r'$A_{h,10}$',
                   color='purple')
    axs[1, 0].plot(A_ext[:, 2, 0], lw=width, label=r'$A_{h,20}$', color='g')

    ranges = (2, 29), (31, 59)
    for ax, rng in zip([axs[0, 1], axs[1, 1]], ranges):
        arr = states[np.arange(*rng, 6)]
        sns.lineplot(data=arr[:, ::c.n_tau, 0].T, ax=ax,
                     lw=width, palette='coolwarm', legend=False)

    # for ax, rng in zip([axs[0, 2], axs[1, 2]], ranges):
    #     arr = causes[np.arange(*rng, 6)]
    #     sns.lineplot(data=arr[:, ::c.n_tau, 0].T, ax=ax,
    #                  lw=width, palette='coolwarm', legend=False)

    for ax, rng in zip([axs[0, 2], axs[1, 2]], ranges):
        arr = est_vel[np.arange(*rng, 6)]
        sns.lineplot(data=np.linalg.norm(arr, axis=2).T, ax=ax,
                     lw=width, palette='coolwarm', legend=False)

    for ax in (axs[0, 0], axs[1, 0]):
        ax.axvline(30, lw=width - 2, ls='--')

    axs[0, 0].legend()
    axs[1, 0].legend()

    plt.tight_layout()
    fig.savefig('plots/learning_ext_' + c.log_name, bbox_inches='tight')
