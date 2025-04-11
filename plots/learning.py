import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import config as c


def plot_learning(log, width):
    # Initialize body
    idxs = {}
    ids = {joint: j for j, joint in enumerate(c.joints)}
    size = np.zeros((c.n_joints, 2))
    for joint in c.joints:
        size[ids[joint]] = c.joints[joint]['size']
        if c.joints[joint]['link']:
            idxs[ids[joint]] = ids[c.joints[joint]['link']]
        else:
            idxs[ids[joint]] = -1

    # Load variables
    pos = log['pos']

    t1_pos = log['t1_pos']
    t2_pos = log['t2_pos']

    counts = log['counts']
    states = log['states']

    vel = log['vel']

    # Initialize plots
    fig, axs = plt.subplots(5, figsize=(20, 25))
    axs[1].set_ylim(-0.05, 1.05)
    axs[2].set_ylim(-0.1, 350)
    axs[3].set_ylim(-0.05, 1.05)
    axs[4].set_ylim(-0.05, 1.05)
    axs[0].tick_params(labelbottom=False)
    axs[1].tick_params(labelbottom=False)
    axs[2].set_xlabel('# of trials')
    axs[3].tick_params(labelbottom=False)
    axs[4].set_xlabel(r'$\tau$')
    axs[0].set_ylabel('Counts (a.u.)')
    axs[1].set_ylabel('Prob.')
    axs[2].set_ylabel('t')
    axs[3].set_ylabel('Prob.')
    axs[4].set_ylabel('Prob.')

    axs[0].plot(counts[:, 0], lw=width, label=r'$d_{t1}$', color='r')
    axs[0].plot(counts[:, 1], lw=width, label=r'$d_{t1}$', color='g')

    axs[1].plot(states[:, 0, 0], lw=width, label=r'$D_{t1}$', color='r')
    axs[1].plot(states[:, 0, 1], lw=width, label=r'$D_{t1}$', color='g')

    onset = []
    for v_t in vel:
        onset.append(np.where(np.linalg.norm(v_t, axis=1) > 6.0)[0][0])
    axs[2].plot(onset, lw=width, color='navy')

    range_reversal = [(0, len(states) // 5),
                      (len(states) // 5, len(states))]
    for ax, rng in zip([axs[3], axs[4]], range_reversal):
        arr = states[rng[0]:rng[1]]
        sns.lineplot(data=arr[::len(arr) // 5, ::c.n_tau, 0].T, ax=ax,
                     lw=width, palette='coolwarm', legend=False)

    for ax in axs[:3]:
        ax.axvline(len(states) // 5, lw=width - 2, ls='--')

    axs[0].legend()
    axs[1].legend()

    plt.tight_layout()
    fig.savefig('plots/learning_' + c.log_name, bbox_inches='tight')

    # Initialize frame of trajectories
    scale = 1.2
    x_range, y_range = (-450, 450), (-100, 500)
    fig, axs = plt.subplots(2, figsize=(
        16, (y_range[1] - y_range[0]) * 32 / (x_range[1] - x_range[0])))

    for ax, rng, title in zip(axs, range_reversal,
                              ['Learning', 'Reversal learning']):
        # Clear plot
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.set_xlim(-450, 450)
        ax.set_ylim(-100, 550)

        for j in range(c.n_joints):
            # Draw real body
            ax.plot(*np.array([pos[0, 0, idxs[j] + 1], pos[0, 0, j + 1]]).T,
                    lw=size[j, 1] * scale, color='b', zorder=1)

        # Draw targets
        t_size = c.t1_size * scale * 400
        ax.scatter(*t1_pos[0, 0], color='r', s=t_size, zorder=0)
        ax.scatter(*t2_pos[0, 0], color='g', s=t_size, zorder=0)
        ax.scatter(*t1_pos[0, 0], color='#eaeaf2', s=t_size * 3 / 4, zorder=0)
        ax.scatter(*t2_pos[0, 0], color='#eaeaf2', s=t_size * 3 / 4, zorder=0)

        # Draw real body trajectory
        arr = pos[rng[0]:rng[1]]
        arr = arr[::len(arr) // 10, ::1, -1]
        x, y, z = arr.shape
        df = pd.DataFrame(arr.transpose(2, 0, 1).reshape(2, -1).T,
                          index=np.repeat(np.arange(x), y),
                          columns=['x', 'y'])
        df['trial'] = df.index
        sns.scatterplot(data=df, x='x', y='y', hue='trial', s=120,
                        palette='coolwarm', ax=ax, legend=False)

        ax.text(x_range[0] + 40, y_range[0] + 40, title,
                color='grey', size=55, weight='bold')

    plt.tight_layout()
    plt.savefig('plots/trajectories_' + c.log_name, bbox_inches='tight')
