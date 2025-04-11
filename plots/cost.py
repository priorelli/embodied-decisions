import numpy as np
import matplotlib.pyplot as plt
import utils
import config as c


def plot_cost(log, width):
    # Load variables
    pos = log['pos']

    t1_pos = log['t1_pos']
    t2_pos = log['t2_pos']

    states = log['states']

    causes = log['causes']

    vel = log['vel']
    est_vel = log['est_vel']
    F_m = log['F_m']

    L_ext = log['L_ext'] * 3.0
    L_softmax = np.zeros_like(L_ext)
    for t, trial in enumerate(L_ext):
        for s, step in enumerate(trial):
            L_softmax[t, s] = utils.softmax(step)

    # Initialize plots
    fig, axs = plt.subplots(3, figsize=(18, 27))
    axs[0].set_ylim(-0.05, 1.05)
    axs[1].set_ylim(-11, 0.1)
    axs[2].set_ylim(-0.05, 1.05)
    axs[0].set_xlim(1, 18)
    axs[1].set_xlim(1, 18)
    axs[2].set_xlim(1, 18)
    axs[1].set_yscale('symlog')
    axs[0].tick_params(labelbottom=False)
    axs[1].tick_params(labelbottom=False)
    axs[2].set_xlabel(r'$\tau$')
    axs[0].set_ylabel('Prob.')
    axs[1].set_ylabel('a.u.')
    axs[2].set_ylabel('Prob.')

    axs[0].plot(causes[0, :, 0][::c.n_tau], lw=width,
                label=r'$o_{h,t1}$' + ' (low)')
    axs[0].plot(causes[1, :, 0][::c.n_tau], lw=width,
                label=r'$o_{h,t1}$' + ' (medium)')
    axs[0].plot(causes[2, :, 0][::c.n_tau], lw=width,
                label=r'$o_{h,t1}$' + ' (high)')

    axs[1].plot(L_ext[0, :, 0][::c.n_tau], lw=width,
                label=r'$L_{h,t1}$' + ' (low)')
    axs[1].plot(L_ext[1, :, 0][::c.n_tau], lw=width,
                label=r'$L_{h,t1}$' + ' (medium)')
    axs[1].plot(L_ext[2, :, 0][::c.n_tau], lw=width,
                label=r'$L_{h,t1}$' + ' (high)')

    axs[2].plot(L_softmax[0, :, 0][::c.n_tau], lw=width,
                label=r'$\sigma(L_h)_{t1}$' + ' (low)')
    axs[2].plot(L_softmax[1, :, 0][::c.n_tau], lw=width,
                label=r'$\sigma(L_h)_{t1}$' + ' (medium)')
    axs[2].plot(L_softmax[2, :, 0][::c.n_tau], lw=width,
                label=r'$\sigma(L_h)_{t1}$' + ' (high)')

    axs[0].legend()
    axs[1].legend()
    axs[2].legend()

    plt.tight_layout()
    fig.savefig('plots/cost_' + c.log_name, bbox_inches='tight')

    x_range, y_range = (-550, 550), (-50, 500)
    # Initialize quivers
    titles = ['Low distance', 'Medium distance', 'High distance']
    for trial in range(len(states)):
        fig, axs = plt.subplots(1, figsize=(
            20, (y_range[1] - y_range[0]) * 20 / (x_range[1] - x_range[0])))
        axs.set_xlim(*x_range)
        axs.set_ylim(*y_range)

        x_true, u_true = get_quiver(pos[trial, :, -1], est_vel[0])
        x_pred1, u_pred1 = get_quiver(pos[trial, :, -1], F_m[0, :, 0])
        x_pred2, u_pred2 = get_quiver(pos[trial, :, -1], F_m[0, :, 1])

        t_size = c.t1_size * 250
        axs.scatter(*t1_pos[trial, 0], color='r', s=t_size, zorder=0)
        axs.scatter(*t2_pos[trial, 0], color='g', s=t_size, zorder=0)
        axs.scatter(*t1_pos[trial, 0], color='#eaeaf2',
                    s=t_size * 3 / 4, zorder=0)
        axs.scatter(*t2_pos[trial, 0], color='#eaeaf2',
                    s=t_size * 3 / 4, zorder=0)

        q = axs.quiver(*x_true.T, *u_true.T, angles='xy', color='navy',
                       width=0.003, scale=300)
        q = axs.quiver(*x_pred1.T, *u_pred1.T, zorder=-1, angles='xy',
                       color='r', width=0.003, scale=2000)
        q = axs.quiver(*x_pred2.T, *u_pred2.T, zorder=-1, angles='xy',
                       color='g', width=0.003, scale=2000)

        axs.text(x_range[0] + 40, y_range[0] + 40, titles[trial], color='grey',
                 size=50, weight='bold')

        axs.xaxis.set_visible(False)
        axs.yaxis.set_visible(False)

        fig.savefig('plots/quiver_' + c.log_name + str(trial),
                    bbox_inches='tight')


def get_quiver(position, velocity):
    n = 10
    x, u = [], []

    for step in range(0, c.n_steps, n):
        x.append(position[step])
        u.append(velocity[step])

    return np.array(x), np.array(u)
