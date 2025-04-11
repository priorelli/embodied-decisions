import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
from pylab import tight_layout
import time
import sys
import utils
import config as c


def record_video(log, width):
    plot_type = 0
    frame = c.n_steps - 1  # c.n_steps - 1
    trial = 0
    text = r'$\tau=1$'
    # text = 'Incongruent \t\t\t  ' + r'$k_d=0.2$'
    # text = r'$\alpha_c=0.49$' + '\t\t\t\t' + r'$\alpha_h=0.2$'
    dynamics = True

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
    pos = log['pos'][trial]
    est_pos = log['est_pos'][trial]

    t1_pos = log['t1_pos'][trial]
    t2_pos = log['t2_pos'][trial]

    states = log['states'][trial]
    causes = log['causes'][trial]

    est_vel = log['est_vel'][trial]
    F_m = log['F_m'][trial]

    L_ext = log['L_ext'][trial]
    L_softmax = np.zeros((len(L_ext), 2))
    for s, step in enumerate(L_ext):
        L_softmax[s] = utils.softmax(step[:2] * c.gain_evidence)
    L_softmax[:30, :] = 0.5

    cues = log['cues'][trial]
    cues_pos = []
    for n_cue, cue in enumerate(cues):
        # np.random.seed(n_cue)
        r = np.random.randint(c.t1_size)
        theta = np.radians(np.random.randint(360))
        cue_pos = np.array([r * np.cos(theta), r * np.sin(theta)])

        if cue[0] == 1:
            cues_pos.append(cue_pos + t1_pos[0])
        elif cue[1] == 1:
            cues_pos.append(cue_pos + t2_pos[0])
        else:
            cues_pos.append((0, 0))

    # Create plot
    scale = 1.2
    x_range, y_range = (-450, 450), (-200, 600)
    if dynamics:
        fig = plt.figure(figsize=(32, (y_range[1] - y_range[0]) * 16 /
                                  (x_range[1] - x_range[0])))
        gs = GridSpec(2, 2, figure=fig)

        axs = [fig.add_subplot(gs[:, 0]), fig.add_subplot(gs[0, 1]),
               fig.add_subplot(gs[1, 1])]
    else:
        fig = plt.figure(figsize=(16, (y_range[1] - y_range[0]) * 16 /
                                  (x_range[1] - x_range[0])))
        gs = GridSpec(1, 1, figure=fig)

        axs = [fig.add_subplot(gs[:, 0])]

    xlims = [x_range, (0, 750), (0, 750)]
    ylims = [y_range, (-0.05, 1.05), (-0.05, 1.05)]
    titles = ['', '', '']

    def animate(n):
        if (n + 1) % 10 == 0:
            sys.stdout.write('\rStep: {:d}'.format(n + 1))
            sys.stdout.flush()

        # Clear plot
        n_axs = len(axs)
        for w, xlim, ylim, title in zip(range(n_axs), xlims, ylims, titles):
            axs[w].clear()
            axs[w].set_xlim(xlim)
            axs[w].set_ylim(ylim)
            # axs[w].title.set_text(title)
        axs[0].get_xaxis().set_visible(False)
        axs[0].get_yaxis().set_visible(False)
        tight_layout()
        # fig.subplots_adjust(wspace=0.4, hspace=1.0)

        #########
        # VIDEO #
        #########

        # Draw text
        axs[0].text(x_range[0] + 40, y_range[0] + 40, '%d' % n,
                    color='grey', size=60, weight='bold')

        for j in range(c.n_joints):
            # Draw real body
            axs[0].plot(*np.array([pos[n, idxs[j] + 1], pos[n, j + 1]]).T,
                        lw=size[j, 1] * scale, color='b', zorder=1)

        # Draw targets
        t_size = c.t1_size * scale * 400
        axs[0].scatter(*t1_pos[n], color='r', s=t_size, zorder=0)
        axs[0].scatter(*t2_pos[n], color='g', s=t_size, zorder=0)
        axs[0].scatter(*t1_pos[n], color='#eaeaf2', s=t_size * 3/4, zorder=0)
        axs[0].scatter(*t2_pos[n], color='#eaeaf2', s=t_size * 3/4, zorder=0)

        # Draw real body trajectory
        axs[0].scatter(*pos[n - (n % len(states)): n + 1, -1].T,
                       color='darkblue', lw=1, zorder=2, alpha=0.2)

        # Draw average trajectory
        # axs[0].scatter(*np.average(log['pos'][200:, :, -1], axis=0).T,
        #                color='darkblue', lw=width - 2, zorder=2)
        # plt.plot([pos[0, -1, 0], pos[-1, -1, 0]],
        #          [pos[0, -1, 1], pos[-1, -1, 1]],
        #          lw=width - 1, zorder=2, ls='--', color='darkblue')

        # Draw cues
        for n_cue, cue_pos in enumerate(cues_pos[:n // c.n_tau]):
            axs[0].scatter(*cue_pos, color='grey', s=500, zorder=0)
        axs[0].scatter(*cues_pos[n // c.n_tau], color='purple',
                       s=1500, zorder=0)
        # for n_cue, cue_pos in enumerate(cues_pos[:n // c.n_tau + 1]):
        #     axs[0].scatter(*cue_pos, color='grey', s=500, zorder=0)

        # Draw quivers
        if dynamics:
            x_true, u_true = pos[n, -1], est_vel[n]
            x_pred1, u_pred1 = pos[n, -1], F_m[n, 0]
            x_pred2, u_pred2 = pos[n, -1], F_m[n, 1]

            q = axs[0].quiver(*x_true.T, *u_true.T, angles='xy', color='navy',
                              width=0.006, scale=200)
            q = axs[0].quiver(*x_pred1.T, *u_pred1.T, angles='xy',
                              color='r', width=0.006, scale=1000)
            q = axs[0].quiver(*x_pred2.T, *u_pred2.T, angles='xy',
                              color='g', width=0.006, scale=1000)

        ############
        # DYNAMICS #
        ############

        if dynamics:
            axs[1].plot(states[:n, 0], lw=width, label=r'$s_{t1}$', color='r')
            axs[1].plot(states[:n, 1], lw=width, label=r'$s_{t2}$', color='g')

            # axs[2].plot(causes[:n, 0], lw=width, label=r'$o_{h,t1}$', color='r')
            # axs[2].plot(causes[:n, 1], lw=width, label=r'$o_{h,t2}$', color='g')
            # axs[2].plot(causes[:n, 2], lw=width, label=r'$o_{h,s}$', color='purple')

            axs[2].plot(np.repeat(L_softmax[:, 0][::c.n_tau], c.n_tau)[:n],
                        lw=width, label=r'$\sigma(L_h)_{t1}$', color='r')
            axs[2].plot(np.repeat(L_softmax[:, 1][::c.n_tau], c.n_tau)[:n],
                        lw=width, label=r'$\sigma(L_h)_{t2}$', color='g')
            # axs[2].plot(L_softmax[:n, 0], lw=width,
            #             label=r'$\sigma(L_h)_{s}$', color='purple')

            # axs[2].plot(np.linalg.norm(Vels_pred[:n, 0], axis=1), lw=width,
            #             label=r'$f_{t1}$', color='r')
            # axs[2].plot(np.linalg.norm(Vels_pred[:n, 1], axis=1), lw=width,
            #             label=r'$f_{t2}$', color='g')
            # axs[2].plot(np.linalg.norm(vel_pred[:n], axis=1), lw=width,
            #             label=r'$\mu_h^\prime$', color='navy', ls='--')

            axs[1].legend(loc='upper right')
            axs[2].legend(loc='upper right')

    # Plot video
    if plot_type == 0:
        start = time.time()
        ani = animation.FuncAnimation(fig, animate, len(states))
        writer = animation.writers['ffmpeg'](fps=50)
        ani.save('plots/video.mp4', writer=writer)
        print('\nTime elapsed:', time.time() - start)

    # Plot frame sequence
    elif plot_type == 1:
        for i in range(0, len(states), c.n_tau):
            animate(i)
            plt.savefig('plots/frame_%d' % i)

    # Plot single frame
    elif plot_type == 2:
        animate(frame)
        plt.savefig('plots/frame_' + c.log_name, bbox_inches='tight')
