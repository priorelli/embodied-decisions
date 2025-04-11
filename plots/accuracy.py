import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import config as c


def plot_accuracy(logs, width):
    conds = ('Decide only', 'Decide then act',
             'Motor inference', 'No motor inference')
    states = logs[0]['states']
    cues = logs[0]['cues'], logs[1]['cues'], logs[2]['cues']
    successes = logs[1]['success'], logs[2]['success']
    reacts = logs[1]['react'], logs[2]['react']
    colors = 'g', 'r'

    fig, axs = plt.subplots(1, figsize=(20, 16))
    axs.set_ylim(-1, 51)
    axs.set_xlim(90, 750)
    axs.set_ylabel('Mean error (%)')
    axs.set_xlabel('React time')

    # Plot decision only
    speed, accuracy = [], []
    for threshold in range(68, 100, 1):
        times, errors = [], []
        for cue, trial in zip(cues[0], states):
            n = int(np.count_nonzero(cue[1:16, 0]) < c.n_cues / 2)

            time1 = np.where(trial[:, 0] > threshold / 100)[0]
            time2 = np.where(trial[:, 1] > threshold / 100)[0]

            if time1.any() or time2.any():
                time, error = check_react(n, time1, time2)
                times.append(time)
                errors.append(error)

        speed.append(np.average(times))
        accuracy.append(np.average(errors) * 100)

    popt, pcov = curve_fit(func, speed, accuracy, p0=(100000, 0.04))
    xx = np.linspace(np.min(speed), 750, 500)
    yy = func(xx, *popt)
    axs.scatter(speed, accuracy, s=100)
    axs.plot(xx, yy, lw=width, label=conds[0], ls='--')

    # Plot serial process
    xx = np.linspace(np.min(speed), 750, 500)
    yy = func(xx, *popt)
    axs.plot(xx + 400, yy, lw=width, label=conds[1], ls='--')

    # Plot other conditions
    for cond, success, react, color in zip(
            conds[2:], successes, reacts, colors):
        speed, accuracy, std = [], [], []
        for trial in range(0, len(success), 100):
            times_notnan = react[trial:trial + 100][np.argwhere(
                ~np.isnan(react[trial:trial + 100])).T[0]]

            success_notnan = success[trial:trial + 100][np.argwhere(
                ~np.isnan(success[trial:trial + 100])).T[0]]

            if times_notnan.any() and success_notnan.any():
                speed.append(np.average(times_notnan))
                accuracy.append(100 - np.average(success_notnan) * 100)
                std.append(np.std(times_notnan))

        popt, pcov = curve_fit(func, speed, accuracy, p0=(100000, 0.04))
        xx = np.linspace(np.min(speed), 750, 500)
        yy = func(xx, *popt)
        axs.scatter(speed, accuracy, s=100, color=color)
        axs.plot(xx, yy, lw=width, label=cond, color=color)

    axs.legend()

    plt.tight_layout()
    fig.savefig('plots/accuracy_' + c.log_name, bbox_inches='tight')


def plot_tradeoff(logs, width):
    conds = ('Decide only', 'Decide then act',
             'Motor inference', 'No motor inference')
    states = logs[0]['states']
    cues = logs[0]['cues'], logs[1]['cues'], logs[2]['cues']
    successes = logs[1]['success'], logs[2]['success']
    reacts = logs[1]['react'], logs[2]['react']
    pos_logs = logs[1]['pos'], logs[2]['pos']
    t1_pos_logs = logs[1]['t1_pos'], logs[2]['t1_pos']
    t2_pos_logs = logs[1]['t2_pos'], logs[2]['t2_pos']

    fig, axs = plt.subplots(1, figsize=(20, 16))
    axs.set_ylim(-1, 51)
    axs.set_xlim(0, 750)
    axs.set_ylabel('Mean error (%)')
    axs.set_xlabel('t')

    # Plot decision only
    speed, accuracy = [], []
    for threshold in range(61, 100, 1):
        times, errors = [], []
        for cue, trial in zip(cues[0], states):
            n = int(np.count_nonzero(cue[1:16, 0]) < c.n_cues / 2)

            time1 = np.where(trial[:, 0] > threshold / 100)[0]
            time2 = np.where(trial[:, 1] > threshold / 100)[0]

            if time1.any() or time2.any():
                time, error = check_react(n, time1, time2)
                times.append(time)
                errors.append(error)

        speed.append(np.average(times))
        accuracy.append(np.average(errors) * 100)

    axs.plot(speed, accuracy, lw=width, label=conds[0], ls='--')

    # Plot serial process
    speed_serial = np.array(speed) + 210

    axs.plot(speed_serial, accuracy, lw=width, label=conds[1], ls='--')

    # Plot other conditions
    for cond, success, react, cue, pos, t1_pos, t2_pos in zip(
            conds[2:], successes, reacts, cues[1:],
            pos_logs, t1_pos_logs, t2_pos_logs):
        times, errors = [], []

        for trial in range(0, len(success), 100):
            n = int(np.count_nonzero(cue[trial][1:16, 0]) < c.n_cues / 2)

            time1 = np.where(norm(pos[trial, :, -1] - t1_pos[n], axis=1) <
                             c.t1_size * c.reach_dist)[0]
            time2 = np.where(norm(pos[trial, :, -1] - t2_pos[n], axis=1) <
                             c.t1_size * c.reach_dist)[0]

            if time1.any() or time2.any():
                time, error = check_react(n, time1, time2)
                times.append(time)
                errors.append(error)

        speed = [np.average(time) for time in times]
        accuracy = [np.average(error) * 100 for error in errors]

        axs.plot(speed, accuracy, lw=width, label=cond)

    axs.legend()

    plt.tight_layout()
    fig.savefig('plots/tradeoff_' + c.log_name, bbox_inches='tight')


def plot_correlation(logs, width):
    fig, axs = plt.subplots(1, figsize=(20, 16))
    axs.set_ylim(-0.01, 1.0)
    axs.set_xlabel(r'$\alpha_h$')
    axs.set_ylabel('Corr. coeff.')

    axs.plot(np.linspace(0.3+0.2/9*5, 0.5, 6),
             correlate(logs[1]['states'], logs[1]['vel_pred'],
                       logs[1]['cues']),
             lw=width, label='Motor inference', color='g')

    axs.plot(np.linspace(0.3+0.2/9*5, 0.5, 6),
             correlate(logs[2]['states'], logs[2]['vel_pred'],
                       logs[2]['cues']),
             lw=width, label='No motor inference', color='r')

    axs.legend()
    plt.tight_layout()
    fig.savefig('plots/correlation_' + c.log_name, bbox_inches='tight')


def correlate(states, vel_pred, cues):
    coeffs = []
    for trial in range(200, len(states), 50):
        n = int(np.count_nonzero(cues[trial][1:16, 0]) < c.n_cues / 2)

        c1 = states[trial:trial + 50, :300, n].flatten()
        c2 = np.linalg.norm(vel_pred[trial:trial + 50, :300],
                            axis=2).flatten()
        coeffs.append(np.corrcoef(c1, c2)[0, 1])

    return coeffs


def check_react(n, time1, time2):
    if time1.any() and time2.any():
        g = np.argmin([time1[0], time2[0]])
        time = np.min([time1[0], time2[0]])
    elif not time2.any():
        g, time = 0, time1[0]
    else:
        g, time = 1, time2[0]
    error = 0 if n == g else 1

    return time, error


def func(x, b, c):
    return 0 + b * np.exp(-c * x)
