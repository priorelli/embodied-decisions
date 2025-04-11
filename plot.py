import seaborn as sns
import numpy as np
import utils
import config as c
from plots.video import record_video
from plots.evidence import plot_evidence
from plots.urgency import plot_urgency
from plots.cost import plot_cost
from plots.accuracy import plot_accuracy, plot_correlation
from plots.commitment import plot_commitment
from plots.learning import plot_learning
from plots.likelihood import plot_cue, plot_ext

sns.set_theme(style='darkgrid', font_scale=3.5)


def main():
    width = 6

    # Parse arguments
    options = utils.get_plot_options()

    # Load log
    log = np.load('simulation/log_{}.npz'.format(c.log_name))

    # Choose plot to display
    if options.learning:
        plot_learning(log, width)
    elif options.evidence:
        plot_evidence(log, width)
    elif options.urgency:
        plot_urgency(log, width)
    elif options.commitment:
        plot_commitment(log, width)
    elif options.cost:
        plot_cost(log, width)
    elif options.likelihood:
        plot_cue(log, width)
        # plot_ext(log, width)
    elif options.accuracy:
        logs = [np.load('simulation/log_speed_acc_decide.npz'),
                np.load('simulation/log_speed_acc_inf01.npz'),
                np.load('simulation/log_speed_acc_noinf.npz')]
        plot_accuracy(logs, width)
        # plot_correlation(logs, width)
    else:
        record_video(log, width)


if __name__ == '__main__':
    main()
