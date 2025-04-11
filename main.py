import utils
import config as c
from simulation.manual_control import ManualControl
from simulation.inference import Inference


def main():
    # Parse arguments
    options = utils.get_sim_options()

    # Choose simulation
    if options.manual_control:
        sim = ManualControl()

    else:
        if options.evidence:
            c.log_name = 'evidence'
            c.task = 'evidence'
            c.mod = 'congruent'
            c.t1_pos = [-250, 400]
            c.t2_pos = [250, 400]
            c.n_trials = 300
            c.gain_evidence = 1.0
            c.k_e = 0.0
            c.w_c = 1.1
            c.w_e = 1.0
            c.omega_d = 1.0
            c.eta_d = 0.0
            c.alpha_c = 0.4
            c.alpha_e = 0.0

        elif options.urgency:
            c.log_name = 'urgency'
            c.task = 'urgency'
            c.mod = 'custom'
            c.t1_pos = [-250, 400]
            c.t2_pos = [250, 400]
            c.n_trials = 3
            c.gain_prior = 1.0
            c.gain_evidence = 1.0
            c.k_e = 0.0
            c.w_c = 1.0
            c.w_e = 1.0
            c.omega_d = 1.0
            c.eta_d = 0.0
            c.alpha_c = 0.4
            c.alpha_e = 0.0

        elif options.modulation:
            c.log_name = 'modulation'
            c.task = 'modulation'
            c.mod = 'custom'
            c.t1_pos = [-250, 400]
            c.t2_pos = [250, 400]
            c.n_trials = 2
            c.gain_evidence = 1.0
            c.k_e = 0.0
            c.w_c = 1.0
            c.w_e = 15.0
            c.omega_d = 1.0
            c.eta_d = 0.0
            c.alpha_c = 0.1
            c.alpha_e = 0.5

        elif options.commitment:
            c.log_name = 'commitment'
            c.task = 'commitment'
            c.mod = 'custom'
            c.t1_pos = [-250, 400]
            c.t2_pos = [250, 400]
            c.n_trials = 4
            c.gain_evidence = 1.0
            c.k_e = 0.0
            c.w_c = 1.0
            c.w_e = 1.0
            c.omega_d = 1.0
            c.eta_d = 0.0
            c.alpha_c = 0.4
            c.alpha_e = 0.2
            c.cue_sequence = [0, 0, 1, 0, 0, 0, 0, 0,
                              1, 0, 0, 0, 0, 0, 0]

        elif options.cost:
            c.log_name = 'cost'
            c.task = 'cost'
            c.mod = 'custom'  # 'neutral'
            c.t1_pos = [-150, 400]
            c.t2_pos = [150, 400]
            c.n_trials = 3  # 300
            c.gain_evidence = 3.0
            c.k_e = 0.1
            c.w_c = 1.0
            c.w_e = 1.0
            c.omega_d = 1.0
            c.eta_d = 0.0
            c.alpha_c = 0.4
            c.alpha_e = 0.0
            c.cue_sequence = [1, 0, 1, 0, 1, 0, 0, 0,
                              0, 0, 0, 0, 0, 0, 0]

        elif options.learning:
            c.log_name = 'learning'
            c.task = 'learning'
            c.mod = 'incongruent'
            c.t1_pos = [-250, 400]
            c.t2_pos = [250, 400]
            c.n_trials = 50
            c.gain_evidence = 1.0
            c.k_e = 0.0  # 0.0, 0.05
            c.w_c = 1.0
            c.w_e = 15.0
            c.omega_d = 0.99
            c.eta_d = 0.2
            c.alpha_c = 0.4
            c.alpha_e = 0.4

        elif options.accuracy:
            c.task = 'accuracy'
            c.mod = 'neutral'
            c.t1_pos = [-250, 400]
            c.t2_pos = [250, 400]
            c.n_trials = 500
            c.gain_evidence = 1.0
            c.k_e = 0.15
            c.w_c = 1.0
            c.w_e = 1.0
            c.omega_d = 1.0
            c.eta_d = 0.0
            c.alpha_c = 0.4
            c.alpha_e = 0.3
            c.n_wait = 10
            c.n_steps = c.n_cues * c.n_tau + c.n_tau * c.n_wait

        elif options.likelihood:
            c.log_name = 'likelihood'
            c.task = 'likelihood'
            c.mod = 'congruent'
            c.cue_prob = 0.3
            c.t1_pos = [-250, 400]
            c.t2_pos = [250, 400]
            c.n_trials = 60
            c.gain_prior = 1.0
            c.gain_evidence = 1.0
            c.k_e = 0.0
            c.w_c = 1.0
            c.w_e = 2.0  # 8.0, 1.0
            c.omega_d = 1.0
            c.eta_d = 0.0
            # c.omega_a_ext = 0.99
            # c.eta_a_ext = 0.01
            c.omega_a_cue = 0.98
            c.eta_a_cue = 0.01
            c.alpha_c = 0.4
            c.alpha_e = 0.3
            c.n_wait = 10
            c.n_steps = c.n_cues * c.n_tau + c.n_tau * c.n_wait

        else:
            c.task = 'custom'

        sim = Inference()

    # Run simulation
    sim.run()


if __name__ == '__main__':
    main()
