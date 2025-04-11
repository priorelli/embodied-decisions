import numpy as np
import utils
import time
import config as c
from environment.window import Window
from environment.log import Log
from simulation.brain import Brain


# Define inference class
class Inference(Window):
    def __init__(self):
        super().__init__()

        # Initialize brain
        self.brain = Brain()
        self.brain.init_belief(self.body.get_angles(),
                               self.body.get_pos()[-1])
        self.cue = np.zeros(2)

        # Initialize error tracking
        self.log = Log()
        self.time = time.time()

    def update(self, dt):
        dt = 1 / c.fps

        # Get cue
        if ((self.step + 1) % c.n_tau == 0 and
                self.step < c.n_steps - c.n_tau * c.n_wait):
            self.cue = self.get_cue(self.step)

        # Track log
        self.log.track(self.trial, self.step, self.brain,
                       self.body, self.objects, self.cue)

        # Get observations
        O = [self.get_prop_obs(), self.get_visual_obs(), self.cue]

        # Perform free energy step
        action = self.brain.inference_step(O, self.step)

        # Update body
        self.body.update(action)

        # Update physics
        for i in range(c.phys_steps):
            self.space.step(c.speed / (c.fps * c.phys_steps))

        # Move sprites
        self.update_sprites()

        # Print info
        if (self.step + 1) % 100 == 0:
            utils.print_info(self.trial, c.n_trials, self.step,
                             c.n_steps, self.log.success, self.log.react)

        # Track average speed
        self.track_speed(self.step)

        # Stop simulation
        self.step += 1
        if self.step == c.n_steps:
            self.reset_trial()

    def reset_trial(self):
        self.task_done(self.log)

        # Learn prior and likelihood
        self.brain.discrete.learn_prior()
        if c.task == 'likelihood':
            self.brain.discrete.learn_likelihood()

        # Simulation done
        if (self.trial + 1) == c.n_trials:
            utils.print_score(self.log)
            print('\nTime elapsed: {:.2f}s'.format(time.time() - self.time))
            self.log.save_log()
            self.stop()

        # Reset simulation
        else:
            self.step = 0
            self.trial += 1

            if c.task == 'evidence':
                if self.trial == 100:
                    c.mod = 'incongruent'
                elif self.trial == 200:
                    c.mod = 'neutral'

            elif c.task == 'urgency':
                if self.trial == 1:
                    c.w_e = 15.0
                    c.alpha_e = 0.45
                    self.brain.discrete.A_ext = self.brain.discrete.get_A_ext()
                elif self.trial == 2:
                    c.w_e = 15.0
                    c.alpha_e = 0.5
                    self.brain.discrete.A_ext = self.brain.discrete.get_A_ext()

            elif c.task == 'modulation':
                if self.trial == 1:
                    c.alpha_c = 0.49
                    c.alpha_e = 0.2
                    self.brain.discrete.A_cue = self.brain.discrete.get_A_cue()
                    self.brain.discrete.A_ext = self.brain.discrete.get_A_ext()

            elif c.task == 'commitment':
                if self.trial == 1:
                    c.k_e = 0.2
                if self.trial == 2:
                    c.cue_sequence = [1, 1, 1, 1, 0, 0, 0, 0,
                                      0, 0, 0, 0, 0, 0, 0]

                    c.k_e = 0.0
                if self.trial == 3:
                    c.k_e = 0.2

            elif c.task == 'cost':
                if self.trial == 1:
                    c.t1_pos = [-250, 400]
                    c.t2_pos = [250, 400]
                elif self.trial == 2:
                    c.t1_pos = [-450, 400]
                    c.t2_pos = [450, 400]

            elif c.task == 'learning':
                if self.trial == c.n_trials // 5:
                    c.mod = 'reversed'

            elif c.task == 'accuracy':
                if self.trial % 100 == 0:
                    # c.w_e += 0.9
                    c.alpha_e += 0.2 / 4
                    self.brain.discrete.A_ext = self.brain.discrete.get_A_ext()

            elif c.task == 'likelihood':
                self.target = (self.target + 1) % 2
                if self.trial == 30:
                    c.mod = 'incongruent'

            self.init_sim()

            self.brain.init_belief(self.body.get_angles(),
                                   self.body.get_pos()[-1])
            self.cue = np.zeros(2)
