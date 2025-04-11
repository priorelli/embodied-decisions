import numpy as np
import utils
import config as c


# Define log class
class Log:
    def __init__(self):
        # Initialize logs
        self.angles = np.zeros((c.n_trials, c.n_steps, c.n_joints))
        self.est_angles = np.zeros_like(self.angles)

        self.pos = np.zeros((c.n_trials, c.n_steps, c.n_joints + 1, 2))
        self.est_pos = np.zeros((c.n_trials, c.n_steps, 2))

        self.t1_pos = np.zeros((c.n_trials, c.n_steps, 2))
        self.t2_pos = np.zeros_like(self.t1_pos)

        self.counts = np.zeros((c.n_trials, 2))
        self.states = np.zeros((c.n_trials, c.n_steps, 2))
        self.cues = np.zeros((c.n_trials, c.n_steps // c.n_tau, 2))
        self.causes = np.zeros((c.n_trials, c.n_steps, 3))

        self.a_cue = np.zeros((c.n_trials, 2, 2))
        self.A_cue = np.zeros_like(self.a_cue)
        self.a_ext = np.zeros((c.n_trials, 3, 2))
        self.A_ext = np.zeros_like(self.a_ext)

        self.vel = np.zeros((c.n_trials, c.n_steps, 2))
        self.est_vel = np.zeros_like(self.vel)
        self.F_m = np.zeros((c.n_trials, c.n_steps, 3, 2))
        self.L_ext = np.zeros((c.n_trials, c.n_steps, 3))

        self.success = np.zeros(c.n_trials)
        self.react = np.zeros_like(self.success)

    # Track logs for each iteration
    def track(self, trial, step, brain, body, objects, cue):
        self.angles[trial, step] = body.get_angles()
        est_angles = brain.prop.predict().detach().numpy()
        self.est_angles[trial, step] = utils.denormalize(
            est_angles, c.norm_polar)

        self.pos[trial, step, 1:] = body.get_pos()
        est_pos = brain.vis.predict()[0].detach().numpy()
        self.est_pos[trial, step] = utils.denormalize(est_pos, c.norm_cart)

        self.t1_pos[trial, step] = objects.t1.get_pos()
        self.t2_pos[trial, step] = objects.t2.get_pos()

        self.counts[trial] = brain.discrete.d
        self.states[trial, step] = brain.discrete.prior
        if c.n_tau < step < c.n_steps - c.n_tau * (c.n_wait - 1):
            self.cues[trial, step // c.n_tau] = cue
        self.causes[trial, step] = brain.ext.v

        self.a_cue[trial] = brain.discrete.a_cue
        self.A_cue[trial] = brain.discrete.A_cue
        self.a_ext[trial] = brain.discrete.a_ext
        self.A_ext[trial] = brain.discrete.A_ext

        self.vel[trial, step] = body.joints[-1].get_vel()
        self.est_vel[trial, step] = utils.denormalize(
            brain.ext.x[1].detach().numpy(), c.norm_cart)
        self.F_m[trial, step] = utils.denormalize(
            brain.ext.Preds_x.detach().numpy(), c.norm_cart)
        self.L_ext[trial, step] = brain.discrete.L_ext

    # Save log to file
    def save_log(self):
        np.savez_compressed('simulation/log_{}'.format(c.log_name),
                            angles=self.angles,
                            est_angles=self.est_angles,
                            pos=self.pos,
                            est_pos=self.est_pos,
                            t1_pos=self.t1_pos,
                            t2_pos=self.t2_pos,
                            counts=self.counts,
                            states=self.states,
                            cues=self.cues,
                            causes=self.causes,
                            a_cue=self.a_cue,
                            A_cue=self.A_cue,
                            a_ext=self.a_ext,
                            A_ext=self.A_ext,
                            vel=self.vel,
                            est_vel=self.est_vel,
                            F_m=self.F_m,
                            L_ext=self.L_ext,
                            success=self.success,
                            react=self.react)
