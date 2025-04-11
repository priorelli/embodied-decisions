import numpy as np
import itertools
import utils
import config as c


# Define discrete class
class Discrete:
    def __init__(self):
        # Likelihood matrix
        pos_states = ['T1', 'T2']

        self.states = [pos_states]
        self.n_states = len(pos_states)
        self.state_to_idx, self.idx_to_state = self.get_idx()

        self.a_cue = np.array([[2., 1.8], [1.8, 2.]])
        self.A_cue = utils.norm_counts(self.a_cue)
        self.coincidence_cue = np.zeros_like(self.a_cue)

        self.a_ext = np.array([[.6, 0.], [0., .6], [.4, .4]])
        self.A_ext = utils.norm_counts(self.a_ext)
        self.coincidence_ext = np.zeros_like(self.a_ext)

        if c.task != 'likelihood':
            self.A_cue = self.get_A_cue()
            self.A_ext = self.get_A_ext()

        # Transition matrix
        self.actions = ['STAY']
        self.n_actions = len(self.actions)
        self.B = self.get_B()

        # Preference matrix
        self.C = np.ones(self.n_states)
        self.C /= self.n_states

        # Prior matrix
        self.d = np.full(self.n_states, 0.5)
        self.prior = utils.softmax(self.d)

        # Initialize policies and habit matrix
        self.policies = self.construct_policies()
        self.E = np.zeros(len(self.policies))

        # Compute entropy
        self.H_A = self.entropy()

        # Initialize observations and log evidences
        self.o_ext = np.array([0.0, 0.0, 1.0])
        self.L_ext = np.zeros(len(self.A_ext))

        if c.debug:
            print('prior: {:.2f}, {:.2f}'.format(*self.prior))

    # Get coincidences for likelihoods
    def get_coincidence(self, cue, qs):
        self.coincidence_cue += np.outer(cue, qs)
        self.coincidence_ext += np.outer(self.o_ext[:], qs)

    # Learn likelihood matrix
    def learn_likelihood(self):
        self.a_cue = (c.omega_a_cue * self.a_cue +
                      c.eta_a_cue * self.coincidence_cue)
        self.coincidence_cue = np.zeros_like(self.a_cue)
        self.A_cue = utils.norm_counts(self.a_cue)

        self.a_ext = (c.omega_a_ext * self.a_ext +
                      c.eta_a_ext * self.coincidence_ext)
        self.coincidence_ext = np.zeros_like(self.a_ext)
        self.A_ext = utils.norm_counts(self.a_ext)

        if c.debug:
            print('\n', self.A_cue)

    # Learn prior
    def learn_prior(self):
        self.d = c.omega_d * self.d + c.eta_d * self.prior
        self.prior = utils.softmax(self.d)

    # Get state-index mappings
    def get_idx(self):
        state_to_idx = {}
        idx_to_state = {}
        c = 0
        for i in self.states[0]:
            state_to_idx[i] = c
            idx_to_state[c] = i
            c += 1

        return state_to_idx, idx_to_state

    # Get cue likelihood matrix
    def get_A_cue(self):
        A_cue = np.zeros((2, self.n_states))

        for state, idx in self.state_to_idx.items():
            if state == 'T1':
                A_cue[0, idx] = 1 - c.alpha_c
                A_cue[1, idx] = c.alpha_c
            elif state == 'T2':
                A_cue[1, idx] = 1 - c.alpha_c
                A_cue[0, idx] = c.alpha_c

        return A_cue

    # Get extrinsic likelihood matrix
    def get_A_ext(self):
        A_ext = np.zeros((3, self.n_states))

        for state, idx in self.state_to_idx.items():
            if state == 'T1':
                A_ext[0, idx] = 1 - c.alpha_e
                A_ext[2, idx] = c.alpha_e
            if state == 'T2':
                A_ext[1, idx] = 1 - c.alpha_e
                A_ext[2, idx] = c.alpha_e

        return A_ext

    # Get transition matrix
    def get_B(self):
        B = np.zeros((self.n_states, self.n_states, self.n_actions))

        for state, idx in self.state_to_idx.items():
            for action_id, action_label in enumerate(self.actions):
                next_label = state

                next_idx = self.state_to_idx[next_label]
                B[next_idx, idx, action_id] = 1.0

        return B

    # Get all policies
    def construct_policies(self):
        x = [self.n_actions] * c.n_policy

        policies = list(itertools.product(*[list(range(i)) for i in x]))
        for pol_i in range(len(policies)):
            policies[pol_i] = np.array(policies[pol_i]).reshape(c.n_policy, 1)

        return policies

    # Compute likelihood entropy
    def entropy(self):
        H_A = - (self.A_ext * utils.log_stable(self.A_ext)).sum(axis=0)

        return H_A

    # Infer current states
    def infer_states(self, o_t, r_t):
        # Get expected state from cue
        qs_t_cue = self.A_cue.T.dot(o_t)

        # Get expected state from observation
        qs_t_ext = self.A_ext.T.dot(r_t)

        log_prior = utils.log_stable(self.prior)
        log_post_cue = utils.log_stable(qs_t_cue)
        log_post_ext = utils.log_stable(qs_t_ext) * c.k_e

        qs = utils.softmax(log_prior + log_post_cue + log_post_ext, c.w_c)

        return qs

    # Compute expected states
    def get_expected_states(self, qs_current, action):
        qs_u = self.B[:, :, action].dot(qs_current)

        return qs_u

    # Compute expected observations
    def get_expected_obs(self, qs_u):
        qo_u = self.A_ext.dot(qs_u)

        return qo_u

    # Compute KL divergence
    def kl_divergence(self, qs_u):
        return (utils.log_stable(qs_u) - utils.log_stable(self.C)).dot(qs_u)

    # Compute expected free energy
    def compute_G(self, qs_current):
        G = np.zeros(len(self.policies))

        for policy_id, policy in enumerate(self.policies):
            qs_pi_t = 0

            for t in range(policy.shape[0]):
                action = policy[t, 0]
                qs_prev = qs_current if t == 0 else qs_pi_t

                qs_pi_t = self.get_expected_states(qs_prev, action)

                kld = self.kl_divergence(qs_pi_t)

                G[policy_id] += kld

        return G

    # Compute action posterior
    def compute_prob_actions(self, Q_pi):
        P_u = np.zeros(self.n_actions)

        for policy_id, policy in enumerate(self.policies):
            P_u[int(policy[0, 0])] += Q_pi[policy_id]

        P_u = utils.norm_dist(P_u)

        return P_u

    # Get next states
    def get_qs_next(self, P_u, qs_t):
        qs_next = np.zeros(self.n_states)

        for action_idx, prob in enumerate(P_u):
            qs_next += prob * self.B[:, :, action_idx].dot(qs_t)

        return qs_next

    # Run discrete step
    def step(self, cue):
        # Perform BMC
        self.o_ext[:] = utils.bmc(self.o_ext, self.L_ext, c.w_e,
                                  c.gain_prior, c.gain_evidence)

        # Infer current state
        qs_current = self.infer_states(cue, self.o_ext[:])

        # Compute coincidences
        self.get_coincidence(cue, qs_current)

        # Compute expected free energy
        G = self.compute_G(qs_current)

        # Marginalize P(u|pi)
        Q_pi = utils.softmax(self.E - G)

        # Compute action posterior
        P_u = self.compute_prob_actions(Q_pi)

        # Compute next observations
        self.prior = self.get_qs_next(P_u, qs_current)

        self.L_ext[:] = 0
        self.o_ext[:] = utils.bmc(self.get_expected_obs(self.prior),
                                  self.L_ext, c.w_e,
                                  c.gain_prior, c.gain_evidence)

        if c.debug:
            np.set_printoptions(precision=2, suppress=True)
            print('\ncue:', cue)
            print('L:', self.L_ext)
            print('qs:', qs_current)
            print('qs_next:', self.prior)
            print('v:', self.o_ext)
            input()
