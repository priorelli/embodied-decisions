# Window
width = 1000
height = 1000
off_x = 0
off_y = -200

debug = 0
fps = 60
speed = 10
phys_steps = 100

# Environment
dt = 0.3

t1_pos = [-250, 400]
t1_size = 60
t1_vel = 0
t1_dir = None

t2_pos = [250, 400]
t2_size = 60
t2_vel = 0
t2_dir = None

# Brain (continuous)
eta_x_int = [35, 150, 0]
x_int_start = None

pi_eta_x_int = 0.0
pi_x_int = 1.0
p_x_int = 10.0
pi_prop = 1.0

pi_eta_x_ext = 0.5
pi_x_ext = pi_x_int
p_x_ext = p_x_int
pi_vis = 1.0

lambda_ext = 0.5
lr_ext = 1.0
lambda_int = 0.0
lr_int = 1.0
lr_a = 1.0

# Brain (discrete)
n_tau = 30
n_policy = 1

gain_prior = 1.0
gain_evidence = 1.0
k_e = 0.0
w_c = 1.1
w_e = 1.0

omega_d = 1.0
eta_d = 0.0
omega_a_ext = 1.0
eta_a_ext = 0.0
omega_a_cue = 1.0
eta_a_cue = 0.0
alpha_c = 0.4
alpha_e = 0.2

# Simulation
n_trials = 1
n_cues = 15
n_wait = 5
n_steps = n_cues * n_tau + n_tau * n_wait
n_orders = 2
log_name = ''

task = 'custom'
mod = 'custom'  # congruent, incongruent, neutral, custom, reversed
cue_sequence = [0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0]
cue_prob = 0.5
reach_dist = 3.0

# Body
start = eta_x_int
lengths = [250, 150, 50]

joints = {}
joints['trunk'] = {'link': None, 'angle': start[0],
                   'size': (lengths[0], 60)}
joints['shoulder'] = {'link': 'trunk', 'angle': start[1],
                      'size': (lengths[1], 50)}
joints['elbow'] = {'link': 'shoulder', 'angle': start[2],
                   'size': (lengths[2], 40)}
n_joints = len(joints)

norm_polar = [-180.0, 180.0]
norm_cart = [-sum(lengths), sum(lengths)]
