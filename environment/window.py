import numpy as np
import pyglet
import pymunk
import utils
import config as c
from environment.body import Body
from environment.objects import Objects


class Window(pyglet.window.Window):
    def __init__(self):
        super().__init__(c.width, c.height, 'Embodied decisions', vsync=False)
        # Start physics engine
        self.space = pymunk.Space()
        self.space.gravity = 0, 0

        self.keys = set()
        self.batch = pyglet.graphics.Batch()
        self.fps_display = pyglet.window.FPSDisplay(self)

        # Initialize body
        self.body = Body(self.batch, self.space)

        # Initialize objects
        self.objects = Objects(self.batch, self.space)

        self.update_sprites()

        # Initialize brain
        self.brain = None

        # Initialize simulation variables
        self.step, self.trial = 0, 0
        self.reached, self.react = -1, 0
        self.target = 0

        # Set background
        pyglet.gl.glClearColor(1, 1, 1, 1)

    def init_sim(self):
        self.space = pymunk.Space()
        self.space.gravity = 0, 0
        self.batch = pyglet.graphics.Batch()

        self.body = Body(self.batch, self.space)
        self.objects = Objects(self.batch, self.space)
        self.update_sprites()

    def on_key_press(self, sym, mod):
        self.keys.add(sym)

    def on_key_release(self, sym, mod):
        self.keys.remove(sym)

    def on_draw(self):
        self.clear()
        self.batch.draw()
        self.fps_display.draw()

    # Update function to override
    def update(self, dt):
        pass

    # Run simulation with custom update function
    def run(self):
        if c.fps == 0:
            pyglet.clock.schedule(self.update)
        else:
            pyglet.clock.schedule_interval(self.update, 1 / c.fps)
        pyglet.app.run()

    # Stop simulation
    def stop(self):
        pyglet.app.exit()
        self.close()

    # Update sprites rotation and position
    def update_sprites(self):
        sprites = [self.objects.t1, self.objects.t2]
        for sprite in self.body.links + self.body.joints + sprites:
            sprite.position = sprite.body.position
            sprite.rotation = -np.degrees(sprite.body.angle)

    # Get proprioceptive observation
    def get_prop_obs(self):
        return utils.normalize(self.body.get_angles(), c.norm_polar)

    # Get visual observation
    def get_visual_obs(self):
        pos = self.body.get_pos()[-1]
        vel = self.body.joints[-1].get_vel()

        return utils.normalize([pos, vel], c.norm_cart)

    # Get cue based on task
    def get_cue(self, step):
        if c.mod == 'custom':
            num = c.cue_sequence[step // c.n_tau]
        else:
            if c.mod == 'congruent':
                cue_prob = min(0.8 + (0.2 / 8) * (step // c.n_tau), 1.0)
            elif c.mod == 'incongruent':
                cue_prob = min(0.0 + (1.0 / 10) * (step // c.n_tau), 1.0)
            elif c.mod == 'reversed':
                cue_prob = max(1.0 - (1.0 / 8) * (step // c.n_tau), 0.0)
            elif c.mod == 'neutral':
                cue_prob = min(0.5 + (0.5 / 8) * (step // c.n_tau), 1.0)
            elif c.mod == 'fixed':
                cue_prob = np.clip(np.random.normal(c.cue_prob, scale=0.1),
                                   0, 1)
            else:
                cue_prob = 0.5 if step // c.n_tau <= 10 else 1.0
                # cue_prob = np.random.random()

            num = np.random.choice(2, p=[cue_prob, 1 - cue_prob])
            if self.target == 1 and c.task == 'likelihood':
                num = np.random.choice(2, p=[1 - cue_prob, cue_prob])

        self.objects.add_cue(num)

        return np.array([1 - num, num])

    # Check if task is successful
    def task_done(self, log):
        if self.reached == -1:
            log.react[self.trial] = np.nan
            log.success[self.trial] = np.nan
        else:
            log.react[self.trial] = self.react

            n = int(np.count_nonzero(log.cues[self.trial, 1:16, 0])
                    < c.n_cues / 2)
            log.success[self.trial] = 1 if n == self.reached else 0

        self.reached = -1

    # Track average speed
    def track_speed(self, step):
        if self.reached == -1:
            dists = [np.linalg.norm(self.body.get_pos()[-1] - c.t1_pos)
                     < c.t1_size * c.reach_dist,
                     np.linalg.norm(self.body.get_pos()[-1] - c.t2_pos)
                     < c.t1_size * c.reach_dist]

            if any(dists):
                self.reached = 0 if dists[0] else 1
                self.react = step
