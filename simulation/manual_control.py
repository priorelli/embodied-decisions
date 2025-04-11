from pyglet.window import key
import utils
import config as c
from environment.window import Window
from environment.log import Log


# Define manual control class
class ManualControl(Window):
    def __init__(self):
        super().__init__()

        # Initialize error tracking
        self.log = Log()

    def update(self, dt):
        dt = 1 / c.fps

        # Get action from user
        action = self.get_pressed()

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

        # Stop simulation
        self.step += 1
        if self.step == c.n_steps:
            self.stop()

    # Get action from user input
    def get_pressed(self):
        return [(key.LEFT in self.keys) - (key.RIGHT in self.keys),
                (key.UP in self.keys) - (key.DOWN in self.keys),
                (key.Z in self.keys) - (key.X in self.keys)]
