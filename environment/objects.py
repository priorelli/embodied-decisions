import config as c
import numpy as np
from environment.sprites import Arc, Ball, Wall


# Define objects class
class Objects:
    def __init__(self, batch, space):
        self.batch, self.space = batch, space

        self.t1, self.t2, self.walls = self.setup()
        self.cues = []

    def setup(self):
        t1 = Arc(self.batch, self.space, c.t1_size, (200, 100, 0), c.t1_pos)
        t2 = Arc(self.batch, self.space, c.t2_size, (100, 200, 0), c.t2_pos)

        walls = []
        corners = [(0, 0), (0, c.height), (c.width, c.height),
                   (c.width, 0), (0, 0)]

        for i in range(len(corners) - 1):
            walls.append(Wall(self.space, corners[i], corners[i + 1]))

        return t1, t2, walls

    # Add cue
    def add_cue(self, num):
        r = np.random.randint(c.t1_size)
        theta = np.radians(np.random.randint(360))
        pos = np.array([r * np.cos(theta), r * np.sin(theta)])

        # Change colors
        for cue in self.cues:
            cue.color = (100, 100, 100)
            cue.radius = 10

        if num == 0:
            cue = Ball(self.batch, self.space, 16, (100, 100, 200),
                       pos + c.t1_pos)
        else:
            cue = Ball(self.batch, self.space, 16, (100, 100, 200),
                       pos + c.t2_pos)

        self.cues.append(cue)
