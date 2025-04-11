import numpy as np
import config as c
from environment.sprites import Origin, Joint, Link


# Define body class
class Body:
    def __init__(self, batch, space):
        # Initialize body parameters
        ids = {joint: j for j, joint in enumerate(c.joints)}

        self.idxs = {}
        self.start = np.zeros(c.n_joints)
        self.size = np.zeros((c.n_joints, 2))

        for joint in c.joints:
            self.start[ids[joint]] = c.joints[joint]['angle']
            self.size[ids[joint]] = c.joints[joint]['size']

            if c.joints[joint]['link']:
                self.idxs[ids[joint]] = ids[c.joints[joint]['link']]
            else:
                self.idxs[ids[joint]] = -1

        self.joints, self.links = self.setup(batch, space)

    def setup(self, batch, space):
        # Initialize origin
        origin = Origin(space)

        # Add joints and links
        joints = [Joint(batch, space, 30, origin.body, (0, 0), (0, 0))]
        links = []
        for j in range(c.n_joints):
            idx = self.idxs[j] + 1
            angle = np.sum(self.start[:j + 1])

            link = Link(batch, space, self.size[j], joints[idx].body, angle)
            joint = Joint(batch, space, self.size[j, 1] / 2,
                          link.body, (link.width, 0), link.get_end())

            links.append(link)
            joints.append(joint)

        return joints, links

    # Set velocity
    def update(self, action):
        for j in range(c.n_joints):
            self.links[j].motor.rate = -action[j]

    # Get relative angles
    def get_angles(self):
        angles = np.zeros(c.n_joints)

        for l, link in enumerate(self.links):
            angles[l] = -link.rotation - np.sum(angles[:l])

        return angles

    # Get relative rates
    def get_rates(self):
        vel = np.zeros(c.n_joints)

        for j, link in enumerate(self.links):
            vel[j] = link.body.angular_velocity - np.sum(vel[:j])

        return np.degrees(vel)

    # Get relative torques
    def get_torques(self):
        return [link.body.torque for link in self.links]

    # Get link positions
    def get_pos(self):
        return np.array([link.get_end() for link in self.links])
