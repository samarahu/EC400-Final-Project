import gym
from gym import error, spaces, utils
from gym.utils import seeding
import random
import pystk
import logging
import numpy as np

logger = logging.getLogger(__name__)

class PystkEnv(gym.Env):
  metadata = {'render.modes': ['human']}

  def __init__(self):
    super().__init__()

    # continuous action space
    self.action_space = spaces.Box(
        low= np.array([-1,-1,0]), high=self.max_angle
    )
    pass
  def step(self, action):
    pass
  def reset(self):
    pass
  def render(self, mode='human', close=False):
        """ Viewer only supports human mode currently. """
        if close:
            if self.viewer is not None:
                os.kill(self.viewer.pid, signal.SIGKILL)
        else:
            if self.viewer is None:
                self._start_viewer()
  def close(self):
    pass