import gym
from gym import error, spaces, utils
import pystk
import logging
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
import dense_transforms
import os, subprocess, time, signal

logger = logging.getLogger(__name__)

RESCUE_TIMEOUT = 30
TRACK_OFFSET = 15

class PystkEnv(gym.Env):
  metadata = {'render.modes': ['human']}

  def __init__(self, screen_width=128, screen_height=96):
    super().__init__()
    self.action = pystk.Action()
    # continuous action space
    self.action_space = spaces.Box(
        low= np.array([-1,0]).astype(np.float32), high=np.array([+1,+1].astype(np.float32))
    )

    assert PystkEnv._singleton is None, "Cannot create more than one pytux object"
    PystkEnv._singleton = self
    self.config = pystk.GraphicsConfig.hd()
    self.config.screen_width = screen_width
    self.config.screen_height = screen_height
    pystk.init(self.config)
    self.k = None

  @staticmethod
  def _point_on_track(distance, track, offset=0.0):
    """
    Get a point at `distance` down the `track`. Optionally applies an offset after the track segment if found.
    Returns a 3d coordinate
    """
    node_idx = np.searchsorted(track.path_distance[..., 1],
                                distance % track.path_distance[-1, 1]) % len(track.path_nodes)
    d = track.path_distance[node_idx]
    x = track.path_nodes[node_idx]
    t = (distance + offset - d[0]) / (d[1] - d[0])
    return x[1] * t + x[0] * (1 - t)

  def steer(self, s):
    self.action.steer = s

  @staticmethod
  def _to_image(x, proj, view):
        p = proj @ view @ np.array(list(x) + [1])
        return np.clip(np.array([p[0] / p[-1], -p[1] / p[-1]]), -1, 1)

  def step(self, track, controller, planner=None, max_frames=1000, verbose=False, data_callback=None):
    """
    Play a level (track) for a single round.
    :param track: Name of the track
    :param controller: low-level controller, see controller.py
    :param planner: high-level planner, see planner.py
    :param max_frames: Maximum number of frames to play for
    :param verbose: Should we use matplotlib to show the agent drive?
    :param data_callback: Rollout calls data_callback(time_step, image, 2d_aim_point) every step, used to store the
                            data
    :return: Number of steps played
    """
    if self.k is not None and self.k.config.track == track:
        self.k.restart()
        self.k.step()
    else:
        if self.k is not None:
            self.k.stop()
            del self.k
        config = pystk.RaceConfig(num_kart=1, laps=1,track=track)
        config.players[0].controller = pystk.PlayerConfig.Controller.PLAYER_CONTROL

        self.k = pystk.Race(config)
        self.k.start()
        self.k.step()

    state = pystk.WorldState()
    track = pystk.Track()

    last_rescue = 0

    if verbose:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 1)

    for t in range(max_frames):
        
        state.update()
        track.update()

        kart = state.players[0].kart

        if np.isclose(kart.overall_distance / track.length, 1.0, atol=2e-3):
            
            if verbose:
                print("Finished at t=%d" % t)
            break

        proj = np.array(state.players[0].camera.projection).T
        view = np.array(state.players[0].camera.view).T

        aim_point_world = self._point_on_track(kart.distance_down_track+TRACK_OFFSET, track)
        aim_point_image = self._to_image(aim_point_world, proj, view)
        if data_callback is not None:
            data_callback(t, np.array(self.k.render_data[0].image), aim_point_image)

        if planner:
            image = np.array(self.k.render_data[0].image)
            aim_point_image = planner(TF.to_tensor(image)[None]).squeeze(0).cpu().detach().numpy()

        current_vel = np.linalg.norm(kart.velocity)
        action = controller(aim_point_image, current_vel, kart.location)

        if current_vel < 1.0 and t - last_rescue > RESCUE_TIMEOUT:
            last_rescue = t
            action.rescue = True

        if verbose:
            ax.clear()
            ax.imshow(self.k.render_data[0].image)
            WH2 = np.array([self.config.screen_width, self.config.screen_height]) / 2
            ax.add_artist(plt.Circle(WH2*(1+self._to_image(kart.location, proj, view)), 2, ec='b', fill=False, lw=1.5))
            ax.add_artist(plt.Circle(WH2*(1+self._to_image(aim_point_world, proj, view)), 2, ec='r', fill=False, lw=1.5))
            if planner:
                ap = self._point_on_track(kart.distance_down_track + TRACK_OFFSET, track)
                ax.add_artist(plt.Circle(WH2*(1+aim_point_image), 2, ec='g', fill=False, lw=1.5))
            plt.pause(1e-3)

        self.k.step(action)
        t += 1
    return t, kart.overall_distance / track.length

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
    """
    Call this function, once you're done with PyTux
    """
    if self.k is not None:
        self.k.stop()
        del self.k
    pystk.clean()