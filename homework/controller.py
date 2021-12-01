# Samarah Uriarte
# Collaborators: Prithika Ganesh, Brian Mahabir, Robert Ling
# 10/21/21

import pystk
import numpy as np

def get_vector_from_this_to_that(me, obj, normalize=True):
    """
    Expects numpy arrays as input
    """
    vector = obj - me

    if normalize:
        return vector / np.linalg.norm(vector)

    return vector

def to_numpy(location):
    """
    Don't care about location[1], which is the height
    """
    return np.float32([location[0], location[2]])

def control(aim_point, current_vel):
    """
    Set the Action for the low-level controller
    :param aim_point: Aim point, in screen coordinate frame [-1..1]
    :param current_vel: Current velocity of the kart
    :return: a pystk.Action (set acceleration, brake, steer, drift)
    """
    action = pystk.Action()
    state = pystk.WorldState()
    kart = pystk.Kart

    """
    Your code here
    Hint: Use action.acceleration (0..1) to change the velocity. Try targeting a target_velocity (e.g. 20).
    Hint: Use action.brake to True/False to brake (optionally)
    Hint: Use action.steer to turn the kart towards the aim_point, clip the steer angle to -1..1
    Hint: You may want to use action.drift=True for wide turns (it will turn faster)
    """

    target_vel = 22
    action.nitro = True

    # Acceleration adjustments
    if current_vel < target_vel / 3:
        action.acceleration = 1
    elif current_vel < target_vel / 2:
        action.acceleration = 0.91
    elif current_vel < target_vel:
        action.acceleration = 0.85
    else:
        action.acceleration = -0.85

    # To flip the y-axis and find the angle of the aim point
    aim_point[1] = aim_point[1] * -1
    angle = np.arctan(aim_point[1] / aim_point[0])

    # If the aim points goes too far to the left or right, then the steer angle must compensate
    if aim_point[0] < -0.256:
        action.steer = -1
        action.drift = True
    elif aim_point[0] > 0.256:
        action.steer = 1
        action.drift = True
    else:
        action.steer = angle
        action.drift = False
    
    #print(kart.location)

    # pos_me = to_numpy(pystk.Kart.location)

    # # Look for the closest pick up (bomb, gift, etc..).
    # closest_item = to_numpy(state.items[0].location)
    # closest_item_distance = np.inf

    state.update()

    for item in state.items:
        print(item)

    # from https://github.com/philkr/pystk/blob/253ef48b4d68bd6cc7e9db6d4877a1f01a80bdcd/examples/hockey_gamestate.py#L10:
    
    #     item_norm = np.linalg.norm(
    #             get_vector_from_this_to_that(pos_me, to_numpy(item.location), normalize=False))

    #     if item_norm < closest_item_distance:
    #         closest_item = to_numpy(item.location)
    #         closest_item_distance = item_norm
        
    #     print(closest_item)

    # Get some directional vectors.
        # front_me = to_numpy(state.karts[0].front)
        # ori_me = get_vector_from_this_to_that(pos_me, front_me)
        # ori_to_ai = get_vector_from_this_to_that(pos_me, pos_ai)
        # ori_to_item = get_vector_from_this_to_that(pos_me, closest_item)
    
    # if abs(1 - np.dot(ori_me, ori_to_item)) > 1e-4:
    #     uis[0].current_action.steer = np.sign(np.cross(ori_to_item, ori_me))
    # , state

    return action

if __name__ == '__main__':
    from utils import PyTux
    from argparse import ArgumentParser

    def test_controller(args):
        import numpy as np
        pytux = PyTux()
        for t in args.track:
            steps, how_far = pytux.rollout(t, control, max_frames=1000, verbose=args.verbose)
            print(steps, how_far)
        pytux.close()


    parser = ArgumentParser()
    parser.add_argument('track', nargs='+')
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()
    test_controller(args)
