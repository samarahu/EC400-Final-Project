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

def control_steer(aim_point, current_vel):
    action = pystk.Action()
    target_vel = 21.9579
    action.drift = False

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
    
    return action

def control(aim_point, current_vel, location):
    """
    Set the Action for the low-level controller
    :param aim_point: Aim point, in screen coordinate frame [-1..1]
    :param current_vel: Current velocity of the kart
    :return: a pystk.Action (set acceleration, brake, steer, drift)
    """
    action = pystk.Action()
    state = pystk.WorldState()

    """
    Your code here
    Hint: Use action.acceleration (0..1) to change the velocity. Try targeting a target_velocity (e.g. 20).
    Hint: Use action.brake to True/False to brake (optionally)
    Hint: Use action.steer to turn the kart towards the aim_point, clip the steer angle to -1..1
    Hint: You may want to use action.drift=True for wide turns (it will turn faster)
    """

    action.nitro = True
    target_distance = 9.3

    state.update()
    pos_me = to_numpy(location)

    # Look for the closest pick up (bomb, gift, etc..).
    closest_item = to_numpy(state.items[0].location)
    closest_item_distance = np.inf

    for item in state.items:
    # from https://github.com/philkr/pystk/blob/253ef48b4d68bd6cc7e9db6d4877a1f01a80bdcd/examples/hockey_gamestate.py#L10:
    
        # current item to kart
        item_norm = np.linalg.norm(
                get_vector_from_this_to_that(pos_me, to_numpy(item.location), normalize=False))

        # updates closest item
        if item_norm < closest_item_distance:
            closest_item_location = to_numpy(item.location)
            closest_item = item.type
            closest_item_distance = item_norm

        #print(closest_item_location[0] - location[0])

        # checks for closer items within a relevant range
        if item_norm < target_distance and closest_item_location[0] - location[0] > 5:
            # banana
            if closest_item == 1:
                # steers away from this item
                if closest_item_location[1] - location[2] >= 0:
                    action.steer = -0.5
                if closest_item_location[1] - location[2] < 0:
                    action.steer = 0.5
            # nitro small or big
            elif closest_item == 3 or closest_item == 2:
                # steers towards item
                if closest_item_location[1] - location[2] < 0:
                    action.steer = -0.5
                elif closest_item_location[1] - location[2] > 0:
                    action.steer = 0.5
                else:
                    action.steer = 0
        else:
            action = control_steer(aim_point, current_vel)

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
