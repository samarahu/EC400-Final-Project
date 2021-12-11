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

def control(aim_point, current_vel, kart):
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

    # action.nitro = True
    # target_distance = 9.3

    # state.update()

    # kartBackForth = kart.location[0]
    # kartHorizontal = kart.location[2]

    # pos_me = to_numpy(kart.location)
    
    # # Look for the closest pick up (bomb, gift, etc..).
    # closest_item = to_numpy(state.items[0].location)
    # closest_item_distance = np.inf

    # for item in state.items:
    # # from https://github.com/philkr/pystk/blob/253ef48b4d68bd6cc7e9db6d4877a1f01a80bdcd/examples/hockey_gamestate.py#L10:
    
    #     # current item to kart
    #     item_norm = np.linalg.norm(
    #             get_vector_from_this_to_that(pos_me, to_numpy(item.location), normalize=False))

    #     # updates closest item
    #     if item_norm < closest_item_distance:
    #         closest_item_location = to_numpy(item.location)
    #         closest_item = item.type
    #         closest_item_distance = item_norm

    #     # Get some directional vectors.
    #     front_me = to_numpy(kart.front)
    #     ori_me = get_vector_from_this_to_that(front_me, pos_me)
    #     ori_to_item = get_vector_from_this_to_that(pos_me, closest_item_location)

    #     # print(closest_item)
    #     #print(abs(1 - np.dot(ori_me, ori_to_item)))


    #     if abs(1 - np.dot(ori_me, ori_to_item)) > 1e-4: # if the dot product between orientation of kart and items are close to 1, then the kart is steering towards the item

    #         if np.sign(np.cross(ori_to_item, ori_me)) > 0:
    #             action.steer = 0.5
    #         else:
    #             action.steer = -0.5

        #print(closest_item_location[0] - location[0])
        # # print(closest_item_location[0] - location[0])
        # print(get_vector_from_this_to_that(pos_me, to_numpy(item.location)))

        # # checks for closer items within a relevant range
        # # 0 = x (back and forth)
        # # 1 = y
        # # 2 = z (horizontal)
        # # closest_item_location[0] - location[0]
        # if closest_item_location[0] - location[0] < 5 and closest_item_location[0] - location[0] > -5:
        #     print(closest_item)
        #     # banana
        #     if closest_item == 1:
        #         print("away from banana")
        #         #print(closest_item_location[1] - location[2])
        #         # steers away from this item
        #         # closest_item_location[1] = z-value, location[2] = z-value
        #         if closest_item_location[1] > location[2]:
        #             action.steer = -0.3
        #             print("steer left")
        #             return action
        #         if closest_item_location[1] < location[2]:
        #             action.steer = 0.3
        #             print("steer right")
        #             return action
        #     # nitro small or big
        #     elif closest_item == 3 or closest_item == 2:
        #         print("towards nitro")
        #         # steers towards item
        #         print(closest_item_location[1] - location[2])
        #         if closest_item_location[1] < location[2]:
        #             action.steer = -0.3
        #             print("steer left")
        #             return action
        #         elif closest_item_location[1] > location[2]:
        #             action.steer = 0.3
        #             print("steer right")
        #             return action
        #         else:
        #             action.steer = 0
        #             print("straight")
        #             return action

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
