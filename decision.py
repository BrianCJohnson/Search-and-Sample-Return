import numpy as np


# This is where you can build a decision tree for determining throttle, brake and steer 
# commands based on the output of the perception_step() function
def decision_step(Rover):

    # Implement conditionals to decide what to do given perception data
    # Here you're all set up with some basic functionality but you'll need to
    # improve on this decision tree to do a good job of navigating autonomously!

    # Example:
    # Check if we have vision data to make decisions with
    if Rover.nav_angles is not None:
        # Check for Rover.mode status
        if Rover.mode == 'forward': 
            # Check the extent of navigable terrain
            if len(Rover.nav_angles) >= Rover.stop_forward:  
                # If mode is forward, navigable terrain looks good 
                # and velocity is below max, then throttle
                safe = np.argwhere(np.abs(Rover.nav_angles) < 5.0)
                safe = np.argwhere(Rover.nav_dists[safe] < 10.0)
                stopping_dist = np.mean(Rover.nav_dists[safe])
                if Rover.vel < Rover.max_vel and Rover.vel < 0.5 * stopping_dist:
                    # Set throttle value to throttle setting
                    Rover.throttle = Rover.throttle_set
                else: # Else coast
                    Rover.throttle = 0
                Rover.brake = 0
                # Set steering to average angle clipped to the range +/- 15
                close_indices = np.argwhere(Rover.nav_dists < 40.0)
                if len(close_indices) >10:
                    close_angles = Rover.nav_angles[close_indices]
                else:
                    close_angles = Rover.nav_angles
                #print('num nav_dists:{:5d}, '.format(len(Rover.nav_dists))
                #print('max dist:{:5d}, '.format(np.argmax(Rover.nav_dists)), end = '')
                #steer = np.clip(np.mean(Rover.nav_angles * 180/np.pi) + 2.0, -15, 15) # plus is to the left, minus is to the right
                if Rover.vel > 0.0:
                    max_turn = min(45.0, 15.0 / abs(Rover.vel)) # at a vel of 1, a turn of 15 was OK, 10 would be better? no more than 45 though
                else:
                    max_turn = 45.0
                mean_close = np.mean(close_angles) * 180/np.pi
                steer =  np.clip(mean_close + 4, -max_turn, max_turn)
                #steer = np.clip(np.mean(close_angles * 180/np.pi) + 2.0, -15, 15) # plus is to the left, minus is to the right
                print('mode: {}, mean_close:{:5.2f}, max_turn:{:5.2f}, steer:{:5.2f}, '.format(Rover.mode, mean_close, max_turn, steer), end = '')
                new_w = 0.2
                Rover.steer = (1.0 - new_w) * Rover.steer + new_w * steer # low pass filter on steering, new_w : weight given to new input
            # If there's a lack of navigable terrain pixels then go to 'stop' mode
            elif len(Rover.nav_angles) < Rover.stop_forward:
                    # Set mode to "stop" and hit the brakes!
                    Rover.throttle = 0
                    # Set brake to stored brake value
                    Rover.brake = Rover.brake_set
                    Rover.steer = 0
                    Rover.mode = 'stop'

        # If we're already in "stop" mode then make different decisions
        elif Rover.mode == 'stop':
            # If we're in stop mode but still moving keep braking
            if Rover.vel > 0.2:
                Rover.throttle = 0
                Rover.brake = Rover.brake_set
                Rover.steer = 0
            # If we're not moving (vel < 0.2) then do something else
            elif Rover.vel <= 0.2:
                # Now we're stopped and we have vision data to see if there's a path forward
                if len(Rover.nav_angles) < Rover.go_forward:
                    Rover.throttle = 0
                    # Release the brake to allow turning
                    Rover.brake = 0
                    # Turn range is +/- 15 degrees, when stopped the next line will induce 4-wheel turning
                    Rover.steer = -30 # Could be more clever here about which way to turn
                # If we're stopped but see sufficient navigable terrain in front then go!
                if len(Rover.nav_angles) >= Rover.go_forward:
                    # Set throttle back to stored value
                    Rover.throttle = Rover.throttle_set
                    # Release the brake
                    Rover.brake = 0
                    # Set steer to mean angle
                    Rover.steer = np.clip(np.mean(Rover.nav_angles * 180/np.pi), -30, 30)
                    Rover.mode = 'forward'
    # Just to make the rover do something 
    # even if no modifications have been made to the code
    else:
        Rover.throttle = Rover.throttle_set
        Rover.steer = 0
        Rover.brake = 0
    
    # see if a rock sample is near
    #if Rover.rock_angles is not None and len(Rover.rock_angles) > 0 and np.mean(Rover.rock_angles) > 0.0:
    if Rover.rock_angles is not None and len(Rover.rock_angles) > 0:
        # go to rocks on the left since mean is > 0
        #print('rock angles:',Rover.rock_angles, end = '')
        if Rover.vel > 0.0:
            max_turn = min(45.0, 15.0 / abs(Rover.vel)) # at a vel of 1, a turn of 15 was OK, 10 would be better? no more than 45 though
        else:
            max_turn = 45.0
        steer =  np.clip((0.8 * 180/np.pi * np.mean(Rover.rock_angles)), -max_turn, max_turn) # turn part way toward it
        new_w = 0.5
        Rover.steer =  (1.0 - new_w) * Rover.steer + new_w * steer

        # want velocity proportional to distance
        rock_dist = np.amin(Rover.rock_dists)
        if rock_dist == 0.0:
            Rover.brake = 10.0
            des_vel = 0.0 # just for print statement
            vel_err = 0.0 # just for print statement
        else:
            des_vel = 0.05 * rock_dist
            vel_err = Rover.vel - des_vel
            vel_hys = 0.0
            if Rover.vel < (des_vel - vel_hys):
                Rover.throttle = min(Rover.throttle_set, 0.1 * (des_vel - Rover.vel))
                Rover.brake = 0.0
            elif Rover.vel > (des_vel + vel_hys):
                Rover.throttle = 0.0
                Rover.brake = 0.05 * (Rover.vel - des_vel)
            else:
                # close enough, coast
                Rover.throttle = 0.0
                Rover.brake = 0.0
        #print()
        #print('Rover.rock_dists',Rover.rock_dists)
        print('GO TO ROCK !!!, steer:{:5.2f}, dist:{:5.2f}, des_vel:{:5.2f}, vel:{:5.2f}, vel_err:{:5.2f}, '.format(Rover.steer, rock_dist, des_vel, Rover.vel, vel_err), end = '')

    # If in a state where want to pickup a rock send pickup command
    if Rover.near_sample and Rover.vel == 0 and not Rover.picking_up:
        Rover.send_pickup = True
    
    return Rover

