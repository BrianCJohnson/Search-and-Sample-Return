import numpy as np
import cv2

# Identify pixels above the threshold
# Threshold of RGB > 160 does a nice job of identifying ground pixels only
def color_thresh(img, rgb_thresh):
    # Create an array of zeros same xy size as img, but single channel
    color_select = np.zeros_like(img[:,:,0])
    # Require that each pixel be above all three threshold values in RGB
    # above_thresh will now contain a boolean array with "True"
    # where threshold was met
    #above_thresh = (img[:,:,0] > rgb_thresh[0]) \
    #            & (img[:,:,1] > rgb_thresh[1]) \
    #            & (img[:,:,2] > rgb_thresh[2])
    in_thresh = (img[:,:,0] >= rgb_thresh[0][0]) & (img[:,:,0] <= rgb_thresh[1][0]) \
                & (img[:,:,1] >= rgb_thresh[0][1]) & (img[:,:,1] <= rgb_thresh[1][1]) \
                & (img[:,:,2] >= rgb_thresh[0][2]) & (img[:,:,2] <= rgb_thresh[1][2])
    # Index the array of zeros with the boolean array and set to 1
    #color_select[above_thresh] = 1
    color_select[in_thresh] = 1
    # Return the binary image
    return color_select


# Define a function to convert from image coords to rover coords
def rover_coords(binary_img):
    # Identify nonzero pixels
    ypos, xpos = binary_img.nonzero()
    # Calculate pixel positions with reference to the rover position being at the 
    # center bottom of the image.  
    x_pixel = -(ypos - binary_img.shape[0]).astype(np.float)
    y_pixel = -(xpos - binary_img.shape[1]/2 ).astype(np.float)
    return x_pixel, y_pixel


# Define a function to convert to radial coords in rover space
def to_polar_coords(x_pixel, y_pixel):
    # Convert (x_pixel, y_pixel) to (distance, angle) 
    # in polar coordinates in rover space
    # Calculate distance to each pixel
    x = x_pixel.astype(np.float)
    y = y_pixel.astype(np.float)
    #dist = np.sqrt(1.0*x_pixel**2 + 1.0*y_pixel**2)
    dist = np.sqrt(x**2 + y**2)
    # Calculate angle away from vertical for each pixel
    #angles = np.arctan2(y_pixel, x_pixel)
    angles = np.arctan2(y, x)
    return dist, angles


# Define a function to map rover space pixels to world space
def rotate_pix(xpix, ypix, yaw):
    # Convert yaw to radians
    yaw_rad = yaw * np.pi / 180
    xpix_rotated = (xpix * np.cos(yaw_rad)) - (ypix * np.sin(yaw_rad))
                            
    ypix_rotated = (xpix * np.sin(yaw_rad)) + (ypix * np.cos(yaw_rad))
    # Return the result  
    return xpix_rotated, ypix_rotated


def translate_pix(xpix_rot, ypix_rot, xpos, ypos, scale): 
    # Apply a scaling and a translation
    xpix_translated = (xpix_rot / scale) + xpos
    ypix_translated = (ypix_rot / scale) + ypos
    # Return the result  
    return xpix_translated, ypix_translated


# Define a function to apply rotation and translation (and clipping)
# Once you define the two functions above this function should work
def pix_to_world(xpix, ypix, xpos, ypos, yaw, world_size, scale):
    # Apply rotation
    xpix_rot, ypix_rot = rotate_pix(xpix, ypix, yaw)
    # Apply translation
    xpix_tran, ypix_tran = translate_pix(xpix_rot, ypix_rot, xpos, ypos, scale)
    # Perform rotation, translation and clipping all at once
    x_pix_world = np.clip(np.int_(xpix_tran), 0, world_size - 1)
    y_pix_world = np.clip(np.int_(ypix_tran), 0, world_size - 1)
    # Return the result
    return x_pix_world, y_pix_world


# Define a function to perform a perspective transform
def perspect_transform(img, src, dst):
           
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]))# keep same size as input image
    mask = cv2.warpPerspective(np.ones_like(img[:, :, 0]), M, (img.shape[1], img.shape[0]))# keep same size as input image    
    
    return warped, mask


# Apply the above functions in succession and update the Rover state accordingly
def perception_step(Rover):
    # Perform perception steps to update Rover()
    # TODO: 
    # NOTE: camera image is coming to you in Rover.img
    # 1) Define source and destination points for perspective transform
    img = Rover.img

    #find offset due to pitch and roll
    yaw = Rover.yaw
    pitch = Rover.pitch
    if pitch > 180: pitch -= 360
    roll = Rover.roll
    if roll >180: roll -= 360
    xpos = Rover.pos[0]
    ypos = Rover.pos[1]
    img_h = img.shape[0]
    img_w = img.shape[1]

    # create new image to compensate for pitch
    new_img = np.zeros_like(img)
    pixels_per_degree = 0.5 * img_w/90.0 # image width corresponds to a feild of view of 90 degrees
    po = -int(round(pitch * pixels_per_degree)) # pixel offset,  pitch is in +/- degrees
    print('po:{:4d}. '.format(po), end = '')
    if po > 0: # positive value shifts toward beginning, end becomes zeros
        new_img[:-po] = img[po:]
    elif po < 0: # negative value shifts toward end, beginning becomes zeros
        new_img[-po:] = img[:po]
    else:
        new_img = img
    #thresh = ((120, 110, 100), (255, 255, 255)) # low and high rgb thresholds for navagable, blue
    #threshed = color_thresh(warped, thresh)
    #ypix, xpix = threshed.nonzero()
    #rotate_pix(xpix, ypix, roll):


    dst_size = 2
    bot_offset = 6
    a = img_h-bot_offset
    b = img_w/2
    src = np.float32([[14,140], [301, 140], [200, 96], [118, 96]])
    dst = np.float32([[b - dst_size, a], 
                     [b + dst_size, a], 
                     [b + dst_size, a - 2*dst_size], 
                     [b - dst_size, a - 2*dst_size]])
    #print('perception_step')
    #print(src)
    #print(dst)

    # 2) Apply perspective transform
    warped, mask = perspect_transform(new_img, src, dst)

    # 3) Apply color threshold to identify navigable terrain/obstacles/rock samples
    # 4) Update Rover.vision_image (this will be displayed on left side of screen)
    # 5) Convert map image pixel values to rover-centric coords
    # 6) Convert rover-centric pixel values to world coordinates
    # 7) Update Rover worldmap (to be displayed on right side of screen)
        # Example: Rover.worldmap[obstacle_y_world, obstacle_x_world, 0] += 1
        #          Rover.worldmap[rock_y_world, rock_x_world, 1] += 1
        #          Rover.worldmap[navigable_y_world, navigable_x_world, 2] += 1
    world_size = Rover.worldmap.shape[0]
    scale = 2 * dst_size

    roll_limit = 0.5
    pitch_limit = 0.5


    thresh = ((0, 0, 0), (50, 40, 30)) # low and high rgb thresholds for non-navagable, red
    threshed = color_thresh(warped, thresh)
    Rover.vision_image[:,:,0] = 255 * threshed * mask
    xpix, ypix = rover_coords(threshed * mask)
    if abs(pitch) < pitch_limit and abs(roll) < roll_limit:
        x_world, y_world = pix_to_world(xpix, ypix, xpos, ypos, yaw, world_size, scale)
        #Rover.worldmap[y_world, x_world, 0] = np.clip(Rover.worldmap[y_world, x_world, 0] +1, 0, 255)
        #Rover.worldmap[y_world, x_world, 1] = np.clip(Rover.worldmap[y_world, x_world, 1] +1, 0, 255)
        #Rover.worldmap[y_world, x_world, 2] = np.clip(Rover.worldmap[y_world, x_world, 2] -1, 0, 255)
    
    thresh = ((140, 120, 0), (255, 255, 70)) # low and high rgb thresholds for rocks, green
    threshed = color_thresh(warped, thresh)
    Rover.vision_image[:,:,1] = 255 * threshed
    xpix, ypix = rover_coords(threshed)
    if abs(pitch) < pitch_limit and abs(roll) < roll_limit and threshed.any():
        x_world, y_world = pix_to_world(xpix, ypix, xpos, ypos, yaw, world_size, scale)
        rock_dists, rock_angles = to_polar_coords(xpix, ypix)
        Rover.rock_angles = rock_angles
        Rover.rock_dists = rock_dists
        rock_index = np.argmin(rock_dists)
        rock_x = x_world[rock_index]
        rock_y = y_world[rock_index]
        rock_pos = [rock_x, rock_y]
        Rover.new_rock_pos(rock_pos)
        #Rover.worldmap[y_world, x_world, 0] = np.clip(Rover.worldmap[y_world, x_world, 0] -1, 0, 255)
        Rover.worldmap[rock_y, rock_x, 1] = np.clip(Rover.worldmap[rock_y, rock_x, 1] +10, 0, 255)
        #Rover.worldmap[y_world, x_world, 2] = np.clip(Rover.worldmap[y_world, x_world, 2] +1, 0, 255)
    else:
        x_world, y_world = pix_to_world(xpix, ypix, xpos, ypos, yaw, world_size, scale)
        rock_dists, rock_angles = to_polar_coords(xpix, ypix)
        Rover.rock_angles = rock_angles
        Rover.rock_dists = rock_dists

    thresh = ((150, 130, 110), (255, 255, 255)) # low and high rgb thresholds for navagable, blue
    threshed = color_thresh(warped, thresh)
    if abs(pitch) < pitch_limit and abs(roll) < roll_limit:
        Rover.vision_image[:,:,2] = 255 * threshed
        xpix, ypix = rover_coords(threshed)
        # limit to only close pixels so the world map isn't corrupted
        good_points = np.argwhere(xpix**2 + ypix**2 < 10**2)
        #print('good points:{:4d}, '.format(len(good_points)), end = '')
        x_world, y_world = pix_to_world(xpix[good_points], ypix[good_points], xpos, ypos, yaw, world_size, scale)
        #Rover.worldmap[y_world, x_world, 0] = np.clip(Rover.worldmap[y_world, x_world, 0] -1, 0, 255)
        #Rover.worldmap[y_world, x_world, 1] = np.clip(Rover.worldmap[y_world, x_world, 1] +2, 0, 255)
        #Rover.worldmap[y_world, x_world, 2] = np.clip(Rover.worldmap[y_world, x_world, 2] +2, 0, 255)
        Rover.worldmap[y_world, x_world, 2] = 255
        Rover.vision_image[:,:,2] = 255 * threshed
    else:
        Rover.vision_image[:,:,2] = 0

    #xpix, ypix = rover_coords((1-threshed) * mask)
    #if abs(pitch) < pitch_limit and abs(roll) < roll_limit:
    #    x_world, y_world = pix_to_world(xpix, ypix, xpos, ypos, yaw, world_size, scale)
    #    Rover.worldmap[y_world, x_world, 2] = np.clip(Rover.worldmap[y_world, x_world, 2] -1, 0, 255)

    #    Rover.worldmap[:, :, 0] = 255 - Rover.worldmap[:, :, 2]
    #likely_nav = Rover.worldmap[:, :, 2] >=  Rover.worldmap[:, :, 0]
    #map_add = cv2.addWeighted(Rover.worldmap, 0.8, Rover.ground_truth, 0.2, 0)

    # 8) Convert rover-centric pixel positions to polar coordinates
    # Update Rover pixel distances and angles
        # Rover.nav_dists = rover_centric_pixel_distances
        # Rover.nav_angles = rover_centric_angles
    #xpix, ypix =
    #print('pitch:{:6.2f}, roll:{:6.2f}. '.format(pitch, roll), end = '')
    if abs(pitch) < 4*pitch_limit and abs(roll) < 4*roll_limit:
        xpix, ypix = rover_coords(threshed)
        dists, angles = to_polar_coords(xpix, ypix)
        Rover.nav_angles = angles   
        Rover.nav_dists = dists
        #print('Len(angles):{:6d}, mean(angles):{:6.2f}. '.format(len(angles), np.mean(angles)*180/np.pi), end = '')
    else:
        print('pitch/roll too big, pitch:{:5.2f}, roll:{:5.2f}, '.format(abs(pitch), abs(roll)), end = '')
    return Rover