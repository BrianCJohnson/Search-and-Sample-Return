## Project: Search and Sample Return

### [Rubric](https://review.udacity.com/#!/rubrics/916/view) Points
### Here I consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  

You're reading it!

### Notebook Analysis
#### 1. Run the functions provided in the notebook on test images (first with the test data provided, next on data you have recorded). Add/modify functions to allow for color selection of obstacles and rock samples.
I made the following modifications to the test code (other than process_image() which is described below):
1. Added a mask to the output of perspect_transform()

2. Modified color_thresh() to accept a 2D array (2x3) of thresholds to include a minimum and maximum threshold for R, G and B

3. Added examples of Original, Warped, Nav Thresholded and Rock Thresholded images

   ![Orig_warp_threshed_rock](./Orig_warp_threshed_rock.JPG)


#### 2. Populate the `process_image()` function with the appropriate analysis steps to map pixels identifying navigable terrain, obstacles and rock samples into a worldmap.  Run `process_image()` on your test data using the `moviepy` functions provided to create video output of your result.

I made the following modifications to process_image():

1. Identified non-navigable areas

   - Used a lower and upper RGB threshold to identify non-navigable portions of warped image


   - Incremented those locations in the red channel of the world map

2. Identified non-navigable areas

   - Used a lower and upper RGB threshold to identify navigable portions of warped image
   - Incremented those locations in the blue channel of the world map
   - Decremented those locations in the red channel of the world map

3. Identified rocks

   - Used a lower and upper RGB threshold to identify rocks in the warped image


   - Incremented those locations in the red channel of the world map

### Autonomous Navigation and Mapping

#### 1. Fill in the `perception_step()` (at the bottom of the `perception.py` script) and `decision_step()` (in `decision.py`) functions in the autonomous mapping scripts and an explanation is provided in the writeup of how and why these functions were modified as they were.
In the perception_step() I made the following changes:

1. Minimized error of mapping due to pitch and roll variations of the robot with the following
   - Ignored the data if the pitch or roll exceeded a threshold
   - Limited the area of the world map that would be updated to within 20 units of the robot
     This limited errors due to the pitch making it look like the navigable distance was very far away 
2. Added a section of code to calculate rock distances and angles
   - This included adding `Rover.rock_dists` and `Rover.rock_angles` to the Rover object

In the decision_step() I made the following changes:

1. Estimated a new `stopping_dist` based on the average distance to locations between -5 and +5 degrees of forward

2. Used the new `stopping_dist` to determine when to accelerate or coast

3. Calculated a `max_turn` angle based on `Rover.vel`

4. Used `max_turn` to allow robot to turn more sharply when going slowly

5. Created index list `close_angles` of navigable points that were at a distance of less than 10 units away

6. Used `np.mean(nav_angles[close_angles])` to make steering decisions based on what was navigable within 10 units of the robot

7. Applied a simple low pass filter to `Rover.steer` so it wouldn't vary wildly

8. Added a section of code to navigate toward a rock and stop if it found one

   - This included adding `Rover.rock_dists` and `Rover.rock_angles` to the Rover object


   - There is code already in place to pick it up if it stops near a rock

#### 2. Launching in autonomous mode your rover can navigate and map autonomously.    

#### Explain your results and how you might improve them in your writeup.  

##### Results

By running drive_rover.py and launching the simulator in autonomous mode, my rover does a reasonably good job at mapping the environment.

If my rover runs long enough (and doesn't get stuck!), it will map at over 80% of the environment with over 75% fidelity (accuracy) against the ground truth. The fidelity was greatly improved by limiting navigable terrain to locations close to the rover.

It will typically find most rocks, map them on the world map and pick up a few of them.

I ran the simulator at the default resolution of 1024x768 at 'good' quality. With these settings it ran between 8 and 16 FPS.

##### Areas for improvement

###### Compensation of Camera Pitch and Roll

Using the warped image with a fixed transform based on `src` and `dst` to map the navigable areas assumes that the camera does not pitch or roll. When it inevitably does the warped image does not accurately match the ground. I was thinking that it would be good to warp based on the actual dynamic`Rover.pitch` and `Rover.roll` values rather than the fixed `src` and `dst` values. This could effectively compensate for the camera's pitch and roll and produce an accurate warped image and optimize  map fidelity.

###### Wall Following

Wall following can be an effective strategy to insure that a robot maps a simply connected region. If we ignore the rock obstacles in the two open locations this "world" is simply connected and following the right or left hand wall continuously will insure the rover to navigate all regions. As an added bonus all the rock samples are located near the walls. For this to work properly there might have to be an initial "wall-alignment" mode where the rover drove toward the wall and properly aligned itself parallel to the wall.

I tried to have the rover follow the left wall by biasing the steering toward the left by adding a couple of degrees to the `np.mean(nav_angles[close_angles])` . That does bias it to the left but it did not hug the left wall as I had hoped. In order to do that effectively I think it would be more effective to average some number of the left most `nav_angles` to calculate my desired steering direction. This could help optimize the total time to search the "world" and percentage of the "world" mapped.

###### Slow to Steer Sharply 

In some cases it is desirable to make a sharp turn but the rover velocity is too great. It would be good to recognize the desire to make a sharp turn and slow the rover to a speed appropriate for the turn. This could help accurate steering toward a goal or wall following and avoid overshooting or missing a sharp turn and optimize total time to search the "world".

###### Return to start

This was suggested as a potential challenge that I did not implement.

###### Improve Rock Collection

Sometimes the rover is at a bad angle to pick up a rock near the wall due to curvature of the wall or irregularities of the ground by the wall. In these cases it would be ideal if the rover would drive away from the rock sample and wall, then turn around at a point were in line with the rock sample and perpendicular to the wall before approaching the rock sample to pick it up. After it picked up the rock sample it should re-align itself with the wall at that point to continue wall following. This would improve the percentage of rocks successfully collected and help optimize time by insuring that it doesn't get stuck unsuccessfully trying to pick up one rock over and over.

###### Avoid Getting Stuck

In some cases the rover attempted to drive over or under a rock obstacle. Sometimes it was able to do it and some times it got stuck. The obstacles that are on the ground should be recognized in the `nav_angles` and `nav_dists` as locations it should not attempt to drive through. The default implementation just steers to the `np.mean(nav_angles[close_angles])`  which may have a non-navigable area in its center. Ideally the rover would also recognize that some of the obstacles are in its way even though the ground under them looks navigable.

###### Get Un-Stuck

I am sure that no matter how well the implementation avoids getting stuck it will happen once in a while. To handle this case there should be a routine to detect that it is stuck and then try a number of strategies to get unstuck.
