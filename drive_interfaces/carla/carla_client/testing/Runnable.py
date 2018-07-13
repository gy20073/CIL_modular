#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 10 19:12:37 2017

@author: german
"""
from __future__ import print_function
import time
import math
import numpy as np
import os, shutil, cv2

from carla import CARLA
from carla import  SceneDescription,EpisodeStart,EpisodeReady,Control,Measurements

class Runnable(object, ):
    def __init__(self, **kwargs):
        pass
    
    def run_step(self, data):
        pass

    def compute_distance(self, curr, prev, target):
        # no history info
        if(prev[0] == -1 and prev[1] == -1):
            distance = math.sqrt((curr[0] - target[0])*(curr[0] - target[0]) + (curr[1] - target[1])*(curr[1] - target[1]))
        else:
            # distance point to segment
            v1 = [target[0]-curr[0], target[1]-curr[1]]
            v2 = [prev[0]-curr[0], prev[1]-curr[1]]
            
            w1 = v1[0]*v2[0] + v1[1]*v2[1]
            w2 = v2[0]*v2[0] + v2[1]*v2[1]
            t_hat = w1 / (w2 + 1e-4)
            t_start = min(max(t_hat, 0.0), 1.0)
            
            s = [0, 0]
            s[0] = curr[0] + t_start * (prev[0] - curr[0])
            s[1] = curr[1] + t_start * (prev[1] - curr[1])
            distance = math.sqrt((s[0] - target[0])*(s[0] - target[0]) + (s[1] - target[1])*(s[1] - target[1]))

        
        return distance

    def run_until(self, carla, time_out, target):
        

        
        curr_x = -1
        curr_y = -1
        prev_x = -1
        prev_y = -1
        measurements= carla.getMeasurements()
        carla.sendCommand(Control())
        t0 = measurements['GameTime']
        t1=t0
        success = False
        step = 0
        accum_lane_intersect = 0.0
        accum_sidewalk_intersect = 0.0
        distance = 100000
        measurement_vec=[]
        while((t1-t0) < (time_out*1000) and not success):
            measurements = carla.getMeasurements()
            control = self.run_step(measurements,target)
            print ('STEER ',control.steer,'GAS ',control.throttle,'Brake ',control.brake)
            carla.sendCommand(control)


            # meassure distance to target

            prev_x = curr_x
            prev_y = curr_y
            curr_x = measurements['PlayerMeasurements'].transform.location.x
            curr_y = measurements['PlayerMeasurements'].transform.location.y


            measurement_vec.append(measurements['PlayerMeasurements'])

            t1 = measurements['GameTime']
            print (t1-t0)
            
            # accumulate layout related signal
            # accum_lane_intersect += reward.road_intersect
            #accum_sidewalk_intersect += reward.sidewalk_intersect
            step += 1

            distance = self.compute_distance([curr_x, curr_y], [prev_x, prev_y], [target.location.x, target.location.y])
            # debug 
            print('[d=%f] c_x = %f, c_y = %f ---> t_x = %f, t_y = %f' % (float(distance), curr_x, curr_y, target.location.x, target.location.y))
            # TODO: print human readable locations Yang
            def world2image(curr_x, curr_y):
                rotation = np.array([curr_x, curr_y])
                #worldoffset = np.array([544.000000,-10748.000000])
                worldoffset = np.array([0, 0])
                mapoffset = np.array([-1643.022,-1643.022])
                rotation = (rotation + worldoffset - mapoffset) / 16.43
                # since indexing is the reverse order
                rotation = rotation[::-1]
                return rotation.astype(np.int32)
            debug_path = "./temp/carla_0_debug.png"
            if not os.path.exists(debug_path):
                shutil.copy("./drive_interfaces/carla/carla_client/carla/planner/carla_0.png", debug_path)
            img = cv2.imread(debug_path)
            cur = world2image(curr_x, curr_y)
            tar = world2image(target.location.x, target.location.y)
            print("current location", cur, "final location", tar)
            img[cur[0]-3:cur[0]+3, cur[1]-3:cur[1]+3, 0] = 0
            img[cur[0] - 3:cur[0] + 3, cur[1] - 3:cur[1] + 3, 1] = 0
            img[cur[0] - 3:cur[0] + 3, cur[1] - 3:cur[1] + 3, 2] = 255
            img[tar[0]-3:tar[0]+3, tar[1]-3:tar[1]+3, 0] = 255
            img[tar[0] - 3:tar[0] + 3, tar[1] - 3:tar[1] + 3, 1] = 0
            img[tar[0] - 3:tar[0] + 3, tar[1] - 3:tar[1] + 3, 2] = 0
            cv2.imwrite(debug_path, img)
            # end of TODO



            if(distance < 200.0):
                success = True

            # Stop If Game Colides
            #if  reward.collision_gen > 0 or reward.collision_ped >0 or  reward.collision_car >0:
            #    success = False
            #    break


        if(success):
            return (1, measurement_vec, float(t1-t0)/1000.0,distance)
        else:
            return (0, measurement_vec, time_out,distance)
