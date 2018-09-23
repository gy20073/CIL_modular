#!/usr/bin/env bash

sudo docker run -it -p 2000-2002:2000-2002 --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=0 carlasim/carla:0.8.2 /bin/bash

# the path to the town changed to: /Game/Carla/Maps/Town01
./CarlaUE4.sh  -benchmark -carla-server -fps=5 -world-port=2000 -carla-no-hud