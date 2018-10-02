#!/usr/bin/env bash

# ssh -N -L 2000:localhost:2000 jormungandr.ist.berkeley.edu & ssh -N -L 2001:localhost:2001 jormungandr.ist.berkeley.edu & ssh -N -L 2002:localhost:2002 jormungandr.ist.berkeley.edu &

python2 CIL_modular/drive_interfaces/carla/carla_client/manual_control.py \
    --autopilot
