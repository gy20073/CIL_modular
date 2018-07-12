#!/bin/bash
: <<'COMMENT'
#Terminal4
#pm5
./timeout.py 5000 python2 chauffeur.py drive --driver Human -in Carla -dc 9cam_agent_carla_acquire_rc -cy carla_1 -m 0.2 -p 2009  -nm W5_pm5 -cc /home/muellem/Downloads/carla_chauffeur/drive_interfaces/carla/rcCarla_9Cams_pm5_W5.ini; 
./timeout.py 5000 python2 chauffeur.py drive --driver Human -in Carla -dc 9cam_agent_carla_acquire_rc -cy carla_1 -m 0.2 -p 2009  -nm W9_pm5 -cc /home/muellem/Downloads/carla_chauffeur/drive_interfaces/carla/rcCarla_9Cams_pm5_W9.ini;
./timeout.py 5000 python2 chauffeur.py drive --driver Human -in Carla -dc 9cam_agent_carla_acquire_rc -cy carla_1 -m 0.2 -p 2009  -nm W13_pm5 -cc /home/muellem/Downloads/carla_chauffeur/drive_interfaces/carla/rcCarla_9Cams_pm5_W13.ini
#pp5
./timeout.py 5000 python2 chauffeur.py drive --driver Human -in Carla -dc 9cam_agent_carla_acquire_rc -cy carla_1 -m 0.2 -p 2009  -nm W5_pp5 -cc /home/muellem/Downloads/carla_chauffeur/drive_interfaces/carla/rcCarla_9Cams_pp5_W5.ini; 
./timeout.py 5000 python2 chauffeur.py drive --driver Human -in Carla -dc 9cam_agent_carla_acquire_rc -cy carla_1 -m 0.2 -p 2009  -nm W9_pp5 -cc /home/muellem/Downloads/carla_chauffeur/drive_interfaces/carla/rcCarla_9Cams_pp5_W9.ini;
./timeout.py 5000 python2 chauffeur.py drive --driver Human -in Carla -dc 9cam_agent_carla_acquire_rc -cy carla_1 -m 0.2 -p 2009  -nm W13_pp5 -cc /home/muellem/Downloads/carla_chauffeur/drive_interfaces/carla/rcCarla_9Cams_pp5_W13.ini
#w600
./timeout.py 5000 python2 chauffeur.py drive --driver Human -in Carla -dc 9cam_agent_carla_acquire_rc -cy carla_1 -m 0.2 -p 2009  -nm W5_w600 -cc /home/muellem/Downloads/carla_chauffeur/drive_interfaces/carla/rcCarla_9Cams_w600_W5.ini; 
./timeout.py 5000 python2 chauffeur.py drive --driver Human -in Carla -dc 9cam_agent_carla_acquire_rc -cy carla_1 -m 0.2 -p 2009  -nm W9_w600 -cc /home/muellem/Downloads/carla_chauffeur/drive_interfaces/carla/rcCarla_9Cams_w600_W9.ini;
./timeout.py 5000 python2 chauffeur.py drive --driver Human -in Carla -dc 9cam_agent_carla_acquire_rc -cy carla_1 -m 0.2 -p 2009  -nm W13_w600 -cc /home/muellem/Downloads/carla_chauffeur/drive_interfaces/carla/rcCarla_9Cams_w600_W13.ini
#w700
./timeout.py 5000 python2 chauffeur.py drive --driver Human -in Carla -dc 9cam_agent_carla_acquire_rc -cy carla_1 -m 0.2 -p 2009  -nm W5_w700 -cc /home/muellem/Downloads/carla_chauffeur/drive_interfaces/carla/rcCarla_9Cams_w700_W5.ini; 
./timeout.py 5000 python2 chauffeur.py drive --driver Human -in Carla -dc 9cam_agent_carla_acquire_rc -cy carla_1 -m 0.2 -p 2009  -nm W9_w700 -cc /home/muellem/Downloads/carla_chauffeur/drive_interfaces/carla/rcCarla_9Cams_w700_W9.ini;
./timeout.py 5000 python2 chauffeur.py drive --driver Human -in Carla -dc 9cam_agent_carla_acquire_rc -cy carla_1 -m 0.2 -p 2009  -nm W13_w700 -cc /home/muellem/Downloads/carla_chauffeur/drive_interfaces/carla/rcCarla_9Cams_w700_W13.ini
#w900
./timeout.py 5000 python2 chauffeur.py drive --driver Human -in Carla -dc 9cam_agent_carla_acquire_rc -cy carla_1 -m 0.2 -p 2009  -nm W5_w900 -cc /home/muellem/Downloads/carla_chauffeur/drive_interfaces/carla/rcCarla_9Cams_w900_W5.ini; 
./timeout.py 5000 python2 chauffeur.py drive --driver Human -in Carla -dc 9cam_agent_carla_acquire_rc -cy carla_1 -m 0.2 -p 2009  -nm W9_w900 -cc /home/muellem/Downloads/carla_chauffeur/drive_interfaces/carla/rcCarla_9Cams_w900_W9.ini;
./timeout.py 5000 python2 chauffeur.py drive --driver Human -in Carla -dc 9cam_agent_carla_acquire_rc -cy carla_1 -m 0.2 -p 2009  -nm W13_w900 -cc /home/muellem/Downloads/carla_chauffeur/drive_interfaces/carla/rcCarla_9Cams_w900_W13.ini
#w1000
./timeout.py 5000 python2 chauffeur.py drive --driver Human -in Carla -dc 9cam_agent_carla_acquire_rc -cy carla_1 -m 0.2 -p 2009  -nm W5_w1000 -cc /home/muellem/Downloads/carla_chauffeur/drive_interfaces/carla/rcCarla_9Cams_w1000_W5.ini; 
./timeout.py 5000 python2 chauffeur.py drive --driver Human -in Carla -dc 9cam_agent_carla_acquire_rc -cy carla_1 -m 0.2 -p 2009  -nm W9_w1000 -cc /home/muellem/Downloads/carla_chauffeur/drive_interfaces/carla/rcCarla_9Cams_w1000_W9.ini;
./timeout.py 5000 python2 chauffeur.py drive --driver Human -in Carla -dc 9cam_agent_carla_acquire_rc -cy carla_1 -m 0.2 -p 2009  -nm W13_w1000 -cc /home/muellem/Downloads/carla_chauffeur/drive_interfaces/carla/rcCarla_9Cams_w1000_W13.ini
#w1100
./timeout.py 5000 python2 chauffeur.py drive --driver Human -in Carla -dc 9cam_agent_carla_acquire_rc -cy carla_1 -m 0.2 -p 2009  -nm W5_w1100 -cc /home/muellem/Downloads/carla_chauffeur/drive_interfaces/carla/rcCarla_9Cams_w1100_W5.ini; 
./timeout.py 5000 python2 chauffeur.py drive --driver Human -in Carla -dc 9cam_agent_carla_acquire_rc -cy carla_1 -m 0.2 -p 2009  -nm W9_w1100 -cc /home/muellem/Downloads/carla_chauffeur/drive_interfaces/carla/rcCarla_9Cams_w1100_W9.ini;
./timeout.py 5000 python2 chauffeur.py drive --driver Human -in Carla -dc 9cam_agent_carla_acquire_rc -cy carla_1 -m 0.2 -p 2009  -nm W13_w1100 -cc /home/muellem/Downloads/carla_chauffeur/drive_interfaces/carla/rcCarla_9Cams_w1100_W13.ini
#w1200
./timeout.py 5000 python2 chauffeur.py drive --driver Human -in Carla -dc 9cam_agent_carla_acquire_rc -cy carla_1 -m 0.2 -p 2009  -nm W5_w1200 -cc /home/muellem/Downloads/carla_chauffeur/drive_interfaces/carla/rcCarla_9Cams_w1200_W5.ini; 
./timeout.py 5000 python2 chauffeur.py drive --driver Human -in Carla -dc 9cam_agent_carla_acquire_rc -cy carla_1 -m 0.2 -p 2009  -nm W9_w1200 -cc /home/muellem/Downloads/carla_chauffeur/drive_interfaces/carla/rcCarla_9Cams_w1200_W9.ini;
./timeout.py 5000 python2 chauffeur.py drive --driver Human -in Carla -dc 9cam_agent_carla_acquire_rc -cy carla_1 -m 0.2 -p 2009  -nm W13_w1200 -cc /home/muellem/Downloads/carla_chauffeur/drive_interfaces/carla/rcCarla_9Cams_w1200_W13.ini
#z50
./timeout.py 5000 python2 chauffeur.py drive --driver Human -in Carla -dc 9cam_agent_carla_acquire_rc -cy carla_1 -m 0.2 -p 2009  -nm W5_z50 -cc /home/muellem/Downloads/carla_chauffeur/drive_interfaces/carla/rcCarla_9Cams_z50_W5.ini; 
./timeout.py 5000 python2 chauffeur.py drive --driver Human -in Carla -dc 9cam_agent_carla_acquire_rc -cy carla_1 -m 0.2 -p 2009  -nm W9_z50 -cc /home/muellem/Downloads/carla_chauffeur/drive_interfaces/carla/rcCarla_9Cams_z50_W9.ini;
./timeout.py 5000 python2 chauffeur.py drive --driver Human -in Carla -dc 9cam_agent_carla_acquire_rc -cy carla_1 -m 0.2 -p 2009  -nm W13_z50 -cc /home/muellem/Downloads/carla_chauffeur/drive_interfaces/carla/rcCarla_9Cams_z50_W13.ini
#z150
./timeout.py 5000 python2 chauffeur.py drive --driver Human -in Carla -dc 9cam_agent_carla_acquire_rc -cy carla_1 -m 0.2 -p 2009  -nm W5_z150 -cc /home/muellem/Downloads/carla_chauffeur/drive_interfaces/carla/rcCarla_9Cams_z150_W5.ini; 
./timeout.py 5000 python2 chauffeur.py drive --driver Human -in Carla -dc 9cam_agent_carla_acquire_rc -cy carla_1 -m 0.2 -p 2009  -nm W9_z150 -cc /home/muellem/Downloads/carla_chauffeur/drive_interfaces/carla/rcCarla_9Cams_z150_W9.ini;
./timeout.py 5000 python2 chauffeur.py drive --driver Human -in Carla -dc 9cam_agent_carla_acquire_rc -cy carla_1 -m 0.2 -p 2009  -nm W13_z150 -cc /home/muellem/Downloads/carla_chauffeur/drive_interfaces/carla/rcCarla_9Cams_z150_W13.ini

COMMENT

#Plain
./timeout.py 2500 python2 chauffeur.py drive --driver Human -in Carla -dc 9cam_agent_carla_acquire_rc -cy carla_1 -m 0.2 -p 2009  -nm W5_N -n "Spike" -cc /home/muellem/Downloads/carla_chauffeur/drive_interfaces/carla/rcCarla_9Cams_W5.ini; 
./timeout.py 2500 python2 chauffeur.py drive --driver Human -in Carla -dc 9cam_agent_carla_acquire_rc -cy carla_1 -m 0.2 -p 2009  -nm W9_N -n "Spike" -cc /home/muellem/Downloads/carla_chauffeur/drive_interfaces/carla/rcCarla_9Cams_W9.ini;
./timeout.py 2500 python2 chauffeur.py drive --driver Human -in Carla -dc 9cam_agent_carla_acquire_rc -cy carla_1 -m 0.2 -p 2009  -nm W13_N -n "Spike" -cc /home/muellem/Downloads/carla_chauffeur/drive_interfaces/carla/rcCarla_9Cams_W13.ini

#pm5
./timeout.py 2500 python2 chauffeur.py drive --driver Human -in Carla -dc 9cam_agent_carla_acquire_rc -cy carla_1 -m 0.2 -p 2009  -nm W5_pm5_N -n "Spike" -cc /home/muellem/Downloads/carla_chauffeur/drive_interfaces/carla/rcCarla_9Cams_pm5_W5.ini; 
./timeout.py 2500 python2 chauffeur.py drive --driver Human -in Carla -dc 9cam_agent_carla_acquire_rc -cy carla_1 -m 0.2 -p 2009  -nm W9_pm5_N -n "Spike" -cc /home/muellem/Downloads/carla_chauffeur/drive_interfaces/carla/rcCarla_9Cams_pm5_W9.ini;
./timeout.py 2500 python2 chauffeur.py drive --driver Human -in Carla -dc 9cam_agent_carla_acquire_rc -cy carla_1 -m 0.2 -p 2009  -nm W13_pm5_N -n "Spike" -cc /home/muellem/Downloads/carla_chauffeur/drive_interfaces/carla/rcCarla_9Cams_pm5_W13.ini
#pp5
./timeout.py 2500 python2 chauffeur.py drive --driver Human -in Carla -dc 9cam_agent_carla_acquire_rc -cy carla_1 -m 0.2 -p 2009  -nm W5_pp5_N -n "Spike" -cc /home/muellem/Downloads/carla_chauffeur/drive_interfaces/carla/rcCarla_9Cams_pp5_W5.ini; 
./timeout.py 2500 python2 chauffeur.py drive --driver Human -in Carla -dc 9cam_agent_carla_acquire_rc -cy carla_1 -m 0.2 -p 2009  -nm W9_pp5_N -n "Spike" -cc /home/muellem/Downloads/carla_chauffeur/drive_interfaces/carla/rcCarla_9Cams_pp5_W9.ini;
./timeout.py 2500 python2 chauffeur.py drive --driver Human -in Carla -dc 9cam_agent_carla_acquire_rc -cy carla_1 -m 0.2 -p 2009  -nm W13_pp5_N -n "Spike" -cc /home/muellem/Downloads/carla_chauffeur/drive_interfaces/carla/rcCarla_9Cams_pp5_W13.ini

#z50
./timeout.py 2500 python2 chauffeur.py drive --driver Human -in Carla -dc 9cam_agent_carla_acquire_rc -cy carla_1 -m 0.2 -p 2009  -nm W5_z50_N -n "Spike" -cc /home/muellem/Downloads/carla_chauffeur/drive_interfaces/carla/rcCarla_9Cams_z50_W5.ini; 
./timeout.py 2500 python2 chauffeur.py drive --driver Human -in Carla -dc 9cam_agent_carla_acquire_rc -cy carla_1 -m 0.2 -p 2009  -nm W9_z50_N -n "Spike" -cc /home/muellem/Downloads/carla_chauffeur/drive_interfaces/carla/rcCarla_9Cams_z50_W9.ini;
./timeout.py 2500 python2 chauffeur.py drive --driver Human -in Carla -dc 9cam_agent_carla_acquire_rc -cy carla_1 -m 0.2 -p 2009  -nm W13_z50_N -n "Spike" -cc /home/muellem/Downloads/carla_chauffeur/drive_interfaces/carla/rcCarla_9Cams_z50_W13.ini
#z150
./timeout.py 2500 python2 chauffeur.py drive --driver Human -in Carla -dc 9cam_agent_carla_acquire_rc -cy carla_1 -m 0.2 -p 2009  -nm W5_z150_N -n "Spike" -cc /home/muellem/Downloads/carla_chauffeur/drive_interfaces/carla/rcCarla_9Cams_z150_W5.ini; 
./timeout.py 2500 python2 chauffeur.py drive --driver Human -in Carla -dc 9cam_agent_carla_acquire_rc -cy carla_1 -m 0.2 -p 2009  -nm W9_z150_N -n "Spike" -cc /home/muellem/Downloads/carla_chauffeur/drive_interfaces/carla/rcCarla_9Cams_z150_W9.ini;
./timeout.py 2500 python2 chauffeur.py drive --driver Human -in Carla -dc 9cam_agent_carla_acquire_rc -cy carla_1 -m 0.2 -p 2009  -nm W13_z150_N -n "Spike" -cc /home/muellem/Downloads/carla_chauffeur/drive_interfaces/carla/rcCarla_9Cams_z150_W13.ini
