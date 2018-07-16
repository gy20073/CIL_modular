#!/bin/bash
: <<'COMMENT'
#Terminal1
#pm5
./timeout.py 5000 python2 chauffeur.py drive --driver Human -in Carla -dc 9cam_agent_carla_acquire_rc -cy carla_1 -m 0.2 -p 2000  -nm W1_pm5 -cc /home/muellem/Downloads/carla_chauffeur/drive_interfaces/carla/rcCarla_9Cams_pm5_W1.ini; 
./timeout.py 5000 python2 chauffeur.py drive --driver Human -in Carla -dc 9cam_agent_carla_acquire_rc -cy carla_1 -m 0.2 -p 2000  -nm W6_pm5 -cc /home/muellem/Downloads/carla_chauffeur/drive_interfaces/carla/rcCarla_9Cams_pm5_W6.ini;
./timeout.py 5000 python2 chauffeur.py drive --driver Human -in Carla -dc 9cam_agent_carla_acquire_rc -cy carla_1 -m 0.2 -p 2000  -nm W10_pm5 -cc /home/muellem/Downloads/carla_chauffeur/drive_interfaces/carla/rcCarla_9Cams_pm5_W10.ini
#pp5
./timeout.py 5000 python2 chauffeur.py drive --driver Human -in Carla -dc 9cam_agent_carla_acquire_rc -cy carla_1 -m 0.2 -p 2000  -nm W1_pp5 -cc /home/muellem/Downloads/carla_chauffeur/drive_interfaces/carla/rcCarla_9Cams_pp5_W1.ini; 
./timeout.py 5000 python2 chauffeur.py drive --driver Human -in Carla -dc 9cam_agent_carla_acquire_rc -cy carla_1 -m 0.2 -p 2000  -nm W6_pp5 -cc /home/muellem/Downloads/carla_chauffeur/drive_interfaces/carla/rcCarla_9Cams_pp5_W6.ini;
./timeout.py 5000 python2 chauffeur.py drive --driver Human -in Carla -dc 9cam_agent_carla_acquire_rc -cy carla_1 -m 0.2 -p 2000  -nm W10_pp5 -cc /home/muellem/Downloads/carla_chauffeur/drive_interfaces/carla/rcCarla_9Cams_pp5_W10.ini
#w600
./timeout.py 5000 python2 chauffeur.py drive --driver Human -in Carla -dc 9cam_agent_carla_acquire_rc -cy carla_1 -m 0.2 -p 2000  -nm W1_w600 -cc /home/muellem/Downloads/carla_chauffeur/drive_interfaces/carla/rcCarla_9Cams_w600_W1.ini; 
./timeout.py 5000 python2 chauffeur.py drive --driver Human -in Carla -dc 9cam_agent_carla_acquire_rc -cy carla_1 -m 0.2 -p 2000  -nm W6_w600 -cc /home/muellem/Downloads/carla_chauffeur/drive_interfaces/carla/rcCarla_9Cams_w600_W6.ini;
./timeout.py 5000 python2 chauffeur.py drive --driver Human -in Carla -dc 9cam_agent_carla_acquire_rc -cy carla_1 -m 0.2 -p 2000  -nm W10_w600 -cc /home/muellem/Downloads/carla_chauffeur/drive_interfaces/carla/rcCarla_9Cams_w600_W10.ini
#w700
./timeout.py 5000 python2 chauffeur.py drive --driver Human -in Carla -dc 9cam_agent_carla_acquire_rc -cy carla_1 -m 0.2 -p 2000  -nm W1_w700 -cc /home/muellem/Downloads/carla_chauffeur/drive_interfaces/carla/rcCarla_9Cams_w700_W1.ini; 
./timeout.py 5000 python2 chauffeur.py drive --driver Human -in Carla -dc 9cam_agent_carla_acquire_rc -cy carla_1 -m 0.2 -p 2000  -nm W6_w700 -cc /home/muellem/Downloads/carla_chauffeur/drive_interfaces/carla/rcCarla_9Cams_w700_W6.ini;
./timeout.py 5000 python2 chauffeur.py drive --driver Human -in Carla -dc 9cam_agent_carla_acquire_rc -cy carla_1 -m 0.2 -p 2000  -nm W10_w700 -cc /home/muellem/Downloads/carla_chauffeur/drive_interfaces/carla/rcCarla_9Cams_w700_W10.ini
#w900
./timeout.py 5000 python2 chauffeur.py drive --driver Human -in Carla -dc 9cam_agent_carla_acquire_rc -cy carla_1 -m 0.2 -p 2000  -nm W1_w900 -cc /home/muellem/Downloads/carla_chauffeur/drive_interfaces/carla/rcCarla_9Cams_w900_W1.ini; 
./timeout.py 5000 python2 chauffeur.py drive --driver Human -in Carla -dc 9cam_agent_carla_acquire_rc -cy carla_1 -m 0.2 -p 2000  -nm W6_w900 -cc /home/muellem/Downloads/carla_chauffeur/drive_interfaces/carla/rcCarla_9Cams_w900_W6.ini;
./timeout.py 5000 python2 chauffeur.py drive --driver Human -in Carla -dc 9cam_agent_carla_acquire_rc -cy carla_1 -m 0.2 -p 2000  -nm W10_w900 -cc /home/muellem/Downloads/carla_chauffeur/drive_interfaces/carla/rcCarla_9Cams_w900_W10.ini
#w1000
./timeout.py 5000 python2 chauffeur.py drive --driver Human -in Carla -dc 9cam_agent_carla_acquire_rc -cy carla_1 -m 0.2 -p 2000  -nm W1_w1000 -cc /home/muellem/Downloads/carla_chauffeur/drive_interfaces/carla/rcCarla_9Cams_w1000_W1.ini; 
./timeout.py 5000 python2 chauffeur.py drive --driver Human -in Carla -dc 9cam_agent_carla_acquire_rc -cy carla_1 -m 0.2 -p 2000  -nm W6_w1000 -cc /home/muellem/Downloads/carla_chauffeur/drive_interfaces/carla/rcCarla_9Cams_w1000_W6.ini;
./timeout.py 5000 python2 chauffeur.py drive --driver Human -in Carla -dc 9cam_agent_carla_acquire_rc -cy carla_1 -m 0.2 -p 2000  -nm W10_w1000 -cc /home/muellem/Downloads/carla_chauffeur/drive_interfaces/carla/rcCarla_9Cams_w1000_W10.ini
#w1100
./timeout.py 5000 python2 chauffeur.py drive --driver Human -in Carla -dc 9cam_agent_carla_acquire_rc -cy carla_1 -m 0.2 -p 2000  -nm W1_w1100 -cc /home/muellem/Downloads/carla_chauffeur/drive_interfaces/carla/rcCarla_9Cams_w1100_W1.ini; 
./timeout.py 5000 python2 chauffeur.py drive --driver Human -in Carla -dc 9cam_agent_carla_acquire_rc -cy carla_1 -m 0.2 -p 2000  -nm W6_w1100 -cc /home/muellem/Downloads/carla_chauffeur/drive_interfaces/carla/rcCarla_9Cams_w1100_W6.ini;
./timeout.py 5000 python2 chauffeur.py drive --driver Human -in Carla -dc 9cam_agent_carla_acquire_rc -cy carla_1 -m 0.2 -p 2000  -nm W10_w1100 -cc /home/muellem/Downloads/carla_chauffeur/drive_interfaces/carla/rcCarla_9Cams_w1100_W10.ini
#w1200
./timeout.py 5000 python2 chauffeur.py drive --driver Human -in Carla -dc 9cam_agent_carla_acquire_rc -cy carla_1 -m 0.2 -p 2000  -nm W1_w1200 -cc /home/muellem/Downloads/carla_chauffeur/drive_interfaces/carla/rcCarla_9Cams_w1200_W1.ini; 
./timeout.py 5000 python2 chauffeur.py drive --driver Human -in Carla -dc 9cam_agent_carla_acquire_rc -cy carla_1 -m 0.2 -p 2000  -nm W6_w1200 -cc /home/muellem/Downloads/carla_chauffeur/drive_interfaces/carla/rcCarla_9Cams_w1200_W6.ini;
./timeout.py 5000 python2 chauffeur.py drive --driver Human -in Carla -dc 9cam_agent_carla_acquire_rc -cy carla_1 -m 0.2 -p 2000  -nm W10_w1200 -cc /home/muellem/Downloads/carla_chauffeur/drive_interfaces/carla/rcCarla_9Cams_w1200_W10.ini
#z50
./timeout.py 5000 python2 chauffeur.py drive --driver Human -in Carla -dc 9cam_agent_carla_acquire_rc -cy carla_1 -m 0.2 -p 2000  -nm W1_z50 -cc /home/muellem/Downloads/carla_chauffeur/drive_interfaces/carla/rcCarla_9Cams_z50_W1.ini; 
./timeout.py 5000 python2 chauffeur.py drive --driver Human -in Carla -dc 9cam_agent_carla_acquire_rc -cy carla_1 -m 0.2 -p 2000  -nm W6_z50 -cc /home/muellem/Downloads/carla_chauffeur/drive_interfaces/carla/rcCarla_9Cams_z50_W6.ini;
./timeout.py 5000 python2 chauffeur.py drive --driver Human -in Carla -dc 9cam_agent_carla_acquire_rc -cy carla_1 -m 0.2 -p 2000  -nm W10_z50 -cc /home/muellem/Downloads/carla_chauffeur/drive_interfaces/carla/rcCarla_9Cams_z50_W10.ini
#z150
./timeout.py 5000 python2 chauffeur.py drive --driver Human -in Carla -dc 9cam_agent_carla_acquire_rc -cy carla_1 -m 0.2 -p 2000  -nm W1_z150 -cc /home/muellem/Downloads/carla_chauffeur/drive_interfaces/carla/rcCarla_9Cams_z150_W1.ini; 
./timeout.py 5000 python2 chauffeur.py drive --driver Human -in Carla -dc 9cam_agent_carla_acquire_rc -cy carla_1 -m 0.2 -p 2000  -nm W6_z150 -cc /home/muellem/Downloads/carla_chauffeur/drive_interfaces/carla/rcCarla_9Cams_z150_W6.ini;
./timeout.py 5000 python2 chauffeur.py drive --driver Human -in Carla -dc 9cam_agent_carla_acquire_rc -cy carla_1 -m 0.2 -p 2000  -nm W10_z150 -cc /home/muellem/Downloads/carla_chauffeur/drive_interfaces/carla/rcCarla_9Cams_z150_W10.ini

COMMENT

#Plain
./timeout.py 2500 python2 chauffeur.py drive --driver Human -in Carla -dc 9cam_agent_carla_acquire_rc -cy carla_1 -m 0.2 -p 2000  -nm W1_N -n "Spike" -cc /home/muellem/Downloads/carla_chauffeur/drive_interfaces/carla/rcCarla_9Cams_W1.ini; 
./timeout.py 2500 python2 chauffeur.py drive --driver Human -in Carla -dc 9cam_agent_carla_acquire_rc -cy carla_1 -m 0.2 -p 2000  -nm W6_N -n "Spike" -cc /home/muellem/Downloads/carla_chauffeur/drive_interfaces/carla/rcCarla_9Cams_W6.ini;
./timeout.py 2500 python2 chauffeur.py drive --driver Human -in Carla -dc 9cam_agent_carla_acquire_rc -cy carla_1 -m 0.2 -p 2000  -nm W10_N -n "Spike" -cc /home/muellem/Downloads/carla_chauffeur/drive_interfaces/carla/rcCarla_9Cams_W10.ini

#pm5
./timeout.py 2500 python2 chauffeur.py drive --driver Human -in Carla -dc 9cam_agent_carla_acquire_rc -cy carla_1 -m 0.2 -p 2000  -nm W1_pm5_N -n "Spike" -cc /home/muellem/Downloads/carla_chauffeur/drive_interfaces/carla/rcCarla_9Cams_pm5_W1.ini; 
./timeout.py 2500 python2 chauffeur.py drive --driver Human -in Carla -dc 9cam_agent_carla_acquire_rc -cy carla_1 -m 0.2 -p 2000  -nm W6_pm5_N -n "Spike" -cc /home/muellem/Downloads/carla_chauffeur/drive_interfaces/carla/rcCarla_9Cams_pm5_W6.ini;
./timeout.py 2500 python2 chauffeur.py drive --driver Human -in Carla -dc 9cam_agent_carla_acquire_rc -cy carla_1 -m 0.2 -p 2000  -nm W10_pm5_N -n "Spike" -cc /home/muellem/Downloads/carla_chauffeur/drive_interfaces/carla/rcCarla_9Cams_pm5_W10.ini
#pp5
./timeout.py 2500 python2 chauffeur.py drive --driver Human -in Carla -dc 9cam_agent_carla_acquire_rc -cy carla_1 -m 0.2 -p 2000  -nm W1_pp5_N -n "Spike" -cc /home/muellem/Downloads/carla_chauffeur/drive_interfaces/carla/rcCarla_9Cams_pp5_W1.ini; 
./timeout.py 2500 python2 chauffeur.py drive --driver Human -in Carla -dc 9cam_agent_carla_acquire_rc -cy carla_1 -m 0.2 -p 2000  -nm W6_pp5_N -n "Spike" -cc /home/muellem/Downloads/carla_chauffeur/drive_interfaces/carla/rcCarla_9Cams_pp5_W6.ini;
./timeout.py 2500 python2 chauffeur.py drive --driver Human -in Carla -dc 9cam_agent_carla_acquire_rc -cy carla_1 -m 0.2 -p 2000  -nm W10_pp5_N -n "Spike" -cc /home/muellem/Downloads/carla_chauffeur/drive_interfaces/carla/rcCarla_9Cams_pp5_W10.ini

#z50
./timeout.py 2500 python2 chauffeur.py drive --driver Human -in Carla -dc 9cam_agent_carla_acquire_rc -cy carla_1 -m 0.2 -p 2000  -nm W1_z50_N -n "Spike" -cc /home/muellem/Downloads/carla_chauffeur/drive_interfaces/carla/rcCarla_9Cams_z50_W1.ini; 
./timeout.py 2500 python2 chauffeur.py drive --driver Human -in Carla -dc 9cam_agent_carla_acquire_rc -cy carla_1 -m 0.2 -p 2000  -nm W6_z50_N -n "Spike" -cc /home/muellem/Downloads/carla_chauffeur/drive_interfaces/carla/rcCarla_9Cams_z50_W6.ini;
./timeout.py 2500 python2 chauffeur.py drive --driver Human -in Carla -dc 9cam_agent_carla_acquire_rc -cy carla_1 -m 0.2 -p 2000  -nm W10_z50_N -n "Spike" -cc /home/muellem/Downloads/carla_chauffeur/drive_interfaces/carla/rcCarla_9Cams_z50_W10.ini
#z150
./timeout.py 2500 python2 chauffeur.py drive --driver Human -in Carla -dc 9cam_agent_carla_acquire_rc -cy carla_1 -m 0.2 -p 2000  -nm W1_z150_N -n "Spike" -cc /home/muellem/Downloads/carla_chauffeur/drive_interfaces/carla/rcCarla_9Cams_z150_W1.ini; 
./timeout.py 2500 python2 chauffeur.py drive --driver Human -in Carla -dc 9cam_agent_carla_acquire_rc -cy carla_1 -m 0.2 -p 2000  -nm W6_z150_N -n "Spike" -cc /home/muellem/Downloads/carla_chauffeur/drive_interfaces/carla/rcCarla_9Cams_z150_W6.ini;
./timeout.py 2500 python2 chauffeur.py drive --driver Human -in Carla -dc 9cam_agent_carla_acquire_rc -cy carla_1 -m 0.2 -p 2000  -nm W10_z150_N -n "Spike" -cc /home/muellem/Downloads/carla_chauffeur/drive_interfaces/carla/rcCarla_9Cams_z150_W10.ini

