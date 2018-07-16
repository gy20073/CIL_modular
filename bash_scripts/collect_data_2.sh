#!/bin/bash
: <<'COMMENT'
#Terminal2
#pm5
./timeout.py 5000 python2 chauffeur.py drive --driver Human -in Carla -dc 9cam_agent_carla_acquire_rc -cy carla_1 -m 0.2 -p 2012  -nm W2_pm5 -cc /home/muellem/Downloads/carla_chauffeur/drive_interfaces/carla/rcCarla_9Cams_pm5_W2.ini; 
./timeout.py 5000 python2 chauffeur.py drive --driver Human -in Carla -dc 9cam_agent_carla_acquire_rc -cy carla_1 -m 0.2 -p 2012  -nm W7_pm5 -cc /home/muellem/Downloads/carla_chauffeur/drive_interfaces/carla/rcCarla_9Cams_pm5_W7.ini;
./timeout.py 5000 python2 chauffeur.py drive --driver Human -in Carla -dc 9cam_agent_carla_acquire_rc -cy carla_1 -m 0.2 -p 2012  -nm W11_pm5 -cc /home/muellem/Downloads/carla_chauffeur/drive_interfaces/carla/rcCarla_9Cams_pm5_W11.ini
#pp5
./timeout.py 5000 python2 chauffeur.py drive --driver Human -in : <<'COMMENT'Carla -dc 9cam_agent_carla_acquire_rc -cy carla_1 -m 0.2 -p 2012  -nm W2_pp5 -cc /home/muellem/Downloads/carla_chauffeur/drive_interfaces/carla/rcCarla_9Cams_pp5_W2.ini; 
./timeout.py 5000 python2 chauffeur.py drive --driver Human -in Carla -dc 9cam_agent_carla_acquire_rc -cy carla_1 -m 0.2 -p 2012  -nm W7_pp5 -cc /home/muellem/Downloads/carla_chauffeur/drive_interfaces/carla/rcCarla_9Cams_pp5_W7.ini;
./timeout.py 5000 python2 chauffeur.py drive --driver Human -in Carla -dc 9cam_agent_carla_acquire_rc -cy carla_1 -m 0.2 -p 2012  -nm W11_pp5 -cc /home/muellem/Downloads/carla_chauffeur/drive_interfaces/carla/rcCarla_9Cams_pp5_W11.ini
#W700
./timeout.py 5000 python2 chauffeur.py drive --driver Human -in Carla -dc 9cam_agent_carla_acquire_rc -cy carla_1 -m 0.2 -p 2012  -nm W2_w600 -cc /home/muellem/Downloads/carla_chauffeur/drive_interfaces/carla/rcCarla_9Cams_w600_W2.ini; 
./timeout.py 5000 python2 chauffeur.py drive --driver Human -in Carla -dc 9cam_agent_carla_acquire_rc -cy carla_1 -m 0.2 -p 2012  -nm W7_w600 -cc /home/muellem/Downloads/carla_chauffeur/drive_interfaces/carla/rcCarla_9Cams_w600_W7.ini;
./timeout.py 5000 python2 chauffeur.py drive --driver Human -in Carla -dc 9cam_agent_carla_acquire_rc -cy carla_1 -m 0.2 -p 2012  -nm W11_w600 -cc /home/muellem/Downloads/carla_chauffeur/drive_interfaces/carla/rcCarla_9Cams_w600_W11.ini
#w700
./timeout.py 5000 python2 chauffeur.py drive --driver Human -in Carla -dc 9cam_agent_carla_acquire_rc -cy carla_1 -m 0.2 -p 2012  -nm W2_w700 -cc /home/muellem/Downloads/carla_chauffeur/drive_interfaces/carla/rcCarla_9Cams_w700_W2.ini; 
./timeout.py 5000 python2 chauffeur.py drive --driver Human -in Carla -dc 9cam_agent_carla_acquire_rc -cy carla_1 -m 0.2 -p 2012  -nm W7_w700 -cc /home/muellem/Downloads/carla_chauffeur/drive_interfaces/carla/rcCarla_9Cams_w700_W7.ini;
./timeout.py 5000 python2 chauffeur.py drive --driver Human -in Carla -dc 9cam_agent_carla_acquire_rc -cy carla_1 -m 0.2 -p 2012  -nm W11_w700 -cc /home/muellem/Downloads/carla_chauffeur/drive_interfaces/carla/rcCarla_9Cams_w700_W11.ini
#w900
./timeout.py 5000 python2 chauffeur.py drive --driver Human -in Carla -dc 9cam_agent_carla_acquire_rc -cy carla_1 -m 0.2 -p 2012  -nm W2_w900 -cc /home/muellem/Downloads/carla_chauffeur/drive_interfaces/carla/rcCarla_9Cams_w900_W2.ini; 
./timeout.py 5000 python2 chauffeur.py drive --driver Human -in Carla -dc 9cam_agent_carla_acquire_rc -cy carla_1 -m 0.2 -p 2012  -nm W7_w900 -cc /home/muellem/Downloads/carla_chauffeur/drive_interfaces/carla/rcCarla_9Cams_w900_W7.ini;
./timeout.py 5000 python2 chauffeur.py drive --driver Human -in Carla -dc 9cam_agent_carla_acquire_rc -cy carla_1 -m 0.2 -p 2012  -nm W11_w900 -cc /home/muellem/Downloads/carla_chauffeur/drive_interfaces/carla/rcCarla_9Cams_w900_W11.ini
#w1000
./timeout.py 5000 python2 chauffeur.py drive --driver Human -in Carla -dc 9cam_agent_carla_acquire_rc -cy carla_1 -m 0.2 -p 2012  -nm W2_w1000 -cc /home/muellem/Downloads/carla_chauffeur/drive_interfaces/carla/rcCarla_9Cams_w1000_W2.ini; 
./timeout.py 5000 python2 chauffeur.py drive --driver Human -in Carla -dc 9cam_agent_carla_acquire_rc -cy carla_1 -m 0.2 -p 2012  -nm W7_w1000 -cc /home/muellem/Downloads/carla_chauffeur/drive_interfaces/carla/rcCarla_9Cams_w1000_W7.ini;
./timeout.py 5000 python2 chauffeur.py drive --driver Human -in Carla -dc 9cam_agent_carla_acquire_rc -cy carla_1 -m 0.2 -p 2012  -nm W11_w1000 -cc /home/muellem/Downloads/carla_chauffeur/drive_interfaces/carla/rcCarla_9Cams_w1000_W11.ini
#w1100
./timeout.py 5000 python2 chauffeur.py drive --driver Human -in Carla -dc 9cam_agent_carla_acquire_rc -cy carla_1 -m 0.2 -p 2012  -nm W2_w1100 -cc /home/muellem/Downloads/carla_chauffeur/drive_interfaces/carla/rcCarla_9Cams_w1100_W2.ini; 
./timeout.py 5000 python2 chauffeur.py drive --driver Human -in Carla -dc 9cam_agent_carla_acquire_rc -cy carla_1 -m 0.2 -p 2012  -nm W7_w1100 -cc /home/muellem/Downloads/carla_chauffeur/drive_interfaces/carla/rcCarla_9Cams_w1100_W7.ini;
./timeout.py 5000 python2 chauffeur.py drive --driver Human -in Carla -dc 9cam_agent_carla_acquire_rc -cy carla_1 -m 0.2 -p 2012  -nm W11_w1100 -cc /home/muellem/Downloads/carla_chauffeur/drive_interfaces/carla/rcCarla_9Cams_w1100_W11.ini
#w1200
./timeout.py 5000 python2 chauffeur.py drive --driver Human -in Carla -dc 9cam_agent_carla_acquire_rc -cy carla_1 -m 0.2 -p 2012  -nm W2_w1200 -cc /home/muellem/Downloads/carla_chauffeur/drive_interfaces/carla/rcCarla_9Cams_w1200_W2.ini; 
./timeout.py 5000 python2 chauffeur.py drive --driver Human -in Carla -dc 9cam_agent_carla_acquire_rc -cy carla_1 -m 0.2 -p 2012  -nm W7_w1200 -cc /home/muellem/Downloads/carla_chauffeur/drive_interfaces/carla/rcCarla_9Cams_w1200_W7.ini;
./timeout.py 5000 python2 chauffeur.py drive --driver Human -in Carla -dc 9cam_agent_carla_acquire_rc -cy carla_1 -m 0.2 -p 2012  -nm W11_w1200 -cc /home/muellem/Downloads/carla_chauffeur/drive_interfaces/carla/rcCarla_9Cams_w1200_W11.ini
#z50
./timeout.py 5000 python2 chauffeur.py drive --driver Human -in Carla -dc 9cam_agent_carla_acquire_rc -cy carla_1 -m 0.2 -p 2012  -nm W2_z50 -cc /home/muellem/Downloads/carla_chauffeur/drive_interfaces/carla/rcCarla_9Cams_z50_W2.ini; 
./timeout.py 5000 python2 chauffeur.py drive --driver Human -in Carla -dc 9cam_agent_carla_acquire_rc -cy carla_1 -m 0.2 -p 2012  -nm W7_z50 -cc /home/muellem/Downloads/carla_chauffeur/drive_interfaces/carla/rcCarla_9Cams_z50_W7.ini;
./timeout.py 5000 python2 chauffeur.py drive --driver Human -in Carla -dc 9cam_agent_carla_acquire_rc -cy carla_1 -m 0.2 -p 2012  -nm W11_z50 -cc /home/muellem/Downloads/carla_chauffeur/drive_interfaces/carla/rcCarla_9Cams_z50_W11.ini
#z150
./timeout.py 5000 python2 chauffeur.py drive --driver Human -in Carla -dc 9cam_agent_carla_acquire_rc -cy carla_1 -m 0.2 -p 2012  -nm W2_z150 -cc /home/muellem/Downloads/carla_chauffeur/drive_interfaces/carla/rcCarla_9Cams_z150_W2.ini; 
./timeout.py 5000 python2 chauffeur.py drive --driver Human -in Carla -dc 9cam_agent_carla_acquire_rc -cy carla_1 -m 0.2 -p 2012  -nm W7_z150 -cc /home/muellem/Downloads/carla_chauffeur/drive_interfaces/carla/rcCarla_9Cams_z150_W7.ini;
./timeout.py 5000 python2 chauffeur.py drive --driver Human -in Carla -dc 9cam_agent_carla_acquire_rc -cy carla_1 -m 0.2 -p 2012  -nm W11_z150 -cc /home/muellem/Downloads/carla_chauffeur/drive_interfaces/carla/rcCarla_9Cams_z150_W11.ini

COMMENT

#Plain
./timeout.py 2500 python2 chauffeur.py drive --driver Human -in Carla -dc 9cam_agent_carla_acquire_rc -cy carla_1 -m 0.2 -p 2012  -nm W2_N -n "Spike" -cc /home/muellem/Downloads/carla_chauffeur/drive_interfaces/carla/rcCarla_9Cams_W2.ini; 
./timeout.py 2500 python2 chauffeur.py drive --driver Human -in Carla -dc 9cam_agent_carla_acquire_rc -cy carla_1 -m 0.2 -p 2012  -nm W7_N -n "Spike" -cc /home/muellem/Downloads/carla_chauffeur/drive_interfaces/carla/rcCarla_9Cams_W7.ini;
./timeout.py 2500 python2 chauffeur.py drive --driver Human -in Carla -dc 9cam_agent_carla_acquire_rc -cy carla_1 -m 0.2 -p 2012  -nm W11_N -n "Spike" -cc /home/muellem/Downloads/carla_chauffeur/drive_interfaces/carla/rcCarla_9Cams_W11.ini

#pm5
./timeout.py 2500 python2 chauffeur.py drive --driver Human -in Carla -dc 9cam_agent_carla_acquire_rc -cy carla_1 -m 0.2 -p 2012  -nm W2_pm5_N -n "Spike" -cc /home/muellem/Downloads/carla_chauffeur/drive_interfaces/carla/rcCarla_9Cams_pm5_W2.ini; 
./timeout.py 2500 python2 chauffeur.py drive --driver Human -in Carla -dc 9cam_agent_carla_acquire_rc -cy carla_1 -m 0.2 -p 2012  -nm W7_pm5_N -n "Spike" -cc /home/muellem/Downloads/carla_chauffeur/drive_interfaces/carla/rcCarla_9Cams_pm5_W7.ini;
./timeout.py 2500 python2 chauffeur.py drive --driver Human -in Carla -dc 9cam_agent_carla_acquire_rc -cy carla_1 -m 0.2 -p 2012  -nm W11_pm5_N -n "Spike" -cc /home/muellem/Downloads/carla_chauffeur/drive_interfaces/carla/rcCarla_9Cams_pm5_W11.ini
#pp5
./timeout.py 2500 python2 chauffeur.py drive --driver Human -in Carla -dc 9cam_agent_carla_acquire_rc -cy carla_1 -m 0.2 -p 2012  -nm W2_pp5_N -n "Spike" -cc /home/muellem/Downloads/carla_chauffeur/drive_interfaces/carla/rcCarla_9Cams_pp5_W2.ini; 
./timeout.py 2500 python2 chauffeur.py drive --driver Human -in Carla -dc 9cam_agent_carla_acquire_rc -cy carla_1 -m 0.2 -p 2012  -nm W7_pp5_N -n "Spike" -cc /home/muellem/Downloads/carla_chauffeur/drive_interfaces/carla/rcCarla_9Cams_pp5_W7.ini;
./timeout.py 2500 python2 chauffeur.py drive --driver Human -in Carla -dc 9cam_agent_carla_acquire_rc -cy carla_1 -m 0.2 -p 2012  -nm W11_pp5_N -n "Spike" -cc /home/muellem/Downloads/carla_chauffeur/drive_interfaces/carla/rcCarla_9Cams_pp5_W11.ini

#z50
./timeout.py 2500 python2 chauffeur.py drive --driver Human -in Carla -dc 9cam_agent_carla_acquire_rc -cy carla_1 -m 0.2 -p 2012  -nm W2_z50_N -n "Spike" -cc /home/muellem/Downloads/carla_chauffeur/drive_interfaces/carla/rcCarla_9Cams_z50_W2.ini; 
./timeout.py 2500 python2 chauffeur.py drive --driver Human -in Carla -dc 9cam_agent_carla_acquire_rc -cy carla_1 -m 0.2 -p 2012  -nm W7_z50_N -n "Spike" -cc /home/muellem/Downloads/carla_chauffeur/drive_interfaces/carla/rcCarla_9Cams_z50_W7.ini;
./timeout.py 2500 python2 chauffeur.py drive --driver Human -in Carla -dc 9cam_agent_carla_acquire_rc -cy carla_1 -m 0.2 -p 2012  -nm W11_z50_N -n "Spike" -cc /home/muellem/Downloads/carla_chauffeur/drive_interfaces/carla/rcCarla_9Cams_z50_W11.ini
#z150
./timeout.py 2500 python2 chauffeur.py drive --driver Human -in Carla -dc 9cam_agent_carla_acquire_rc -cy carla_1 -m 0.2 -p 2012  -nm W2_z150_N -n "Spike" -cc /home/muellem/Downloads/carla_chauffeur/drive_interfaces/carla/rcCarla_9Cams_z150_W2.ini; 
./timeout.py 2500 python2 chauffeur.py drive --driver Human -in Carla -dc 9cam_agent_carla_acquire_rc -cy carla_1 -m 0.2 -p 2012  -nm W7_z150_N -n "Spike" -cc /home/muellem/Downloads/carla_chauffeur/drive_interfaces/carla/rcCarla_9Cams_z150_W7.ini;
./timeout.py 2500 python2 chauffeur.py drive --driver Human -in Carla -dc 9cam_agent_carla_acquire_rc -cy carla_1 -m 0.2 -p 2012  -nm W11_z150_N -n "Spike" -cc /home/muellem/Downloads/carla_chauffeur/drive_interfaces/carla/rcCarla_9Cams_z150_W11.ini
