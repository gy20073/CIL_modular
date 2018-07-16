#!/bin/bash
: <<'COMMENT'
#Terminal3
#pm5
./timeout.py 5000 python2 chauffeur.py drive --driver Human -in Carla -dc 9cam_agent_carla_acquire_rc -cy carla_1 -m 0.2 -p 2006  -nm W3_pm5 -cc /home/muellem/Downloads/carla_chauffeur/drive_interfaces/carla/rcCarla_9Cams_pm5_W3.ini; 
./timeout.py 5000 python2 chauffeur.py drive --driver Human -in Carla -dc 9cam_agent_carla_acquire_rc -cy carla_1 -m 0.2 -p 2006  -nm W8_pm5 -cc /home/muellem/Downloads/carla_chauffeur/drive_interfaces/carla/rcCarla_9Cams_pm5_W8.ini;
./timeout.py 5000 python2 chauffeur.py drive --driver Human -in Carla -dc 9cam_agent_carla_acquire_rc -cy carla_1 -m 0.2 -p 2006  -nm W12_pm5 -cc /home/muellem/Downloads/carla_chauffeur/drive_interfaces/carla/rcCarla_9Cams_pm5_W12.ini
#pp5
./timeout.py 5000 python2 chauffeur.py drive --driver Human -in Carla -dc 9cam_agent_carla_acquire_rc -cy carla_1 -m 0.2 -p 2006  -nm W3_pp5 -cc /home/muellem/Downloads/carla_chauffeur/drive_interfaces/carla/rcCarla_9Cams_pp5_W3.ini; 
./timeout.py 5000 python2 chauffeur.py drive --driver Human -in Carla -dc 9cam_agent_carla_acquire_rc -cy carla_1 -m 0.2 -p 2006  -nm W8_pp5 -cc /home/muellem/Downloads/carla_chauffeur/drive_interfaces/carla/rcCarla_9Cams_pp5_W8.ini;
./timeout.py 5000 python2 chauffeur.py drive --driver Human -in Carla -dc 9cam_agent_carla_acquire_rc -cy carla_1 -m 0.2 -p 2006  -nm W12_pp5 -cc /home/muellem/Downloads/carla_chauffeur/drive_interfaces/carla/rcCarla_9Cams_pp5_W12.ini
#w600
./timeout.py 5000 python2 chauffeur.py drive --driver Human -in Carla -dc 9cam_agent_carla_acquire_rc -cy carla_1 -m 0.2 -p 2006  -nm W3_w600 -cc /home/muellem/Downloads/carla_chauffeur/drive_interfaces/carla/rcCarla_9Cams_w600_W3.ini; 
./timeout.py 5000 python2 chauffeur.py drive --driver Human -in Carla -dc 9cam_agent_carla_acquire_rc -cy carla_1 -m 0.2 -p 2006  -nm W8_w600 -cc /home/muellem/Downloads/carla_chauffeur/drive_interfaces/carla/rcCarla_9Cams_w600_W8.ini;
./timeout.py 5000 python2 chauffeur.py drive --driver Human -in Carla -dc 9cam_agent_carla_acquire_rc -cy carla_1 -m 0.2 -p 2006  -nm W12_w600 -cc /home/muellem/Downloads/carla_chauffeur/drive_interfaces/carla/rcCarla_9Cams_w600_W12.ini
#w700
./timeout.py 5000 python2 chauffeur.py drive --driver Human -in Carla -dc 9cam_agent_carla_acquire_rc -cy carla_1 -m 0.2 -p 2006  -nm W3_w700 -cc /home/muellem/Downloads/carla_chauffeur/drive_interfaces/carla/rcCarla_9Cams_w700_W3.ini; 
./timeout.py 5000 python2 chauffeur.py drive --driver Human -in Carla -dc 9cam_agent_carla_acquire_rc -cy carla_1 -m 0.2 -p 2006  -nm W8_w700 -cc /home/muellem/Downloads/carla_chauffeur/drive_interfaces/carla/rcCarla_9Cams_w700_W8.ini;
./timeout.py 5000 python2 chauffeur.py drive --driver Human -in Carla -dc 9cam_agent_carla_acquire_rc -cy carla_1 -m 0.2 -p 2006  -nm W12_w700 -cc /home/muellem/Downloads/carla_chauffeur/drive_interfaces/carla/rcCarla_9Cams_w700_W12.ini
#w900
./timeout.py 5000 python2 chauffeur.py drive --driver Human -in Carla -dc 9cam_agent_carla_acquire_rc -cy carla_1 -m 0.2 -p 2006  -nm W3_w900 -cc /home/muellem/Downloads/carla_chauffeur/drive_interfaces/carla/rcCarla_9Cams_w900_W3.ini; 
./timeout.py 5000 python2 chauffeur.py drive --driver Human -in Carla -dc 9cam_agent_carla_acquire_rc -cy carla_1 -m 0.2 -p 2006  -nm W8_w900 -cc /home/muellem/Downloads/carla_chauffeur/drive_interfaces/carla/rcCarla_9Cams_w900_W8.ini;
./timeout.py 5000 python2 chauffeur.py drive --driver Human -in Carla -dc 9cam_agent_carla_acquire_rc -cy carla_1 -m 0.2 -p 2006  -nm W12_w900 -cc /home/muellem/Downloads/carla_chauffeur/drive_interfaces/carla/rcCarla_9Cams_w900_W12.ini
#w1000
./timeout.py 5000 python2 chauffeur.py drive --driver Human -in Carla -dc 9cam_agent_carla_acquire_rc -cy carla_1 -m 0.2 -p 2006  -nm W3_w1000 -cc /home/muellem/Downloads/carla_chauffeur/drive_interfaces/carla/rcCarla_9Cams_w1000_W3.ini; 
./timeout.py 5000 python2 chauffeur.py drive --driver Human -in Carla -dc 9cam_agent_carla_acquire_rc -cy carla_1 -m 0.2 -p 2006  -nm W8_w1000 -cc /home/muellem/Downloads/carla_chauffeur/drive_interfaces/carla/rcCarla_9Cams_w1000_W8.ini;
./timeout.py 5000 python2 chauffeur.py drive --driver Human -in Carla -dc 9cam_agent_carla_acquire_rc -cy carla_1 -m 0.2 -p 2006  -nm W12_w1000 -cc /home/muellem/Downloads/carla_chauffeur/drive_interfaces/carla/rcCarla_9Cams_w1000_W12.ini
#w1100
./timeout.py 5000 python2 chauffeur.py drive --driver Human -in Carla -dc 9cam_agent_carla_acquire_rc -cy carla_1 -m 0.2 -p 2006  -nm W3_w1100 -cc /home/muellem/Downloads/carla_chauffeur/drive_interfaces/carla/rcCarla_9Cams_w1100_W3.ini; 
./timeout.py 5000 python2 chauffeur.py drive --driver Human -in Carla -dc 9cam_agent_carla_acquire_rc -cy carla_1 -m 0.2 -p 2006  -nm W8_w1100 -cc /home/muellem/Downloads/carla_chauffeur/drive_interfaces/carla/rcCarla_9Cams_w1100_W8.ini;
./timeout.py 5000 python2 chauffeur.py drive --driver Human -in Carla -dc 9cam_agent_carla_acquire_rc -cy carla_1 -m 0.2 -p 2006  -nm W12_w1100 -cc /home/muellem/Downloads/carla_chauffeur/drive_interfaces/carla/rcCarla_9Cams_w1100_W12.ini
#w1200
./timeout.py 5000 python2 chauffeur.py drive --driver Human -in Carla -dc 9cam_agent_carla_acquire_rc -cy carla_1 -m 0.2 -p 2006  -nm W3_w1200 -cc /home/muellem/Downloads/carla_chauffeur/drive_interfaces/carla/rcCarla_9Cams_w1200_W3.ini; 
./timeout.py 5000 python2 chauffeur.py drive --driver Human -in Carla -dc 9cam_agent_carla_acquire_rc -cy carla_1 -m 0.2 -p 2006  -nm W8_w1200 -cc /home/muellem/Downloads/carla_chauffeur/drive_interfaces/carla/rcCarla_9Cams_w1200_W8.ini;
./timeout.py 5000 python2 chauffeur.py drive --driver Human -in Carla -dc 9cam_agent_carla_acquire_rc -cy carla_1 -m 0.2 -p 2006  -nm W12_w1200 -cc /home/muellem/Downloads/carla_chauffeur/drive_interfaces/carla/rcCarla_9Cams_w1200_W12.ini
#z50
./timeout.py 5000 python2 chauffeur.py drive --driver Human -in Carla -dc 9cam_agent_carla_acquire_rc -cy carla_1 -m 0.2 -p 2006  -nm W3_z50 -cc /home/muellem/Downloads/carla_chauffeur/drive_interfaces/carla/rcCarla_9Cams_z50_W3.ini; 
./timeout.py 5000 python2 chauffeur.py drive --driver Human -in Carla -dc 9cam_agent_carla_acquire_rc -cy carla_1 -m 0.2 -p 2006  -nm W8_z50 -cc /home/muellem/Downloads/carla_chauffeur/drive_interfaces/carla/rcCarla_9Cams_z50_W8.ini;
./timeout.py 5000 python2 chauffeur.py drive --driver Human -in Carla -dc 9cam_agent_carla_acquire_rc -cy carla_1 -m 0.2 -p 2006  -nm W12_z50 -cc /home/muellem/Downloads/carla_chauffeur/drive_interfaces/carla/rcCarla_9Cams_z50_W12.ini
#z150
./timeout.py 5000 python2 chauffeur.py drive --driver Human -in Carla -dc 9cam_agent_carla_acquire_rc -cy carla_1 -m 0.2 -p 2006  -nm W3_z150 -cc /home/muellem/Downloads/carla_chauffeur/drive_interfaces/carla/rcCarla_9Cams_z150_W3.ini; 
./timeout.py 5000 python2 chauffeur.py drive --driver Human -in Carla -dc 9cam_agent_carla_acquire_rc -cy carla_1 -m 0.2 -p 2006  -nm W8_z150 -cc /home/muellem/Downloads/carla_chauffeur/drive_interfaces/carla/rcCarla_9Cams_z150_W8.ini;
./timeout.py 5000 python2 chauffeur.py drive --driver Human -in Carla -dc 9cam_agent_carla_acquire_rc -cy carla_1 -m 0.2 -p 2006  -nm W12_z150 -cc /home/muellem/Downloads/carla_chauffeur/drive_interfaces/carla/rcCarla_9Cams_z150_W12.ini

COMMENT

#Plain
./timeout.py 2500 python2 chauffeur.py drive --driver Human -in Carla -dc 9cam_agent_carla_acquire_rc -cy carla_1 -m 0.2 -p 2006  -nm W3_N -n "Spike" -cc /home/muellem/Downloads/carla_chauffeur/drive_interfaces/carla/rcCarla_9Cams_W3.ini; 
./timeout.py 2500 python2 chauffeur.py drive --driver Human -in Carla -dc 9cam_agent_carla_acquire_rc -cy carla_1 -m 0.2 -p 2006  -nm W8_N -n "Spike" -cc /home/muellem/Downloads/carla_chauffeur/drive_interfaces/carla/rcCarla_9Cams_W8.ini;
./timeout.py 2500 python2 chauffeur.py drive --driver Human -in Carla -dc 9cam_agent_carla_acquire_rc -cy carla_1 -m 0.2 -p 2006  -nm W12_N -n "Spike" -cc /home/muellem/Downloads/carla_chauffeur/drive_interfaces/carla/rcCarla_9Cams_W12.ini

#pm5
./timeout.py 2500 python2 chauffeur.py drive --driver Human -in Carla -dc 9cam_agent_carla_acquire_rc -cy carla_1 -m 0.2 -p 2006  -nm W3_pm5_N -n "Spike" -cc /home/muellem/Downloads/carla_chauffeur/drive_interfaces/carla/rcCarla_9Cams_pm5_W3.ini; 
./timeout.py 2500 python2 chauffeur.py drive --driver Human -in Carla -dc 9cam_agent_carla_acquire_rc -cy carla_1 -m 0.2 -p 2006  -nm W8_pm5_N -n "Spike" -cc /home/muellem/Downloads/carla_chauffeur/drive_interfaces/carla/rcCarla_9Cams_pm5_W8.ini;
./timeout.py 2500 python2 chauffeur.py drive --driver Human -in Carla -dc 9cam_agent_carla_acquire_rc -cy carla_1 -m 0.2 -p 2006  -nm W12_pm5_N -n "Spike" -cc /home/muellem/Downloads/carla_chauffeur/drive_interfaces/carla/rcCarla_9Cams_pm5_W12.ini
#pp5
./timeout.py 2500 python2 chauffeur.py drive --driver Human -in Carla -dc 9cam_agent_carla_acquire_rc -cy carla_1 -m 0.2 -p 2006  -nm W3_pp5_N -n "Spike" -cc /home/muellem/Downloads/carla_chauffeur/drive_interfaces/carla/rcCarla_9Cams_pp5_W3.ini; 
./timeout.py 2500 python2 chauffeur.py drive --driver Human -in Carla -dc 9cam_agent_carla_acquire_rc -cy carla_1 -m 0.2 -p 2006  -nm W8_pp5_N -n "Spike" -cc /home/muellem/Downloads/carla_chauffeur/drive_interfaces/carla/rcCarla_9Cams_pp5_W8.ini;
./timeout.py 2500 python2 chauffeur.py drive --driver Human -in Carla -dc 9cam_agent_carla_acquire_rc -cy carla_1 -m 0.2 -p 2006  -nm W12_pp5_N -n "Spike" -cc /home/muellem/Downloads/carla_chauffeur/drive_interfaces/carla/rcCarla_9Cams_pp5_W12.ini

#z50
./timeout.py 2500 python2 chauffeur.py drive --driver Human -in Carla -dc 9cam_agent_carla_acquire_rc -cy carla_1 -m 0.2 -p 2006  -nm W3_z50_N -n "Spike" -cc /home/muellem/Downloads/carla_chauffeur/drive_interfaces/carla/rcCarla_9Cams_z50_W3.ini; 
./timeout.py 2500 python2 chauffeur.py drive --driver Human -in Carla -dc 9cam_agent_carla_acquire_rc -cy carla_1 -m 0.2 -p 2006  -nm W8_z50_N -n "Spike" -cc /home/muellem/Downloads/carla_chauffeur/drive_interfaces/carla/rcCarla_9Cams_z50_W8.ini;
./timeout.py 2500 python2 chauffeur.py drive --driver Human -in Carla -dc 9cam_agent_carla_acquire_rc -cy carla_1 -m 0.2 -p 2006  -nm W12_z50_N -n "Spike" -cc /home/muellem/Downloads/carla_chauffeur/drive_interfaces/carla/rcCarla_9Cams_z50_W12.ini
#z150
./timeout.py 2500 python2 chauffeur.py drive --driver Human -in Carla -dc 9cam_agent_carla_acquire_rc -cy carla_1 -m 0.2 -p 2006  -nm W3_z150_N -n "Spike" -cc /home/muellem/Downloads/carla_chauffeur/drive_interfaces/carla/rcCarla_9Cams_z150_W3.ini; 
./timeout.py 2500 python2 chauffeur.py drive --driver Human -in Carla -dc 9cam_agent_carla_acquire_rc -cy carla_1 -m 0.2 -p 2006  -nm W8_z150_N -n "Spike" -cc /home/muellem/Downloads/carla_chauffeur/drive_interfaces/carla/rcCarla_9Cams_z150_W8.ini;
./timeout.py 2500 python2 chauffeur.py drive --driver Human -in Carla -dc 9cam_agent_carla_acquire_rc -cy carla_1 -m 0.2 -p 2006  -nm W12_z150_N -n "Spike" -cc /home/muellem/Downloads/carla_chauffeur/drive_interfaces/carla/rcCarla_9Cams_z150_W12.ini
