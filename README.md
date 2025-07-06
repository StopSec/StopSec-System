StopSec is a wireless protocol designed for cooperative spectrum sharing, enabling primary users to reactively stop interference from secondary users.

This profile is intended for doing any experiment using StopSec. 

This profile can allocate X310 radios (+ compute) and Database Server.

Instructions:

**1) Instantiate this profile with appropriate parameters**

At the "Parameterize" step, add at least 2 rooftop radios in the CBRS band that are needed for your planned experiment. By default, one database server from the group d430 or d740 will be added. 
Also, speceify the freqeuncy ranges if you are planning to use transmitter(s) in your experiment. Also by default, all the nodes will be setup in one LAN. Try to ping from one node to the other to try if they are in the same LAN.
Once you have these parameters selected, click through the rest of the profile and then click "Finish" to instantiate.  It will take 10 to 15 minutes for the experiment to finish setting up.  Once it is "green", proceed to the next step.

**2) Setting up the experiment**
- once the experiment is ready, the database serever will be indicated on the list view. Other nodes will be indicated as rooftop nodes which the user has to choose. 
- select one of the rooftop nodes to be the primary user (PU)
- select at least one of the other rooftop nodes as the secondary user (SU)
- run the following command on each of the nodes
  ```
  ssh -Y <username>@<radio_hostname>
  ```
  
**3) Cloning StopSec to Each Node**
Run the following command on each node to clone StopSec repository to your nodes. For example: ssh -Y username@pc707.emulab.net
  ```
git clone https://github.com/StopSec/StopSec-System.git
  ```
Run the following command on each node to move to the directory that contains the StopSec files.

  ```
cd StopSec-System
  ```
  


