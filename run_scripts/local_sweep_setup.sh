#!/usr/bin/bash

ssh_ip="$1"
ssh_port="$2"

rsync -rltvz --progress -e "ssh -p $ssh_port -i ~/.ssh/id_ed25519" /home/mattylev/projects/simplex/SAEs/mess3_sae/outputs/saes/multipartite_003e/* root@$ssh_ip:/workspace/outputs/saes/multipartite_003e/
rsync -rltvz --progress -e "ssh -p $ssh_port -i ~/.ssh/id_ed25519" /home/mattylev/projects/simplex/SAEs/mess3_sae/outputs/checkpoints/multipartite_003/checkpoint_step_6000_best.pt root@$ssh_ip:/workspace/outputs/checkpoints/multipartite_003/checkpoint_step_6000_best.pt