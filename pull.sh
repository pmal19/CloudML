#!/bin/bash

git pull origin master
sbatch run_job.sh
squeue -u pm2758