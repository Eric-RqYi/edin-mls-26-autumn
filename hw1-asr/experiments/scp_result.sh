#! /usr/bin/env bash
TASK=$1
scp -r \
  -o GSSAPIAuthentication=yes \
  -o GSSAPIDelegateCredentials=yes \
  -J s2955206@student.ssh.inf.ed.ac.uk \
  s2955206@icf.inf.ed.ac.uk:~/edin-mls-26-autumn/hw1-asr/experiments/${TASK}/results \
  /Users/eric/Library/CloudStorage/OneDrive-UniversityofEdinburgh/Projects/MLS-group-project/edin-mls-26-autumn/hw1-asr/experiments/${TASK}/        