#!/bin/bash



mkdir -p result/base/neural/
mkdir -p result/total/10/
mkdir -p result/total/500/
mkdir -p result/reward/1/
mkdir -p result/reward/10/
mkdir -p result/reward/100/
mkdir -p result/init_state/random/
mkdir -p result/epsilon/0.3/
mkdir -p result/epsilon/0.6/
mkdir -p result/episodes/1/
mkdir -p result/episodes/50/
mkdir -p result/decay/0.1/
mkdir -p result/decay/0.5/
# 
python pybrain_rl_simple2.py -p ./result/total/10/ -t 10
python pybrain_rl_simple2.py -p ./result/total/500/ -t 50

# 
# python pybrain_rl_simple2.py -p ./result/base/neural/
# 
# python pybrain_rl_simple2.py -p ./result/total/10/ -t 10
# python pybrain_rl_simple2.py -p ./result/total/500/ -t 500
# 
# python pybrain_rl_simple2.py -p ./result/reward/1/ -r 1
# python pybrain_rl_simple2.py -p ./result/reward/10/ -r 10
# python pybrain_rl_simple2.py -p ./result/reward/100/ -r 100
# 
# python pybrain_rl_simple2.py -p ./result/init_state/random/ -i True
# 
# python pybrain_rl_simple2.py -p ./result/epsilon/0.3/ -l 0.3
# python pybrain_rl_simple2.py -p ./result/epsilon/0.6/ -l 0.6
# 
# python pybrain_rl_simple2.py -p ./result/episodes/1/ -e 1
# python pybrain_rl_simple2.py -p ./result/episodes/50/ -e 50
# python pybrain_rl_simple2.py -p ./result/episodes/100/ -e 100
# 
# python pybrain_rl_simple2.py -p ./result/decay/0.1/ -d 0.1
# python pybrain_rl_simple2.py -p ./result/decay/0.5 -d 0.5
