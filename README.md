# PEARL: Peg-in-hole

The repo is forked from https://github.com/katerakelly/oyster, focusing on peg-in-hole problem. 

Now pegs are in shape 0, 1, 2. Peg 0 is the training peg.

You can run experiment by

```
python launch_experiment.py ./configs/peginhole.json --gpu 0 --exp_id pegx
```

You can see the trained policy and make videos by

```
python sim_policy.py ./configs/peginhole.json ./videos ./output/peginhole --video --mode train --exp_id pegx
```

