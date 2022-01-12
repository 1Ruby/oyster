# PEARL: Peg-in-hole

The repo is forked from https://github.com/katerakelly/oyster, focusing on peg-in-hole problem. 

Now pegs are in shape 0, 1, 2, 3, 4, each with 2 holes with different tolerance(1mm and 2mm).

By defaults, the training set is peg 0-4 with large tolerance and peg 0,1,2 with small tolerance, and the evaluating set is peg 3,4 with small tolerance.

Peg and holes are stored as .stl and .obj in peginhole/assets.

You can run experiment by

```
python launch_experiment.py ./configs/peginhole.json --gpu 0 --exp_id pegx
```

You can see the trained policy and make videos by

```
python sim_policy.py ./configs/peginhole.json ./videos ./output/peginhole --video --mode train --exp_id pegx
```

