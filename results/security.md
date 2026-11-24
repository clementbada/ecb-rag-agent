# Prompt-injection test

8 attacks (5 direct, 3 indirect through poisoned passages added to an in-memory copy of the index), each run with the output guard off and on. 3 attacks were written after the first run and were not used to design the guard.

| output_guard   |   attacks |   attack_success_rate |   direct_success |   indirect_success |   new_attacks_success |
|:---------------|----------:|----------------------:|-----------------:|-------------------:|----------------------:|
| off            |         8 |                 0.250 |            0.400 |              0.000 |                 0.333 |
| on             |         8 |                 0.000 |            0.000 |              0.000 |                 0.000 |
