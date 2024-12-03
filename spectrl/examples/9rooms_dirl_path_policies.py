from spectrl.hierarchy.construction import automaton_graph_from_spec
from spectrl.hierarchy.reachability import HierarchicalPolicy, ConstrainedEnv
from spectrl.main.spec_compiler import ev, seq, choose, alw
from spectrl.util.io import parse_command_line_options, save_log_info
from spectrl.util.rl import print_performance, get_rollout
from spectrl.rl.ars import HyperParams

from spectrl.examples.rooms_envs import GRID_PARAMS_LIST, MAX_TIMESTEPS, START_ROOM, FINAL_ROOM
from spectrl.envs.rooms import RoomsEnv

import os

# num_iters = [50, 100, 200, 300, 400, 500]
num_iters = [2]

# Construct Product MDP and learn policy
if __name__ == '__main__':
    flags = parse_command_line_options()
    render = flags['render']
    env_num = flags['env_num']
    folder = flags['folder']
    itno = flags['itno']
    spec_num = flags['spec_num']

    log_info = []

    for i in num_iters:

        grid_params = GRID_PARAMS_LIST[env_num]

        hyperparams = HyperParams(30, i, 30, 15, 0.05, 0.3, 0.15)

        print('\n**** Learning Policy for Spec #{} in Env #{} ****'.format(spec_num, env_num))

        # Step 1: initialize system environment
        system = RoomsEnv(grid_params, START_ROOM[env_num], FINAL_ROOM[env_num])

        # Step 4: List of specs.
        if env_num == 2:
            bottomright = (0, 2)
            topleft = (2, 0)
        if env_num == 3 or env_num == 4:
            bottomright = (0, 3)
            topleft = (3, 0)

        # test specs
        spec0 = ev(grid_params.in_room(FINAL_ROOM[env_num]))
        spec1 = seq(ev(grid_params.in_room(FINAL_ROOM[env_num])), ev(
            grid_params.in_room(START_ROOM[env_num])))
        spec2 = ev(grid_params.in_room(topleft))

        # Goto destination, return to initial
        spec3 = seq(ev(grid_params.in_room(topleft)), ev(grid_params.in_room(START_ROOM[env_num])))
        # Choose between top-right and bottom-left blocks (Same difficulty - learns 3/4 edges)
        spec4 = choose(ev(grid_params.in_room(bottomright)),
                       ev(grid_params.in_room(topleft)))
        # Choose between top-right and bottom-left, then go to Final state (top-right).
        # Only one path is possible (learns 5/5 edges. Should have a bad edge)
        spec5 = seq(choose(ev(grid_params.in_room(bottomright)),
                           ev(grid_params.in_room(topleft))),
                    ev(grid_params.in_room(FINAL_ROOM[env_num])))
        # Add obsacle towards topleft
        spec6 = alw(grid_params.avoid_center((1, 0)), ev(grid_params.in_room(topleft)))
        # Either go to top-left or bottom-right. obstacle on the way to top-left.
        # Then, go to Final state. Only one route is possible
        spec7 = seq(choose(alw(grid_params.avoid_center((1, 0)), ev(grid_params.in_room(topleft))),
                           ev(grid_params.in_room(bottomright))),
                    ev(grid_params.in_room(FINAL_ROOM[env_num])))

        specs = [spec0, spec1, spec2, spec3, spec4, spec5, spec6, spec7]

        # Step 3: construct abstract reachability graph
        _, abstract_reach = automaton_graph_from_spec(specs[spec_num])
        print('\n**** Abstract Graph ****')
        abstract_reach.pretty_print()

        # Step 5: Learn policy
        path_policies = abstract_reach.learn_all_paths(
            system, hyperparams, res_model=None, max_steps=20, render=render,
            neg_inf=-100, safety_penalty=-1, num_samples=500)

