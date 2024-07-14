import sys
import os
import argparse
import numpy as np


def read_file(file_path):
    with open(file_path, 'r') as f:
        return [line.strip() for line in f.readlines()]


def construct_matrix(training_files):
    transitions = {}
    observations = {}
    pos = {}
    transitions_count = {}
    states = []
    for file_path in training_files:
        with open(file_path, 'r') as f:
            current_line = f.readline().strip()
            next_line = f.readline().strip()
            while current_line:
                current_word, current_pos = current_line.split()[0], current_line.split()[-1]
                observations_key = f'{current_word} | {current_pos}'
                observations[observations_key] = observations.get(observations_key, 0) + 1
                if current_pos not in pos:
                    pos[current_pos] = 1
                    states.append(current_pos)
                else:
                    pos[current_pos] += 1
                if next_line:
                    next_pos = next_line.split()[-1]
                    transitions_key = f'{current_pos} | {next_pos}'
                    transitions[transitions_key] = transitions.get(transitions_key, 0) + 1
                    if current_pos not in transitions_count:
                        transitions_count[current_pos] = 1
                    else:
                        transitions_count[current_pos] += 1
                current_line = next_line
                next_line = f.readline().strip()
    return transitions, observations, pos, transitions_count, states


def compute_probabilities(transitions, observations, pos, transitions_count):
    k = 1
    for obs_key in observations:
        word, pos_tag = obs_key.split(' | ')
        observations[obs_key] = (observations.get(obs_key, 0) + k) / (pos[pos_tag] + k * len(pos))
    for trans_key in transitions:
        pos1, pos2 = trans_key.split(' | ')
        transitions[trans_key] = (transitions.get(trans_key, 0) + k) / (transitions_count[pos1] + k * len(pos))
    return transitions, observations


def tag(training_files, test_file, output_file):
    transitions, observations, pos, transitions_count, states = construct_matrix(training_files)
    transitions, observations = compute_probabilities(transitions, observations, pos, transitions_count)
    test = read_file(test_file)
    state_sequence = []
    initial_word = test[0]
    max_prob, max_state = 0, None
    for s in range(len(states)):
        obs_key = f'{initial_word} | {states[s]}'
        if obs_key in observations:
            if observations[obs_key] > max_prob:
                max_prob = observations[obs_key]
                max_state = states[s]
    if not max_state:
        max_state = np.random.choice(states)
    state_sequence.append((initial_word, max_state, max_prob))
    for obs_word in test[1:]:
        new_state_sequence = []
        for s in range(len(states)):
            prev_state = state_sequence[-1][1]
            trans_key = f'{prev_state} | {states[s]}'
            obs_key = f'{obs_word} | {states[s]}'
            transition_prob = transitions.get(trans_key, 0)
            observation_prob = observations.get(obs_key, 0)
            prob = state_sequence[-1][2] * transition_prob * observation_prob
            new_state_sequence.append((obs_word, states[s], prob))
        max_prob = max([s[2] for s in new_state_sequence])
        max_state = [s[1] for s in new_state_sequence if s[2] == max_prob][0]
        state_sequence.append((obs_word, max_state, max_prob))
    with open(output_file, 'w') as f:
        for obs_word, pos, prob in state_sequence:
            f.write(f'{obs_word} : {pos}\n')




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--trainingfiles",
        action="append",
        nargs="+",
        required=True,
        help="The training files."
    )
    parser.add_argument(
        "--testfile",
        type=str,
        required=True,
        help="One test file."
    )
    parser.add_argument(
        "--outputfile",
        type=str,
        required=True,
        help="The output file."
    )
    args = parser.parse_args()
    training_list = args.trainingfiles[0]
    print("training files are {}".format(training_list))
    print("test file is {}".format(args.testfile))
    print("output file is {}".format(args.outputfile))
    print("Starting the tagging process.")
    tag(training_list, args.testfile, args.outputfile)
    print("Finished the tagging process.")
    #construct_matrix(training_list)
