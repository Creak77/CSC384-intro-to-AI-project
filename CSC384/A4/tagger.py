import sys
import os
import argparse
import random


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
            lines = f.readlines()
            lines = [line.strip() for line in lines if line.strip()]

            for i in range(len(lines) - 1):
                current_line = lines[i]
                next_line = lines[i + 1]

                current_word, current_pos = current_line.split()[0], current_line.split()[-1]
                observations_key = f'{current_word} | {current_pos}'
                observations[observations_key] = observations.get(observations_key, 0) + 1

                if current_pos not in pos:
                    pos[current_pos] = 1
                    states.append(current_pos)
                else:
                    pos[current_pos] += 1

                next_pos = next_line.split()[-1]
                transitions_key = f'{current_pos} | {next_pos}'
                transitions[transitions_key] = transitions.get(transitions_key, 0) + 1

                if current_pos not in transitions_count:
                    transitions_count[current_pos] = 1
                else:
                    transitions_count[current_pos] += 1

    return transitions, observations, pos, transitions_count, states


def compute_probabilities(transitions, observations, pos, transitions_count):
    k = 1
    for obs_key in observations:
        word, pos_tag = obs_key.split(' | ')
        observations[obs_key] = (observations.get(obs_key, 0) + k) / (pos[pos_tag] + k * len(pos))
    for trans_key in transitions:
        pos1, pos2 = trans_key.split(' | ')
        transitions[trans_key] = (transitions.get(trans_key, 0) + k) / (transitions_count[pos1] + k * len(pos))
    print(transitions)
    return transitions, observations


def tag(training_files, test_file, output_file):
    transitions, observations, pos, transitions_count, states = construct_matrix(training_files)
    transitions, observations = compute_probabilities(transitions, observations, pos, transitions_count)
    test = read_file(test_file)
    state_sequence = []
    max_prob, max_state = 0, None
    initial_word = test[0]
    for s in range(len(states)):
        obs_key = f'{initial_word} | {states[s]}'
        if obs_key in observations:
            if observations[obs_key] > max_prob:
                max_prob = observations[obs_key]
                max_state = states[s]
    if not max_state:
        max_state = random.choice(states)
    state_sequence.append((initial_word, max_state))
    for obs_word in test[1:]:
        #print(obs_word)
        prob = []
        for s in range(len(states)):
            prev_state = state_sequence[-1][1]
            trans_key = f'{prev_state} | {states[s]}'
            obs_key = f'{obs_word} | {states[s]}'
            transition_prob = transitions.get(trans_key, 0)
            observation_prob = observations.get(obs_key, 0)
            prob.append(transition_prob * observation_prob)
        max_prob = max(prob)
        max_state = states[prob.index(max_prob)]
        state_sequence.append((obs_word, max_state))
    with open(output_file, 'w') as f:
        for i, (obs_word, pos) in enumerate(state_sequence):
            if not obs_word and (i == 0 or state_sequence[i-1][0] == ""):
                continue
            elif not pos:
                continue
            elif not obs_word:
                f.write('\n')
            else:
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
    #construct_matrix(training_list)

    count = 0
    total = 0
    with open("training1.txt") as file1, open("output.txt") as file2:
        for line1, line2 in zip(file1, file2):
            total += 1
            if line1.strip() == line2.strip():
                count += 1

    print(count / total)

