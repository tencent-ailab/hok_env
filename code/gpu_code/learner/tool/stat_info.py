# -*- coding:utf-8 -*-
# plot figures of losses based on the file

import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

if len(sys.argv) != 2:
    print "please used %s %s" % ("stat_info.py", "filename")
    sys.exit(0)

train_info = sys.argv[1]


def remove_token(num):
    return float(num.strip(",[]:"))


# step1: get the training info
all_info = []
with open(train_info, "r") as f:
    for line_num, line in enumerate(f):
        if "loss infos of" in line:
            line = line.split(" ")
            run_step = remove_token(line[8])
            policy_loss = remove_token(line[9])
            value_loss = remove_token(line[10])
            entropy_loss = remove_token(line[11])
            reward = remove_token(line[12])
            all_info.append([run_step, policy_loss, value_loss, entropy_loss, reward])

run_step_list = [elem[0] for elem in all_info]
policy_loss_list = [elem[1] for elem in all_info]
value_loss_list = [elem[2] for elem in all_info]
entropy_loss_list = [elem[3] for elem in all_info]
reward_list = [elem[4] for elem in all_info]

# step2: plot the info
plt.subplot(4, 1, 1)
plt.plot(run_step_list, policy_loss_list, label="policy loss")
plt.ylabel("Policy loss")
plt.title("Training curve")

plt.subplot(4, 1, 2)
plt.plot(run_step_list, value_loss_list, label="value loss")
plt.ylabel("Value loss")

plt.subplot(4, 1, 3)
plt.plot(run_step_list, entropy_loss_list, label="entropy loss")
plt.ylabel("Entropy loss")

plt.subplot(4, 1, 4)
plt.plot(run_step_list, reward_list, label="Sampled Reward")
plt.ylabel("Sampled Reward")
plt.xlabel("training step")

# step3: save the figure
plt.savefig("training_loss" + ".png")
