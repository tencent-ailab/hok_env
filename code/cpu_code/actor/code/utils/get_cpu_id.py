import re
import sys

if __name__ == "__main__":
    cpu_list = sys.argv[1].split(",")
    real_list = []
    for c in cpu_list:
        res1 = re.findall(r"([0-9]+)-([0-9]+)", c)
        res2 = re.findall(r"[0-9]+", c)
        if len(res1) == 1:
            for i in range(int(res1[0][0]), int(res1[0][1]) + 1):
                real_list.append(i)
        else:
            real_list.append(c)
    if len(sys.argv) == 3:
        print(real_list[int(sys.argv[2])])
    else:
        print(real_list)
