# -*- coding: utf-8 -*-

from rl_framework.mem_pool import MemPoolAPIs
from rl_framework.mem_pool import SamplingStrategy
from rl_framework.common import get_host_ip


def test_push_samples():
    local_ip = get_host_ip()
    api = MemPoolAPIs(local_ip, 35201, "mcp")
    array = []
    priorities = []
    for i in range(1, 5):
        array.append(bytes("hello world%s" % (i), encoding="utf8"))
        priorities.append(float(i))
    ret = api.push_samples(array, priorities)
    print("push samples: ret", ret[8:])


def test_pull_sample():
    local_ip = get_host_ip()
    api = MemPoolAPIs(local_ip, 35201, "mcp")
    for _ in range(5):
        seq, sample = api.pull_sample(SamplingStrategy.PriorityGet.value)
        print("get sample: seq %s sample %s" % (seq, sample))


test_push_samples()
print()
test_pull_sample()
