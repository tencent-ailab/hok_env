import sys

current_iplist = open(sys.argv[1]).readlines()
mem_pool_iplist = open(sys.argv[2]).readlines()
# docker_id = np.random.randint(0,4)

ip, user, port, core = current_iplist[0].strip().split()
mempool_num = len(mem_pool_iplist)
