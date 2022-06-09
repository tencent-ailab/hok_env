def get_mem_pool_key(server_ports, base_path):
    mem_pool_keys = []
    for port in server_ports:
        filepath = (
            base_path
            + "/mem_pool_server_p{}/etc/mem_pool_server_p{}_mcd0.conf".format(
                port, port
            )
        )
        with open(filepath, "r") as fin:
            for line in fin.readlines():
                line = line.strip("\n")
                pos = line.find("mem_pool_key")
                if pos > 0:
                    mem_pool_key = line.split("= ")[1]
                    mem_pool_keys.append(int(mem_pool_key))
    return mem_pool_keys


def get_mem_pool_param(param, server_ports, base_path):
    if len(server_ports) > 0:
        port = server_ports[0]
    else:
        port = 35200
    filepath = (
        base_path
        + "/mem_pool_server_p{}/etc/mem_pool_server_p{}_mcd0.conf".format(port, port)
    )
    print("mempool conf {}".format(filepath))
    with open(filepath, "r") as fin:
        for line in fin.readlines():
            line = line.strip("\n")
            pos = line.find(param)
            if pos > 0:
                mem_pool_param = line.split("= ")[1]
                return int(mem_pool_param)
