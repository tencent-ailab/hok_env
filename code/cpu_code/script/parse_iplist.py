import socket
import sys
import time

if __name__ == "__main__":
    src = sys.argv[1]
    dst = sys.argv[2]
    is_strict = int(sys.argv[3])
    iplist = []
    with open(src, "r") as f:
        for line in f.readlines():
            line = line.rstrip()
            vec = line.split(" ")
            hostname = vec[0]
            print(hostname)
            while True:
                try:
                    ip = socket.gethostbyname(hostname)
                    break
                except socket.error as error:
                    print("ip not found: %s" % (hostname))
                    sys.stdout.flush()
                    ip = hostname
                    if not is_strict:
                        break
                time.sleep(1)
            vec[0] = ip
            new_vec = []
            for w in vec:
                new_vec.append(w)
                new_vec.append(" ")
            iplist.append("".join(new_vec).rstrip())
    print(iplist)
    with open(dst, "w") as f:
        for ip in iplist:
            f.write("%s\n" % ip)
