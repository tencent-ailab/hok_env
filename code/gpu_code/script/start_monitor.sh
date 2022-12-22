# setcap 'cap_net_bind_service=+ep' /usr/sbin/grafana-server
# iptables -t nat -A PREROUTING -p tcp --dport 80 -j REDIRECT --to-port 8081

echo Start service... 

# dev 场景不使用influxdb
if [[ -n "$KAIWU_DEV" ]]; then
    "Disable influxdb"
    exit 0
fi

if [[ -z "$NOT_USE_INFLUXDB_EXPORTER" ]];
then
    nohup influxdb_exporter --web.listen-address=":8086"  --udp.bind-address=":8086" > /dev/null 2>&1 &
else
    pgrep influxdb
    if [ $? == 1 ];then
    	nohup /usr/bin/influxd > /dev/null 2>&1 &
            while true; do
                lsof -i :8086 && break
                sleep 1
            done
    	curl -i -XPOST http://localhost:8086/query --data-urlencode "q=CREATE DATABASE monitordb"
    fi
    
    pgrep grafana
    if [ $? == 1 ];then
    	cd /usr/share/grafana
    	nohup /usr/sbin/grafana-server --config=/etc/grafana/grafana.ini cfg:default.paths.provisioning=/etc/grafana/provisioning > /dev/null 2>&1 &
    fi
fi

echo Complete!
