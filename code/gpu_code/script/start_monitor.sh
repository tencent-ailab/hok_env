setcap 'cap_net_bind_service=+ep' /usr/sbin/grafana-server
iptables -t nat -A PREROUTING -p tcp --dport 80 -j REDIRECT --to-port 8081

echo Start service... 
sleep 1

pgrep influxdb
if [ $? == 1 ];then
	nohup /usr/bin/influxd -config /etc/influxdb/influxdb.conf > /dev/null 2>&1 &
        sleep 1
	curl -i -XPOST http://localhost:8086/query --data-urlencode "q=CREATE DATABASE monitordb"
fi

pgrep grafana
if [ $? == 1 ];then
	cd /usr/share/grafana
	nohup /usr/sbin/grafana-server --config=/etc/grafana/grafana.ini --pidfile=/var/run/grafana/grafana-server.pid --packaging=rpm cfg:default.paths.logs=/var/log/grafana cfg:default.paths.data=/var/lib/grafana cfg:default.paths.plugins=/var/lib/grafana/plugins cfg:default.paths.provisioning=/etc/grafana/provisioning > /dev/null 2>&1 &
fi

echo Complete!
