import requests

url = "http://127.0.0.1:23432/v2/newGame"
headers = {
    "Content-Type": "application/json",
}
data = {
    "simulator_type": "remote_repeat",
    "runtime_id": "test-runtime-id-0",
    "simulator_config": {
        "game_mode": "1v1",
        "hero_conf": [
            {
                "hero_id": 139,
            },
            {
                "hero_id": 139,
            },
        ]
    },
}

resp = requests.post(url=url, json=data, headers=headers, verify=False)
if resp.ok:
    ret = resp.json()
    print("Success", ret)
else:
    print("Failed", resp)
