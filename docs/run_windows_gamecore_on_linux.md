# Quick

1. Download the latest gamecore (>= 20230110) that resolves compatibility issues with Wine

2. Install Wine

    See also: https://wiki.winehq.org/Download_zhcn

3. Execute gamecore using Wine

    ```
    export GAMECORE_PATH=${GAMECORE_PATH:-"/rl_framework/gamecore/"}
    export WINEPATH="${GAMECORE_PATH}/lib/;${GAMECORE_PATH}/bin/"

    wine ${GAMECORE_PATH}/bin/sgame_simulator_remote_zmq.exe simulator.conf
    ```

    See also: [sgame_simulator_remote_zmq_wine](../code/remote-gc-server/sgame_simulator_remote_zmq)

# Construct the gamecore image and perform tests in the container

## Build gamecore image

1. Install Docker

    See also: https://www.docker.com/

2. Download gamecore from [website](https://aiarena.tencent.com/aiarena/zh/open-gamecore) and save as `hok_env_gamecore.zip`

3. [Apply for a license](https://aiarena.tencent.com/aiarena/zh/open-gamecore) and save it as `license.dat`

    The working directory should now appear as follows:
    ```
    # tree -L 1 hok_env
    hok_env
    ├── aiarena
    ├── dockerfile
    ├── docs
    ├── hok_env
    ├── hok_env_gamecore.zip  <-- Manually download
    ├── license.dat           <-- Manually download
    ├── README.md
    ├── rl_framework
    └── ...
    ```

4. Construct the gamecore image with a prebuilt CPU image

    ```
    base_image=tencentailab/hok_env:cpu_v2.0.1
    target_image=gamecore

    docker build -f ./dockerfile/dockerfile.gamecore -t ${target_image} --build-arg=BASE_IMAGE=${base_image} .
    ```

## Test the gamecore image

1. Launch a gamecore server container

    ```
    docker run --network host -e SIMULATOR_USE_WINE=1 -it gamecore bash
    ```

    > `--network host` means exposing the gamecore server service to the host network

    > `-e SIMULATOR_USE_WINE=1` means using Wine to run the gamecore

    > To run the gamecore-server container in the background and access it via IP address (default: '127.0.0.0:23432'), use:
    > ```
    > docker run -d --name gamecore --network host -e SIMULATOR_USE_WINE=1 -it gamecore sh /rl_framework/remote-gc-server/run_and_monitor_gamecore_server.sh
    > ```

2. Initiate the gamecore server running in the background

    ```
    sh /rl_framework/remote-gc-server/start_gamecore_server.sh
    ```

    You may receive an output similar to this:
    ```
    COMMAND   PID USER   FD   TYPE     DEVICE SIZE/OFF NODE NAME
    gamecore-  34 root    7u  IPv6 1293702323      0t0  TCP *:23432 (LISTEN)
    ```

    Obtain the gamecore server log:
    ```
    cat /aiarena/logs/gamecore-server.log
    ```

    ```
    2023/01/10 16:54:06 maxprocs: Leaving GOMAXPROCS=8: CPU quota undefined
    [GIN-debug] [WARNING] Running in "debug" mode. Switch to "release" mode in production.
     - using env:   export GIN_MODE=release
     - using code:  gin.SetMode(gin.ReleaseMode)
    
    [GIN-debug] POST   /v1/newGame               --> server.handler.func1 (4 handlers)
    [GIN-debug] POST   /v1/stopGame              --> server.handler.func1 (4 handlers)
    [GIN-debug] POST   /v1/list                  --> server.handler.func1 (4 handlers)
    [GIN-debug] POST   /v1/download              --> server.getHandler.func1 (4 handlers)
    [GIN-debug] POST   /v1/check                 --> server.handler.func1 (4 handlers)
    [GIN-debug] GET    /v1/status                --> server.Server.Status-fm (4 handlers)
    [GIN-debug] POST   /v1/exit                  --> server.getHandler.func1 (4 handlers)
    [GIN-debug] PUT    /v1/token                 --> server.getHandler.func1 (4 handlers)
    [GIN-debug] POST   /v2/newGame               --> server.getHandler.func1 (4 handlers)
    [GIN-debug] POST   /v2/stopGame              --> server.getHandler.func1 (4 handlers)
    [GIN-debug] POST   /v2/exists                --> server.getHandler.func1 (4 handlers)
    time="2023-01-10T16:54:06+08:00" level=info msg="start server" Address=":23432" ApiSvcBaseURL= DisableV2api=false Env=dev GamecoreMgr="map[CallBack:[map[] map[Failed:0 Succeed:0 Total:0] map[] map[SaveRate:0.5 SceneManager:map[ABSDir:/rl_framework/gamecore/scene Lock:map[] MaxABSNum:1000 SceneFile:/rl_framework/gamecore/scene/scene.json]] map[BackupDir:/rl_framework/gamecore/simulator_output Lock:map[] MaxFileNum:4000] map[]] CoreAssets:/rl_framework/gamecore/core_assets GameStore:map[] Preprocessor:map[remote:map[InitABSFile1v1:/rl_framework/gamecore/scene/1V1.abs InitABSFile3v3:/rl_framework/gamecore/scene/3V3.abs InitABSFile5v5:/rl_framework/gamecore/scene/5V5.abs SimulatorRemoteBin:/rl_framework/remote-gc-server/sgame_simulator_remote_zmq] remote_repeat:map[Remote:map[InitABSFile1v1:/rl_framework/gamecore/scene/1V1.abs InitABSFile3v3:/rl_framework/gamecore/scene/3V3.abs InitABSFile5v5:/rl_framework/gamecore/scene/5V5.abs SimulatorRemoteBin:/rl_framework/remote-gc-server/sgame_simulator_remote_zmq] RemoteRate:0.2 Repeat:map[SceneManager:map[ABSDir:/rl_framework/gamecore/scene Lock:map[] MaxABSNum:1000 SceneFile:/rl_framework/gamecore/scene/scene.json] SimulatorRepeatBin:/rl_framework/remote-gc-server/sgame_simulator_repeated_zmq]] repeat:map[SceneManager:map[ABSDir:/rl_framework/gamecore/scene Lock:map[] MaxABSNum:1000 SceneFile:/rl_framework/gamecore/scene/scene.json] SimulatorRepeatBin:/rl_framework/remote-gc-server/sgame_simulator_repeated_zmq]] SimulatorLib:/rl_framework/gamecore/lib]" SGameDir=ai_simulator_remote/ SGameFileName=sgame_simulator_remote_zmq.exe Token=
    ```

3. Send a built-in rule-based AI request to the gamecore server

    ```
    python3 /rl_framework/remote-gc-server/test_client.py
    ```
    ```
    Success {'X-Request-ID': 'f86f66a9-69d5-4041-a6f8-31b3ea66cc60'}
    ```

    Now you can view the `simulator.exe` process
    ```
    ps -e f
    ```
    ```
    PID TTY      STAT   TIME COMMAND
      1 pts/0    Ss     0:00 bash
     12 pts/0    S      0:00 sh /rl_framework/remote-gc-server/run_and_monitor_gamecore_server.sh
     15 pts/0    S      0:00  \_ sh /rl_framework/remote-gc-server/monitor_defunct.sh
    182 pts/0    S      0:00  |   \_ sleep 10
     24 pts/0    S      0:00  \_ bash /rl_framework/remote-gc-server/run_gamecore_server.sh
     27 pts/0    Sl     0:00      \_ ./gamecore-server-linux-amd64 server --server-address=:23432 --simulator-remote-bin /rl_framework/remote-gc-server/sgame_simulator_remote_zmq --simulator-repeat-bin /rl_framework/remote-gc-s
     97 pts/0    S      0:00          \_ /usr/bin/bash /rl_framework/remote-gc-server/sgame_simulator_remote_zmq kaiwu-test-runtime-id-0-1673340926220678557-341.conf
    103 pts/0    S      0:00              \_ start.exe /exec /rl_framework/gamecore/bin/sgame_simulator_remote_zmq.exe
    105 ?        Ss     0:02 /opt/wine-stable/bin/wineserver
    109 ?        Ssl    0:00 C:\windows\system32\services.exe
    114 pts/0    S      0:00 C:\windows\system32\explorer.exe /desktop
    129 ?        Sl     0:00 C:\windows\system32\winedevice.exe
    151 ?        Sl     0:00 C:\windows\system32\winedevice.exe
    163 pts/0    Rl     0:11 Z:\rl_framework\gamecore\bin\sgame_simulator_remote_zmq.exe kaiwu-test-runtime-id-0-1673340926220678557-341.conf.new
    183 pts/0    R+     0:00 ps -e f
    ```

    Obtain the game log:
    ```
    # cat /rl_framework/gamecore/simulator_output/*.log
    ```
    ```
    + python3 /rl_framework/remote-gc-server/process.py kaiwu-test-runtime-id-0-1673340926220678557-341.conf kaiwu-test-runtime-id-0-1673340926220678557-341.conf.new /tmp/kaiwu-test-runtime-id-0-1673340926220678557-3412873246842
    + GAMECORE_PATH=/rl_framework/gamecore
    + export 'WINEPATH=/rl_framework/gamecore/lib/;/rl_framework/gamecore/bin/'
    + WINEPATH='/rl_framework/gamecore/lib/;/rl_framework/gamecore/bin/'
    + mkdir -p /.wine/
    + export WINEPREFIX=/.wine/test-runtime-id-0
    + WINEPREFIX=/.wine/test-runtime-id-0
    + wine /rl_framework/gamecore/bin/sgame_simulator_remote_zmq.exe kaiwu-test-runtime-id-0-1673340926220678557-341.conf.new
    wine: created the configuration directory '/.wine/test-runtime-id-0'
    wine: configuration in L"/.wine/test-runtime-id-0" has been updated.
    Cygwin WARNING:
      Couldn't compute FAST_CWD pointer.  This typically occurs if you're using
      an older Cygwin version on a newer Windows.  Please update to the latest
      available Cygwin version from https://cygwin.com/.  If the problem persists,
      please see https://cygwin.com/problems.html
    
    PlayerNum:2
    AbsPath:../../rl_framework/gamecore/scene/1V1.abs
    PlayerInfo [CampID:0][HeroID:139][Skill:80104][AutoAi:1][AiServer::0:100] [Symbol 0 0 0] [Request:-1]
    PlayerInfo [CampID:1][HeroID:139][Skill:80104][AutoAi:1][AiServer::0:100] [Symbol 0 0 0] [Request:-1]
    SGame Simulator Begin
    init_ret:0
    start_ret:0
    Hero[0] inHeroId:139; outPlayerId:148
    Hero[1] inHeroId:139; outPlayerId:149
    [Hero Info] [HeroID:139] [RuntimeID:8] client_id:0.0.0.0_1020_0_20230110165534_148
    [Hero Info] [HeroID:139] [RuntimeID:9] client_id:0.0.0.0_1020_0_20230110165534_149
    boost_ret finished: 8, gameover_ai_server: 0
    close_ret:0
    uninit_ret:0
    SGame Simulator End [FrameNum:12152][TimeUsed:15009ms]
    ```

    Get the output abs file:
    ```
    # ll /rl_framework/gamecore/simulator_output/*.abs
    ```

    ```
    -rw-r--r-- 1 root root 529613 Jan 10 16:55 /rl_framework/gamecore/simulator_output/AIOSS_230110-1655_linux_1450111_1450123_1_1_20001_kaiwu-test-runtime-id-0-1673340926220678557-341.abs
    ```

    > Note: You can download the ABS file to Windows and utilize the replay tool to view the game video.
    > For more details, please refer to [Replay Tool](https://github.com/tencent-ailab/hok_env/tree/master#replay-tool).

4. Run `test_env.py`

    - 1v1

        ```
        cd /hok_env/hok/hok1v1/unit_test
        python3 ./test_env.py
        ```

    - 3v3

        ```
        python3 -c "from hok.hok3v3.unit_test.test_env import run_test; run_test()"
        ```
