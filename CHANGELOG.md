# 20230817

1. Refactor aiarena/hok_env/rl_framework
2. Support Python 3.6/3.8/3.9 for hok3v3
3. Update config.dat for hok1v1/hok3v3 to support more heroes
4. Add aiarena/process to run the rl_framework with python
5. Fix bugs hok1v1/lib, hok3v3/lib)

# 20230607

3v3 mode now available(python 3.7 required)

1. Refactor hok_env: renamed hok -> hok1v1
2. Support pytorch in rl_framework
3. Support 3v3 mode: hok3v3 added
    1. Run the unit test
    ```
    python -c "from hok.hok3v3.unit_test.test_env import run_test; run_test()"
    ```
4. Example: 3v3 dev image
    1. build image
    ```
    docker build -t test -f dockerfile/dockerfile.dev.3v3 .
    ```
    2. run train test (start gamecore server on `127.0.0.1:23432` before the test)
    ```
    docker run -it --network host test bash
    sh /aiarena/scripts/start_test.sh
    ```

# 20230110

Support running Windows gamecore on Linux using Wine:

1. Update gamecore to fix the compatibility with the Wine

2. Add gamecore-server-linux-amd64 to the gamecore pcakge

3. Add remote gamecore server
    ```
    export SIMULATOR_USE_WINE=1
    nohup sh /rl_framework/remote-gc-server/run_and_monitor_gamecore_server.sh &
    ```

4. Update dockerfile: use ubuntu as the base image
    ```
    sh ./build.sh
    ```
    See also [Github Action](./.github/workflows/)

5. Sync codes
    1. Support `SLOW_TIME`
    2. Fix `NET_CARD_NAME`
    3. remote check_and_send
    4. Fix typos
    5. Fix the zmq server bind error
    6. Wait for the gamecore process done after `gameover`

6. Update hok_env/hok/lib/interface
    1. Remove the init move
    2. Support Python3.8 and Python3.9
