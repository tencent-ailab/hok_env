# 20231228 v2.0.4
rl_framework:
1. refactor(logger): utilize logurus as logger 
    `rl_framework.common.logging` should be replaced by `from rl_framework.common.logging import logger as LOG`.
2. feat(model_manager): support `save_model_seconds`.
3. feat(model_manager): send checkpoints without optimizer state to reduce disk usage cost.
4. feat(send_model): support `backup_ckpt_only`.

aiarena:
1. fix(1v1/agent_demo): typos
2. feat(1v1/agent_demo): return home if ego_hp_rate is less than 0.5.
3. refactor(1v1/3v3): improve code and remove redundant configurations.
4. feat(actor): support `auto_bind_cpu` to bind cpu_id for each actor process according to actor_id.
5. feat(learner): support `load_optimizer_state`.
6. fix(3v3/model): typos

hok_env:
1. feat(3v3): support reward configuration.

Others:
1. Introduce GitHub workflow to upload Python package hok to pypi for every release.
2. Archive network.py for the 3v3 paper (cppo, mappo, ppo).
3. Use a torch-only image, tensorflow training code is now deprecated.
4. Update README.md.

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
