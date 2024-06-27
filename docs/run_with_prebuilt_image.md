## Run with pre-built image

> Please refer to [cluster.md] (./cluster.md) for the linux-only tutorial

1. Start gamecore server on windows

    ```
    cd gamecore
    gamecore-server.exe server --server-address :23432
    ```

2. Start a container

    ```
    docker run -it -p 35000-35400:35000-35400 tencentailab/hok_env:cpu_v2.0.1 bash
    ```

3. set the gamecore

    ```
    export GAMECORE_SERVER_ADDR="host.docker.internal:23432"
    ```
4. Run the `test_env.py`

    ```
    cd /hok_env/hok/hok1v1/unit_test
    python3 test_env.py
    ```

5. Start a job

    1. Start learner

        ```
        cd /aiarena/scripts/learner/
        sh start_learner.sh
        ```

    2. Start actor

        ```
        # if you run the actor in a same node with the learner, no need to start the model pool again
        export NO_MODEL_POOL=1

        cd /aiarena/scripts/actor/
        sh start_actor.sh
        ```

    3. Check the log

        1. Learner

            * train.log

              ```
              # tail /aiarena/logs/learner/train.log 
              t shape (1024, 32)
              2022-12-02 19:22:57.398643: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
              2022-12-02 19:22:57.411737: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2494140000 Hz
              2022-12-02 19:22:57.412222: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x72c1980 executing computations on platform Host. Devices:
              2022-12-02 19:22:57.412253: I tensorflow/compiler/xla/service/service.cc:175]   StreamExecutor device (0): <undefined>, <undefined>
              2022-12-02 19:22:58.073725: W tensorflow/compiler/jit/mark_for_compilation_pass.cc:1412] (One-time warning): Not using XLA:CPU for cluster because envvar TF_XLA_FLAGS=--tf_xla_cpu_global_jit was not set.  If you want XLA:CPU, either set that envvar, or use experimental_jit_scope to enable XLA:CPU.  To confirm that XLA is active, pass --vmodule=xla_compilation_cache=1 (as a proper command-line flag, not via TF_XLA_FLAGS) or set the envvar XLA_FLAGS=--xla_hlo_profile.
              [Debug] The sample is less than half the capacity 0 5000
              [Debug] The sample is less than half the capacity 0 5000
              [Debug] The sample is less than half the capacity 1364 5000
              [Debug] The sample is less than half the capacity 1364 5000
              # tail /code/logs/gpu_log/train.log 
              2022-12-02 19:22:48 log_manager.py[line:152] DEBUG init starting
              2022-12-02 19:22:49 log_manager.py[line:152] DEBUG init finished
              2022-12-02 19:22:56 log_manager.py[line:152] DEBUG local_save_model_secs: 0
              2022-12-02 19:24:59 log_manager.py[line:152] DEBUG Start training...
              2022-12-02 19:31:02 log_manager.py[line:77] DEBUG step: 200 images/sec mean = 39.6 recv_sample/sec = 27 total_loss: 0.011530144 noise_scale: 1.00 batch_noisescale: 64.00 mean noise scale = 0.005000
              2022-12-02 19:36:25 log_manager.py[line:77] DEBUG step: 400 images/sec mean = 39.6 recv_sample/sec = 26 total_loss: -0.4639552 noise_scale: 1.00 batch_noisescale: 64.00 mean noise scale = 0.005000
              2022-12-02 19:41:48 log_manager.py[line:77] DEBUG step: 600 images/sec mean = 39.6 recv_sample/sec = 28 total_loss: 1.4184176 noise_scale: 1.00 batch_noisescale: 64.00 mean noise scale = 0.005000
              ```

            * loss.txt

              ```
              # tail /aiarena/logs/learner/loss.txt 
              {"role": "learner", "ip_address": "127.0.0.1", "step": "200", "timestamp": "12/02/2022-19:31:02", "info_map": "{'loss': 0.011530144, 'value_cost': 0.37531283, 'entropy_cost': -11.996081, 'policy_cost': -0.06388064}"}
              {"role": "learner", "ip_address": "127.0.0.1", "step": "400", "timestamp": "12/02/2022-19:36:25", "info_map": "{'loss': -0.4639552, 'value_cost': 0.29535705, 'entropy_cost': -12.003362, 'policy_cost': -0.4592282}"}
              {"role": "learner", "ip_address": "127.0.0.1", "step": "600", "timestamp": "12/02/2022-19:41:48", "info_map": "{'loss': 1.4184176, 'value_cost': 1.0052822, 'entropy_cost': -11.962294, 'policy_cost': 0.71219283}"}
              ```

        2. Actor

            ```
            # tail /aiarena/logs/actor/actor_0.log 
            [{'hero': 'luban', 'skill': 'frenzy'}, {'hero': 'luban', 'skill': 'frenzy'}]
            I1202 19:41:37.569345 140293844768576 actor.py:145] [{'hero': 'luban', 'skill': 'frenzy'}, {'hero': 'luban', 'skill': 'frenzy'}]
            game not end, send close game at first 6962
            game not end, send close game at first 6962
            check game stopped.
            I1202 19:42:45.335792 140293844768576 actor.py:247] Tower ActorType.ACTOR_TOWER in camp Camp.PLAYERCAMP_1, hp: 0
            I1202 19:42:45.336180 140293844768576 actor.py:247] Tower ActorType.ACTOR_TOWER in camp Camp.PLAYERCAMP_2, hp: 5000
            I1202 19:42:45.336433 140293844768576 actor.py:247] Tower ActorType.ACTOR_CRYSTAL in camp Camp.PLAYERCAMP_1, hp: 0
            I1202 19:42:45.336676 140293844768576 actor.py:247] Tower ActorType.ACTOR_CRYSTAL in camp Camp.PLAYERCAMP_2, hp: 7000
            I1202 19:42:45.475080 140293844768576 actor.py:299] ==================================================
            I1202 19:42:45.475494 140293844768576 actor.py:300] game_id : b'127.0.0.1_28840_0_20221202194137_560'
            I1202 19:42:45.476910 140293844768576 actor.py:309] aiprocess_process | sum: 29435.193 mean:6.477815360915493 max:117.378 times:4544
            I1202 19:42:45.477501 140293844768576 actor.py:309] sample_manger_format_data | sum: 78.99 mean:78.99 max:78.99 times:1
            I1202 19:42:45.478820 140293844768576 actor.py:309] agent_process | sum: 29992.633 mean:13.200982834507043 max:122.47 times:2272
            I1202 19:42:45.479929 140293844768576 actor.py:309] step | sum: 36249.837 mean:15.95503389084507 max:52.794 times:2272
            I1202 19:42:45.480857 140293844768576 actor.py:309] save_sample | sum: 362.001 mean:0.15933142605633802 max:6.798 times:2272
            I1202 19:42:45.481554 140293844768576 actor.py:309] one_frame | sum: 66409.802 mean:29.229666373239436 max:143.894 times:2272
            I1202 19:42:45.481947 140293844768576 actor.py:309] one_episode | sum: 67905.702 mean:67905.702 max:67905.702 times:1
            I1202 19:42:45.482166 140293844768576 actor.py:309] reset | sum: 1293.392 mean:1293.392 max:1293.392 times:1
            I1202 19:42:45.482297 140293844768576 actor.py:312] ==================================================
            I1202 19:42:45.482467 140293844768576 actor.py:325] Agent is_main:False, type:network, camp:Camp.PLAYERCAMP_1,reward:-49.497, win:-1, win_112:-1,h_act_rate:1.0
            I1202 19:42:45.482620 140293844768576 actor.py:334] Agent is_main:False, money_per_frame:0.07, kill:0, death:0, hurt_pf:0.06
            I1202 19:42:45.482745 140293844768576 actor.py:325] Agent is_main:False, type:network, camp:Camp.PLAYERCAMP_2,reward:54.759, win:1, win_112:1,h_act_rate:1.0
            I1202 19:42:45.482844 140293844768576 actor.py:334] Agent is_main:False, money_per_frame:0.55, kill:0, death:0, hurt_pf:7.77
            I1202 19:42:45.482941 140293844768576 actor.py:368] game info length:6964
            I1202 19:42:45.483038 140293844768576 actor.py:370] ==================================================
            ```
