# Cluster Training

This tutorial explains how to manually run a HoK3v3 cluster training on two nodes.

## Quick Steps

1. Launch the gamecore-server and obtain the gamecore-server address, such as `IP:23432`.

    For more information, refer to [run_windows_gamecore_on_linux](./run_windows_gamecore_on_linux.md).

2. Install `hok_env` and `rl_framework` from this Git repository.

    For further details, please see [dockerfile.dev](../dockerfile/dockerfile.dev).

3. Configure `rl_framework` and initiate the training process.

    1. Prepare the directory path:

        1. Link `hok_env/aiarena/` to `/aiarena/`

        2. Link `hok_env/aiarena/3v3` to `/aiarena/code`

        3. Link `hok_env/rl_framework` to `/rl_framework/`

    2. Start the Learner:

        ```
        # Configure the learner IP
        echo "learner_ip root 36000 1" > /aiarena/scripts/learner/learner.iplist

        # Launch the learner
        sh /aiarena/scripts/learner/start_learner.sh
        ```

        > Replace `learner_ip` with your node IP.

    3. Start the Actor:

        ```
        # Configure the learner IP
        echo "learner_ip root 36000 1" > /aiarena/scripts/actor/learner.iplist

        # Set the number of actor processes (one actor per CPU core is recommended)
        export ACTOR_NUM=6

        # Set the gamecore server address
        export GAMECORE_SERVER_ADDR=IP:23432

        # if you run the actor in a same node(container) with the learner, no need to start the model pool again
        export NO_MODEL_POOL=1

        # Launch actor processes
        sh /aiarena/scripts/actor/start_actor.sh
        ```

        > Replace `IP:23432` with your gamecore server address.
        >
        > Replace `learner_ip` with your node IP.

4. Examine the logs:

    ```
    # For the learner container
    /aiarena/logs/learner/train.log
    
    # For the gamecore server container
    /aiarena/logs/gamecore-server.log
    
    # For the actor container
    /aiarena/logs/actor/actor_*.log
    ```

## Requirements

- Docker

> Docker is the only prerequisite. Build the image to install dependencies specified in the Dockerfile ([Dockerfile](../dockerfile)).
>
> The image encompasses all necessary components, so simply initiate the container with the built image and execute the training code.

To utilize a container with GPU, the following are required:

- [Nvidia Docker](https://github.com/NVIDIA/nvidia-docker)
- Nvidia GPU Driver

Assume there are two nodes:

- Node 1: `node1`

  Actor and gamecore processes will function on this node;

- Node 2: `node2`

  A learner operates on this node;

> 1. Substitute `node1` and `node2` with your own node IP addresses.
>
> 2. To run containers on the same node, eliminate the `--network host` flag when initiating a container and use the internal container IP. (Acquire the IP using `ifconfig` after entering the container's shell)

## Learner

This section illustrates the execution of a learner process and its dependencies on Node 2.

> Carry out the following actions in this section on `node2`.

1. Construct the development image containing `hok_env`, `rl_framework`.

    > Clone the repository on `node2`: `git clone git@github.com:tencent-ailab/hok_env.git`

    ```
    docker build -t learner -f dockerfile/dockerfile.dev --target battle --build-arg=BASE_IMAGE=tencentailab/hok_env:gpu_base_v2.0.1 .
    ```

    > 1. A CPU image can also be employed to run the learner on a non-GPU machine.

    For additional GPU software information, consult [dockerfile.base.gpu](../dockerfile/dockerfile.base.gpu)

    For further image information, refer to [dockerfile.dev](../dockerfile/dockerfile.dev)

2. Initiate the learner container

    ```
    docker run -it --gpus all --network host tencentailab/hok_env:gpu_v2.0.1 bash
    ```

    > Utilize the pre-built GPU image: `tencentailab/hok_env:gpu_v2.0.1`

    > If using a CPU image, omit the flag `--gpus all`

    > If you built the image in step 1, substitute `tencentailab/hok_env:gpu_v2.0.1` with your custom learner image name `learner`.

3. Configure learner node information

    Execute the following command within the learner container:
    ```
    echo "node2 root 36000 1" > /aiarena/scripts/learner/learner.iplist
    ```

    > Replace `node2` with your specific node IP address.

4. Initiate learner processes.

    Execute the following command within the learner container:
    ```
    # Create a symbolic link for working code as /aiarena/code
    ln -s /aiarena/3v3 /aiarena/code

    # For 1v1
    # ln -s /aiarena/1v1/ /aiarena/code

    # start learner
    sh /aiarena/scripts/learner/start_learner.sh
    ```

    You should observe the following output:
    ```
    # sh /aiarena/scripts/learner/start_learner.sh
    [2023-08-23 09:53:59] stop learner
    kill: (23): No such process
    [2023-08-23 09:53:59] parse ip list
    node2
    ['127.0.0.1 root 36000 1']
    [2023-08-23 09:53:59] start model_pool
    kill: (44): No such process
    rm: cannot remove 'trpc_go.yaml': No such file or directory
    [2023-08-23 09:53:59] start monitor
    [2023-08-23 09:53:59] Start
    [2023-08-23 09:53:59] Complete!
    [2023-08-23 09:53:59] node_list: 127.0.0.1:1
    [2023-08-23 09:53:59] node_num: 1
    [2023-08-23 09:53:59] learner_num: 1
    [2023-08-23 09:53:59] proc_per_node: 1
    [2023-08-23 09:53:59] NET_CARD_NAME: eth0
    [2023-08-23 09:53:59] mem_pool_list: [35200]
    [2023-08-23 09:53:59] start learner (mpirun)
    ```

    Inspect the log `/aiarena/logs/`.

## Gamecore Server

This section demonstrates how to construct a gamecore server image and execute the gamecore server on Node 1.

The subsequent actions in this section will be performed on `node1`.

1. Construct your custom gamecore server image

    > Clone the repository on `node1` using `git clone git@github.com:tencent-ailab/hok_env.git`

    1. Obtain and download the gamecore and license from [open-gamecore](https://aiarena.tencent.com/aiarena/zh/open-gamecore).

        ```
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

    2. Assemble the gamecore server image

        ```
        docker build -f ./dockerfile/dockerfile.gamecore -t gamecore --build-arg=BASE_IMAGE=tencentailab/hok_env:cpu_v2.0.1 .
        ```

2. Launch the gamecore server container and initiate the gamecore server process

    ```
    docker run -d --name gamecore --network host -e SIMULATOR_USE_WINE=1 -it gamecore sh /rl_framework/remote-gc-server/run_and_monitor_gamecore_server.sh
    ```

3. Evaluate the gamecore server

    ```
    # Access the container
    docker exec -it gamecore bash

    # Execute the test
    python3 /rl_framework/remote-gc-server/test_client.py
    ```

    > Note: If you have installed the `hok_env`, you can utilize the `GamecoreClient` implemented in `hok_env` to initiate a new game.
    > ```
    > from hok.common.gamecore_client import GamecoreClient
    > client = GamecoreClient(server_addr="node1:23432")     # Configure the IP address
    > runtime_id = "test-env"
    > server_config = [None, None] # use built-in rule agent for testing
    > camp_config = {
    >     "mode": "3v3",
    >     "heroes": [
    >         [{"hero_id": 190}, {"hero_id": 173}, {"hero_id": 117}], # camp_1
    >         [{"hero_id": 141}, {"hero_id": 111}, {"hero_id": 107}], # camp_2
    >     ],
    > }
    > client.start_game(
    >     runtime_id,
    >     server_config,
    >     camp_config,
    >     eval_mode=True,
    > )
    > client.wait_game(runtime_id)
    > ```

For additional information regarding the image, please consult [dockerfile.gamecore](../dockerfile/dockerfile.gamecore) and [run_windows_gamecore_on_linux](./run_windows_gamecore_on_linux.md).

## Actor

This section demonstrates how to execute actor processes and their dependencies on Node 1.

1. Construct the development image containing `hok_env` and `rl_framework`.

    > Clone the `hok_env` repository on `node1` using the command: `git clone git@github.com:tencent-ailab/hok_env.git`

    ```
    docker build -t actor -f dockerfile/dockerfile.dev --target battle --build-arg=BASE_IMAGE=tencentailab/hok_env:cpu_base_v2.0.1 .
    ```

    > Note:
    > 1. You can incorporate additional dependencies into `dockerfile.dev` and rebuild the actor image as needed.
    >
    > 2. The gamecore-server image, introduced in the previous section, also contains `hok_env` and `rl_framework`.
    >
    >    You can repurpose this image to run both the gamecore-server and actors within the same container.

2. Launch the actor container

    ```
    docker run -it -e GAMECORE_SERVER_ADDR=node1:23432 --network host tencentailab/hok_env:cpu_v2.0.1 bash
    ```

    - `-e GAMECORE_SERVER_ADDR=node1:23432`: Configure the `GAMECORE_SERVER_ADDR` environment variable to set the gamecore server address

    > In this example, we utilize the pre-built CPU image: `tencentailab/hok_env:cpu_v2.0.1`

    > If you have constructed the image in step 1, replace `tencentailab/hok_env:cpu_v2.0.1` with your custom actor image name `actor`.

3. Configure learner node information

    ```
    echo "node2 root 36000 1" > /aiarena/scripts/actor/learner.iplist
    ```

    > Replace `node2` with your specific node IP address.

4. Set the number of actor processes and initiate actors
    Execute the following command within the learner container:
    ```
    # Create a symbolic link for working code as /aiarena/code
    ln -s /aiarena/3v3 /aiarena/code

    # For 1v1
    # ln -s /aiarena/1v1/ /aiarena/code

    # Determine the number of actor processes. It is recommended to allocate one actor per CPU core.
    export ACTOR_NUM=6

    # Initiate actor processes
    sh /aiarena/scripts/actor/start_actor.sh
    ```

    Examine the log `/aiarena/logs/actor/actor_0.log`

The cluster training is now set up. To verify if the training task has started successfully, check the following log and resource usage:

```
# For learner container
/aiarena/logs/learner/train.log

# For gamecore server container
/aiarena/logs/gamecore-server.log

# For actor container
/aiarena/logs/actor/actor_*.log
```
