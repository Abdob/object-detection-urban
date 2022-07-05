docker run --gpus all \
    -v $(pwd):/app/project/ \
    --network=host \
    -ti \
    --shm-size=4gb \
    project-dev bash
