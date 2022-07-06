docker run --gpus all \
    -v $(pwd):/app/project/ \
    --network=host \
    -ti \
    --shm-size=4gb \
    -p 8265:8265 \
    project-dev bash
