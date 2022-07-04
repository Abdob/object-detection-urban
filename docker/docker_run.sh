docker run --gpus all \
    -v $(pwd):/app/project/ \
    --network=host \
    -ti \
    project-dev bash
    #(eg, `--shm-size`).