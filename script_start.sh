# Arguments for Milvus, Embedding Model, and Chat Model
milvus=$1
embed=$2
chat=$3

# install the dependencies.
pip install -r requirements.txt

# launch Milvus container
if [ "$milvus" == "true" ]; then
  wget https://github.com/milvus-io/milvus/releases/download/v2.4.13-hotfix/milvus-standalone-docker-compose.yml -O docker-compose.yml
  docker compose up -d
fi
 
# Export a local NIM cache
export LOCAL_NIM_CACHE=/raid/sivaks1/.cache/nim
mkdir -p "$LOCAL_NIM_CACHE"

# Launch the embedding NIM model if requested.
if [ "$embed" == "true" ]; then
  docker run -d --rm \
    --runtime=nvidia \
    --gpus all \
    --shm-size=16GB \
    -e NGC_API_KEY=$NGC_API_KEY \
    -v "$LOCAL_NIM_CACHE:/opt/nim/.cache" \
    -u $(id -u) \
    -p 8001:8000 \
    nvcr.io/nim/nvidia/nv-embedqa-e5-v5:latest
fi

# Launch the chat completion NIM model if requested.
if [ "$chat" == "true" ]; then
  docker run -d --rm \
    --runtime=nvidia \
    --gpus all \
    --shm-size=16GB \
    -e NGC_API_KEY=$NGC_API_KEY \
    -v "$LOCAL_NIM_CACHE:/opt/nim/.cache" \
    -u $(id -u) \
    -p 8000:8000 \
    nvcr.io/nim/nvidia/llama-3.1-nemotron-70b-instruct:latest
fi
