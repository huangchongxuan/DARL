set -x

eval "$(conda shell.bash hook)"

conda create -n vllm-server python=3.10 -y # delete this line if you have already created the environment
conda activate vllm-server
pip install -r vllm-server-requirements.txt
python verl/utils/server.py --model Qwen/Qwen2.5-72B-Instruct --gpu-ids 0,1,2,3 --host 127.0.0.1 --port 8001 # change args according to your needs
conda deactivate