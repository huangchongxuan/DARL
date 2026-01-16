set -x

eval "$(conda shell.bash hook)"

conda create -n RLPR python==3.10 -y
conda activate RLPR
# pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip install -e .
pip install flash-attn==2.7.4.post1


# below is for gemma & llama
pip install -U vllm==0.8.5
pip install tensordict==0.6.2