# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from setuptools import setup, find_packages
import os

version_folder = os.path.dirname(os.path.join(os.path.abspath(__file__)))
with open(os.path.join(version_folder, 'verl/version/version')) as f:
    __version__ = f.read().strip()

install_requires = [
  'torch==2.4.0',
  'aiohttp==3.12.13',
  'accelerate==1.6.0',
  'codetiming==1.4.0',
  'datasets==3.5.0',
  'fastapi==0.115.12',
  'dill==0.3.8',
  'hydra-core==1.3.2',
  'numpy==1.26.4',
  'pandas==2.2.3',
  'peft==0.15.2',
  'pyarrow==19.0.1',
  'pybind11==2.13.6',
  'pylatexenc==2.10',
  'ray==2.44.1',
  'tensordict==0.5.0',
  'transformers==4.49.0',
  'vllm==0.6.3',
  'wandb==0.19.9',
  'matplotlib==3.10.1',
  'openpyxl==3.1.5',
  'XlsxWriter==3.2.3',
]

TEST_REQUIRES = [
    'pytest',
    'yapf',
    'py-spy==0.4.0'
]
PRIME_REQUIRES = ['pyext']
GPU_REQUIRES = [
    'liger-kernel',
    'flash-attn==2.7.4.post1'
]

extras_require = {
  'test': TEST_REQUIRES,
  'prime': PRIME_REQUIRES,
  'gpu': GPU_REQUIRES,
  'all': TEST_REQUIRES + PRIME_REQUIRES + GPU_REQUIRES
}

from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name='verl',
    version=__version__,
    package_dir={'': '.'},
    packages=find_packages(where='.'),
    url='https://github.com',
    license='Apache 2.0',
    author='Bytedance - Seed - MLSys',
    author_email='zhangchi.usc1992@bytedance.com, gmsheng@connect.hku.hk',
    install_requires=install_requires,
    extras_require=extras_require,
    package_data={
        'verl': ['version/*', 'trainer/config/*.yaml'],
    },
    include_package_data=True,
    long_description=long_description,
    long_description_content_type='text/markdown'
)