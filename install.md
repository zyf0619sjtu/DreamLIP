conda create -n torch1.11-py3.7 python=3.7 -y
conda activate torch1.11-py3.7
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch -y
pip install regex ftfy tqdm huggingface_hub sentencepiece protobuf==3.20.0 braceexpand pandas webdataset tensorboard flair nltk
pip install open_clip_torch