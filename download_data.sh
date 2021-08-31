#!/bin/bash
set -eu -o pipefail
mkdir -p data
cd data

# Download SQuAD data
if [ ! -d squad ]
then
  mkdir squad
fi
cd squad
wget https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json
wget https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json
wget https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json
wget https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json
# Create indented versions, which are easier to read
for fn in `ls *.json`
do
python -m json.tool $fn > $fn.indented
done
cd ..
echo "SQuAD data downloaded."


# Download MRQA data
if [ ! -d mrqa ]
then
  mkdir mrqa
fi
cd mrqa
git clone https://github.com/mrqa/MRQA-Shared-Task-2019.git shared-task
shared-task/download_train.sh train 
shared-task/download_in_domain_dev.sh dev
shared-task/download_out_of_domain_dev.sh dev
cd train && ls *.jsonl.gz | xargs gunzip && cd ..
cd dev && ls *.jsonl.gz | xargs gunzip && cd ..
cd ..
echo "MRQA data downloaded."

