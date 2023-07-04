# CAQA

Pytorch Implementation of the EMNLP 2021 Paper [Contrastive Domain Adaptation for Question Answering using Limited Text Corpora](https://arxiv.org/abs/2108.13854)

<img src=pics/intro.png>

We propose a novel framework for domain adaptation called contrastive domain adaptation for QA (CAQA). Specifically, CAQA combines techniques from question generation and domain-invariant learning to answer out-of-domain questions in settings with limited text corpora. Here, we train a QA system on both source data and generated data from the target domain with a contrastive adaptation loss that is incorporated in the training objective. By combining techniques from question generation and domain-invariant learning, our model achieved considerable improvements compared to state-of-the-art baselines



## Citing 

Please cite the following paper if you use our methods in your research:
```
@inproceedings{yue2021contrastive,
  title={Contrastive Domain Adaptation for Question Answering using Limited Text Corpora},
  author={Yue, Zhenrui and Kratzwald, Bernhard and Feuerriegel, Stefan},
  booktitle={Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
  year={2021}
}
```


## Data & Requirements

Download MRQA datasets with the following command
```
bash download_data.sh
```
To run our code you need PyTorch & Transformers, see requirements.txt for our running environment


## Train QAGen-T5 Generation Model

Our QAGen-T5 model consists of a question generation model and an answer generation model, we train our models with [question_generation](https://github.com/patil-suraj/question_generation) and we provide all generated synthetic data [here](https://drive.google.com/drive/folders/1RvYkKvlzhIDZQybP8q9L5wnXJ_PgxtEv?usp=sharing)


## Train BERT-QA with Contrastive Adaptation Loss

With the generated synthetic data, BERT-QA model can be trained using both SQuAD and the synthetic data, run following command to train a model with the proposed contrastive adaptation loss
```
CUDA_VISIBLE_DEVICES=0 python src/bert_squad_trans.py \
    --do_lower_case \
    --adaptation_method=smooth_can \
    --do_train \
    --train_both \
    --beta=0.001 \
    --sigma=0.001 \
    --mrqa_train_file=./data/synthetic/qagen_t5large_LM_5_10000_HotpotQA.jsonl \
    --do_predict \
    --predict_squad
```


## Performance

<img src=pics/performance.png>


## Acknowledgement

During the implementation we base our code mostly on [Transformers](https://github.com/huggingface/transformers) from Hugging Face, [question_generation](https://github.com/patil-suraj/question_generation) by Suraj Patil and [MRQA-Shared-Task-2019](https://github.com/mrqa/MRQA-Shared-Task-2019) by Fisch et al. Many thanks to these authors for their great work!
