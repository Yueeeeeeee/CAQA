import numpy as np
import random
import torch

import os
import string
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from config import *


def fix_random_seed_as(random_seed):
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_experiment_name(args):
    exp_name = ''
    if args.do_train:
        if args.train_squad:
            exp_name += 'train_squad'
        elif args.train_mrqa:
            exp_name += 'train_'
            mrqa_file = args.mrqa_train_file.split('/')[-1].split('.')[0].lower()
            exp_name += mrqa_file
        elif args.train_both:
            exp_name += 'train_squad_'
            mrqa_file = args.mrqa_train_file.split('/')[-1].split('.')[0].lower()
            exp_name += mrqa_file

        if args.adaptation_method is not None:
            exp_name += '_'
            exp_name += args.adaptation_method

        if args.do_predict:
            exp_name += '_predict'
            if args.predict_squad:
                exp_name += '_squad'
            elif args.predict_mrqa:
                mrqa_file = args.mrqa_predict_file.split('/')[-1].split('.')[0].lower()
                exp_name += '_' + mrqa_file
        
        if args.adaptation_method is not None:
            exp_name += '_' + str(args.beta) + '_' + str(args.sigma)
        return exp_name
    
    elif args.do_predict and args.output_dir and args.use_pretrained:
        exp_name = args.output_dir + '_predict'
        if args.predict_squad:
            exp_name += '_squad'
        elif args.predict_mrqa:
            exp_name += '_' + args.mrqa_predict_file.split('/')[-1].split('.')[0].lower()

    else:
        print('You should probably train a model or use pretrained model for predictions and inference.')
        exp_name = 'predict'
        if args.predict_squad:
            exp_name += '_squad'
        elif args.predict_mrqa:
            exp_name += '_' + args.mrqa_predict_file.split('/')[-1].split('.')[0].lower()
    
    return exp_name


def visualize_bert_output(output_sample, title):
    coordinates = PCA(n_components=2).fit_transform(output_sample['bert_output'])
    text = output_sample['text']
    plt.clf()
    fig, ax = plt.subplots()
    
    for i, vector in enumerate(coordinates):
        if i >= len(text):
            break
        if text[i] in string.punctuation or text[i] == '[CLS]' or text[i] == '[SEP]':
            continue
        
        color = COLOR_LABEL_MAPPING['default']
        marker = MARKER_LABEL_MAPPING['default']
        if i in output_sample['question_ids']:
            color = COLOR_LABEL_MAPPING['question']
            marker = MARKER_LABEL_MAPPING['question']
        elif i in output_sample['context_ids']:
            color = COLOR_LABEL_MAPPING['context']
            marker = MARKER_LABEL_MAPPING['context']
        elif i in output_sample['answer_ids']:
            continue  # draw answers last for convenience

        ax.scatter(vector[0], vector[1], c=color, marker=marker)
        ax.text(vector[0]+0.1, vector[1]+0.2, text[i], fontsize=6)
    
    for i, vector in enumerate(coordinates):
        if i >= len(text):
            break
        if text[i] in string.punctuation or text[i] == '[CLS]' or text[i] == '[SEP]':
            continue

        color = COLOR_LABEL_MAPPING['answer']
        marker = MARKER_LABEL_MAPPING['answer']
        if i in output_sample['answer_ids']:
            ax.scatter(vector[0], vector[1], c=color, marker=marker)
            ax.text(vector[0]+0.1, vector[1]+0.2, text[i], fontsize=6)

    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.set_title(title)
    fig.add_axes(ax)
    fig.canvas.draw()
    fig_data = np.array(fig.canvas.renderer._renderer)
    plt.close()

    return fig_data