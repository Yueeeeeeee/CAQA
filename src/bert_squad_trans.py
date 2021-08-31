from __future__ import absolute_import, division, print_function

import argparse
import logging
import os
import random
import sys
import pdb
import json
from io import open
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)
from tqdm import tqdm, trange

# from transformers.file_utils import WEIGHTS_NAME, CONFIG_NAME
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from transformers import BertTokenizer

from bert_squad_dataset_utils import *
from evaluation import *
from config import *
from utils import *

if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle


logger = logging.getLogger(__name__)


def main():
    print(args)

    if args.adaptation_method is None:
        from transformers import BertForQuestionAnswering
    elif args.adaptation_method == 'smooth_can':
        from modeling_bert_smooth_can import CustomizedBertForQuestionAnswering as BertForQuestionAnswering

    if not args.do_train and not args.do_predict:
        raise ValueError('At least one of "do_train" or "do_predict" must be True.')
    if args.do_train:
        if not args.train_squad and not args.train_mrqa and not args.train_both:
            raise ValueError('If "do_train" is True, one of "train_squad", "train_mrqa" or "train_both" must be True.')
        if int(args.train_squad) + int(args.train_mrqa) + int(args.train_both) > 1:
            raise ValueError('Only one of "train_squad", "train_mrqa" or "train_both" can be True.')
        if args.train_squad or args.train_both:
            if not args.train_file:
                raise ValueError(
                    'If "train_squad" or "train_both" is True, then "train_file" must be specified.')
        if args.train_mrqa or args.train_both:
            if not args.mrqa_train_file:
                raise ValueError(
                    'If "train_mrqa" or "train_both" is True, then "mrqa_train_file" must be specified.')
    if args.do_predict:
        if not args.predict_squad and not args.predict_mrqa:
            raise ValueError('If "do_predict" is True, at least one of "predict_squad" or "predict_mrqa" must be True.')
        if args.predict_squad:
            if not args.predict_file:
                raise ValueError(
                    'If "predict_squad" is True, then "predict_file" must be specified.')
        if args.predict_mrqa:
            if not args.mrqa_predict_file:
                raise ValueError(
                    'If "predict_mrqa" is True, then "mrqa_predict_file" must be specified.')

    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    fix_random_seed_as(args.seed)

    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.WARN)

    if args.gradient_accumulation_steps < 1:
        raise ValueError('Invalid gradient_accumulation_steps parameter: {}, should be >= 1'.format(
                            args.gradient_accumulation_steps))
    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    if not args.output_dir:
        args.output_dir = get_experiment_name(args)
    export_root = os.path.join(EXPERIMENT_ROOT_FOLDER, args.output_dir)
    if not os.path.exists(export_root):
        os.makedirs(export_root)

    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)
    if args.use_pretrained:
        model = BertForQuestionAnswering.from_pretrained(export_root)
    else:
        model = BertForQuestionAnswering.from_pretrained(args.bert_model)

    print('Model Total Parameters:', sum(p.numel() for p in model.parameters()))
    model.to(device)
    
    if args.do_train:
        if args.train_squad:
            if args.train_file[-4:] == '.pkl':
                cached_train_features_file = args.train_file
            else:
                cached_train_features_file = args.train_file + '.pkl'
        
        elif args.train_mrqa:
            if args.mrqa_train_file[-4:] == '.pkl':
                cached_train_features_file = args.mrqa_train_file
            else:
                cached_train_features_file = args.mrqa_train_file + '.pkl'
        
        elif args.train_both:
            if args.train_file[-4:] == '.pkl':
                cached_train_features_file = args.train_file
            else:
                cached_train_features_file = args.train_file + '.pkl'
            if args.mrqa_train_file[-4:] == '.pkl':
                cached_mrqa_train_features_file = args.mrqa_train_file
            else:
                cached_mrqa_train_features_file = args.mrqa_train_file + '.pkl'

        if Path(cached_train_features_file).is_file():
            print('Already preprocessed. Skip preprocessing...')
            train_features = pickle.load(Path(cached_train_features_file).open('rb'))
        
        elif args.train_squad or args.train_both:
            print('Reading SQuAD examples...')
            train_examples = read_squad_examples(
                input_file=args.train_file, is_training=True, version_2_with_negative=args.version_2_with_negative)
            train_features = convert_examples_to_features(
                examples=train_examples,
                tokenizer=tokenizer,
                max_seq_length=args.max_seq_length,
                doc_stride=args.doc_stride,
                max_query_length=args.max_query_length,
                is_training=True)
            logger.info('Saving train features into cached file %s', cached_train_features_file)
            with open(cached_train_features_file, 'wb') as writer:
                pickle.dump(train_features, writer)
        
        elif args.train_mrqa:
            print('Reading MRQA examples...')
            train_examples = read_mrqa_examples(input_file=args.mrqa_train_file, is_training=True)
            train_features = convert_examples_to_features(
                examples=train_examples,
                tokenizer=tokenizer,
                max_seq_length=args.max_seq_length,
                doc_stride=args.doc_stride,
                max_query_length=args.max_query_length,
                is_training=True)
            logger.info('Saving train features into cached file %s', cached_train_features_file)
            with open(cached_train_features_file, 'wb') as writer:
                pickle.dump(train_features, writer)

        if args.train_both and Path(cached_mrqa_train_features_file).is_file():
            print('Already preprocessed. Skip preprocessing...')
            mrqa_features = pickle.load(Path(cached_mrqa_train_features_file).open('rb'))
        elif args.train_both:
            print('Reading MRQA examples...')
            mrqa_examples = read_mrqa_examples(input_file=args.mrqa_train_file, is_training=True)
            mrqa_features = convert_examples_to_features(
                examples=mrqa_examples,
                tokenizer=tokenizer,
                max_seq_length=args.max_seq_length,
                doc_stride=args.doc_stride,
                max_query_length=args.max_query_length,
                is_training=True)
            logger.info('Saving train features into cached file %s', cached_train_features_file)
            with open(cached_mrqa_train_features_file, 'wb') as writer:
                pickle.dump(mrqa_features, writer)

        if args.train_both:
            if args.squad_samples > 0:
                train_features = np.random.choice(train_features, size=args.squad_samples).tolist()
            if args.mrqa_samples > 0:
                mrqa_features = np.random.choice(mrqa_features, size=args.mrqa_samples).tolist()
            
            all_input_type = torch.tensor([0 for f in train_features] + [1 for f in mrqa_features], dtype=torch.long)
            train_features = train_features + mrqa_features

            all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
            all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
            all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
            all_start_positions = torch.tensor([f.start_position for f in train_features], dtype=torch.long)
            all_end_positions = torch.tensor([f.end_position for f in train_features], dtype=torch.long)
            train_data = TensorDataset(all_input_type, all_input_ids, all_input_mask, all_segment_ids,
                                    all_start_positions, all_end_positions)
            
        else:
            if args.train_squad and args.squad_samples > 0:
                train_features = np.random.choice(train_features, size=args.squad_samples).tolist()

            elif args.train_mrqa and args.mrqa_samples > 0:
                train_features = np.random.choice(train_features, size=args.mrqa_samples).tolist()
            
            all_input_type = torch.tensor([0 for f in train_features], dtype=torch.long)
            all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
            all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
            all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
            all_start_positions = torch.tensor([f.start_position for f in train_features], dtype=torch.long)
            all_end_positions = torch.tensor([f.end_position for f in train_features], dtype=torch.long)
            train_data = TensorDataset(all_input_type, all_input_ids, all_input_mask, all_segment_ids,
                                    all_start_positions, all_end_positions)

        
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": args.weight_decay,
            },
            {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
        ]
        
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
        scheduler = get_linear_schedule_with_warmup(optimizer,
                        num_warmup_steps=args.warmup_proportion*t_total,
                        num_training_steps=t_total)

        global_step = 0
        logger.info('***** Running training *****')
        logger.info('Batch size = %d', args.train_batch_size)
        logger.info('Num steps = %d', t_total)

        model.train()
        if args.adaptation_method is None:
            for epoch in range(int(args.num_train_epochs)):
                for step, batch in enumerate(tqdm(train_dataloader, desc='Epoch {}'.format(epoch + 1))):
                    batch = tuple(t.to(device) for t in batch)
                    input_type, input_ids, input_mask, segment_ids, start_positions, end_positions = batch

                    outputs = model(input_ids=input_ids,
                                    token_type_ids=segment_ids,
                                    attention_mask=input_mask,
                                    start_positions=start_positions,
                                    end_positions=end_positions)
                    loss = outputs[0]
                    if args.gradient_accumulation_steps > 1:
                        loss = loss / args.gradient_accumulation_steps

                    loss.backward()
                    
                    if (step + 1) % args.gradient_accumulation_steps == 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                        optimizer.step()
                        scheduler.step()
                        model.zero_grad()
                        global_step += 1
        
        elif args.adaptation_method == 'smooth_can':
            for epoch in range(int(args.num_train_epochs)):
                for step, batch in enumerate(tqdm(train_dataloader, desc='Epoch {}'.format(epoch + 1))):
                    batch = tuple(t.to(device) for t in batch)
                    input_type, input_ids, input_mask, segment_ids, start_positions, end_positions = batch

                    outputs = model(input_type=input_type,
                                    input_ids=input_ids,
                                    token_type_ids=segment_ids,
                                    attention_mask=input_mask,
                                    start_positions=start_positions,
                                    end_positions=end_positions,
                                    beta=args.beta,
                                    sigma=args.sigma)
                    loss = outputs[0]
                    if args.gradient_accumulation_steps > 1:
                        loss = loss / args.gradient_accumulation_steps

                    loss.backward()

                    if (step + 1) % args.gradient_accumulation_steps == 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                        optimizer.step()
                        scheduler.step()
                        model.zero_grad()
                        global_step += 1

        model.save_pretrained(export_root)
        tokenizer.save_pretrained(export_root)
        output_args_file = os.path.join(export_root, 'training_args.bin')
        torch.save(args, output_args_file)

    from transformers import BertForQuestionAnswering
    if args.do_train or args.use_pretrained:
        model = BertForQuestionAnswering.from_pretrained(export_root)
        tokenizer = BertTokenizer.from_pretrained(export_root, do_lower_case=args.do_lower_case)
    
    else:
        model = BertForQuestionAnswering.from_pretrained(args.bert_model)
        tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)

    model.to(device)
    if args.do_predict:
        if args.test_time_dropout:        
            def apply_dropout(m):
                if type(m) == torch.nn.Dropout:
                    m.train()

        if args.predict_mrqa:
            cached_eval_examples_file = args.mrqa_predict_file + '_examples.pkl'
            cached_eval_features_file = args.mrqa_predict_file + '_features.pkl'

            if Path(cached_eval_examples_file).is_file() and Path(cached_eval_features_file).is_file():
                print('Already preprocessed. Skip preprocessing...')
                eval_examples = pickle.load(Path(cached_eval_examples_file).open('rb'))
                eval_features = pickle.load(Path(cached_eval_features_file).open('rb'))
            else:
                print('Convert examples fot testing...')
                eval_examples = read_mrqa_examples_with_multiple_answers(
                    input_file=args.mrqa_predict_file)
                eval_features = convert_examples_to_features(
                    examples=eval_examples,
                    tokenizer=tokenizer,
                    max_seq_length=args.max_seq_length,
                    doc_stride=args.doc_stride,
                    max_query_length=args.max_query_length,
                    is_training=False,
                    multiple_answers=True)
                logger.info('Saving train features into %s and %s', cached_eval_examples_file, cached_eval_features_file)
                with open(cached_eval_examples_file, 'wb') as writer:
                    pickle.dump(eval_examples, writer)
                with open(cached_eval_features_file, 'wb') as writer:
                    pickle.dump(eval_features, writer)

            logger.info('***** Running predictions *****')
            logger.info('Batch size = %d', args.predict_batch_size)

            all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
            all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
            all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
            all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
            eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_example_index)
            eval_sampler = SequentialSampler(eval_data)
            eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.predict_batch_size)
            
            all_tokens = {}
            all_answer_positions = {}
            all_start_positions = {}
            all_end_positions = {}
            for i in all_example_index.numpy():
                all_start_positions[i] = eval_features[i].start_position
                all_end_positions[i] = eval_features[i].end_position
                answer_spans = [set(list(range(start, end+1))) for start, end in zip(all_start_positions[i], all_end_positions[i])]
                all_answer_positions[i] = set().union(*answer_spans)
                all_tokens[i] = eval_features[i].tokens
            
            model.eval()
            if args.test_time_dropout:
                model.apply(apply_dropout)
            
            all_results = []
            visualization_data_correct = []
            visualization_data_incorrect = []
            logger.info('Start evaluating')
            for input_ids, input_mask, segment_ids, example_indices in tqdm(eval_dataloader, desc='Evaluating'):
                if len(all_results) % 1000 == 0:
                    logger.info('Processing example: %d' % (len(all_results)))
                input_ids = input_ids.to(device)
                input_mask = input_mask.to(device)
                segment_ids = segment_ids.to(device)
                with torch.no_grad():
                    sequence_output = model.bert(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask)[0]
                    logits = model.qa_outputs(sequence_output)
                    start_logits, end_logits = logits.split(1, dim=-1)
                    batch_start_logits = start_logits.squeeze(-1)
                    batch_end_logits = end_logits.squeeze(-1)
                    # outputs = model(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask)
                    # batch_start_logits, batch_end_logits = outputs
                
                for i, example_index in enumerate(example_indices):
                    start_logits = batch_start_logits[i].detach().cpu().tolist()
                    end_logits = batch_end_logits[i].detach().cpu().tolist()
                    eval_feature = eval_features[example_index.item()]
                    unique_id = int(eval_feature.unique_id)
                    all_results.append(RawResult(unique_id=unique_id,
                                                start_logits=start_logits,
                                                end_logits=end_logits))
                    
                    if args.visualize_prediction:
                        start_id = torch.tensor(start_logits).argmax(dim=-1).item()
                        end_id = torch.tensor(end_logits).argmax(dim=-1).item()
                        answer_correct = False
                        for start, end in zip(all_start_positions[example_index.item()], all_end_positions[example_index.item()]):
                            if start == start_id and end == end_id:
                                answer_correct = True
                                break

                        if len(list(all_answer_positions[example_index.item()])) == 1:
                            if list(all_answer_positions[example_index.item()])[0] == 0:
                                continue

                        question_ids = list(range(1, segment_ids[i].detach().cpu().tolist().index(1) - 1))
                        answer_ids = list(all_answer_positions[example_index.item()])
                        context_ids = list(range(segment_ids[i].detach().cpu().tolist().index(1), len(segment_ids[i]) - segment_ids[i].detach().cpu().tolist()[::-1].index(1) - 1))
                        context_ids = list(set(context_ids) - all_answer_positions[example_index.item()])

                        if answer_correct and len(visualization_data_correct) < (args.visualize_amount//2):
                            visualization_sample = {
                                'question_ids': question_ids,
                                'context_ids': context_ids,
                                'answer_ids': answer_ids,
                                'text': all_tokens[example_index.item()],
                                'bert_output': sequence_output[i].detach().cpu().tolist(),
                                'predicted_answer': [torch.tensor(start_logits).argmax(dim=-1).item(), torch.tensor(end_logits).argmax(dim=-1).item()]
                            }
                            visualization_data_correct.append(visualization_sample)

                        elif not answer_correct and len(visualization_data_incorrect) < (args.visualize_amount//2):
                            visualization_sample = {
                                'question_ids': question_ids,
                                'context_ids': context_ids,
                                'answer_ids': answer_ids,
                                'text': all_tokens[example_index.item()],
                                'bert_output': sequence_output[i].detach().cpu().tolist(),
                                'predicted_answer': [torch.tensor(start_logits).argmax(dim=-1).item(), torch.tensor(end_logits).argmax(dim=-1).item()]
                            }
                            visualization_data_incorrect.append(visualization_sample)

            if not args.output_name:
                args.output_name = args.mrqa_predict_file.split('/')[-1].split('.')[0].lower()

            if args.visualize_prediction:
                correct_samples_file = os.path.join(export_root, '{}_correct_samples.json'.format(args.output_name))
                with open(correct_samples_file, 'w') as f:
                    json.dump(visualization_data_correct, f)
                incorrect_samples_file = os.path.join(export_root, '{}_incorrect_samples.json'.format(args.output_name))
                with open(incorrect_samples_file, 'w') as f:
                    json.dump(visualization_data_incorrect, f)

            if args.test_time_dropout:
                dropout_path = os.path.join(export_root, 'dropout_real')
                if not os.path.exists(dropout_path):
                    os.makedirs(dropout_path)
                output_prediction_file = os.path.join(dropout_path, '{}{}_predictions.json'.format(args.output_name, args.dropout_seed))
                output_nbest_file = os.path.join(dropout_path, '{}{}_nbest_predictions.json'.format(args.output_name, args.dropout_seed))
            else:
                output_prediction_file = os.path.join(export_root, '{}_predictions.json'.format(args.output_name))
                output_nbest_file = os.path.join(export_root, '{}_nbest_predictions.json'.format(args.output_name))

            output_null_log_odds_file = os.path.join(export_root, '{}_null_odds.json'.format(args.output_name))
            print('Writing to {}'.format(output_nbest_file))
            write_predictions(eval_examples, eval_features, all_results,
                                args.n_best_size, args.max_answer_length,
                                args.do_lower_case, output_prediction_file,
                                output_nbest_file, output_null_log_odds_file, args.verbose_logging,
                                args.version_2_with_negative, args.null_score_diff_threshold)

            metrics = evaluate_mrqa(args.mrqa_predict_file, output_prediction_file)
            print(metrics)
            with open(os.path.join(export_root, args.output_name+'_prediction_metrics.json'), 'w') as f:
                json.dump(metrics, f, indent=4)
        
        if args.predict_squad:
            cached_eval_examples_file = args.predict_file + '_examples.pkl'
            cached_eval_features_file = args.predict_file + '_features.pkl'

            if Path(cached_eval_examples_file).is_file() and Path(cached_eval_features_file).is_file():
                print('Already preprocessed. Skip preprocessing...')
                eval_examples = pickle.load(Path(cached_eval_examples_file).open('rb'))
                eval_features = pickle.load(Path(cached_eval_features_file).open('rb'))
            else:
                print('Convert examples fot testing...')
                eval_examples = read_squad_examples_with_multiple_answers(
                    input_file=args.predict_file, version_2_with_negative=args.version_2_with_negative)
                eval_features = convert_examples_to_features(
                    examples=eval_examples,
                    tokenizer=tokenizer,
                    max_seq_length=args.max_seq_length,
                    doc_stride=args.doc_stride,
                    max_query_length=args.max_query_length,
                    is_training=False,
                    multiple_answers=True)
                logger.info('Saving train features into %s and %s', cached_eval_examples_file, cached_eval_features_file)
                with open(cached_eval_examples_file, 'wb') as writer:
                    pickle.dump(eval_examples, writer)
                with open(cached_eval_features_file, 'wb') as writer:
                    pickle.dump(eval_features, writer)

            logger.info('***** Running predictions *****')
            logger.info('Batch size = %d', args.predict_batch_size)

            all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
            all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
            all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
            all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
            eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_example_index)
            eval_sampler = SequentialSampler(eval_data)
            eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.predict_batch_size)
            
            all_tokens = {}
            all_answer_positions = {}
            all_start_positions = {}
            all_end_positions = {}
            for i in all_example_index.numpy():
                all_start_positions[i] = eval_features[i].start_position
                all_end_positions[i] = eval_features[i].end_position
                answer_spans = [set(list(range(start, end+1))) for start, end in zip(all_start_positions[i], all_end_positions[i])]
                all_answer_positions[i] = set().union(*answer_spans)
                all_tokens[i] = eval_features[i].tokens
            
            model.eval()
            if args.test_time_dropout:
                model.apply(apply_dropout)
            
            all_results = []
            visualization_data_correct = []
            visualization_data_incorrect = []
            logger.info('Start evaluating')
            for input_ids, input_mask, segment_ids, example_indices in tqdm(eval_dataloader, desc='Evaluating'):
                if len(all_results) % 1000 == 0:
                    logger.info('Processing example: %d' % (len(all_results)))
                input_ids = input_ids.to(device)
                input_mask = input_mask.to(device)
                segment_ids = segment_ids.to(device)
                with torch.no_grad():
                    sequence_output = model.bert(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask)[0]
                    logits = model.qa_outputs(sequence_output)
                    start_logits, end_logits = logits.split(1, dim=-1)
                    batch_start_logits = start_logits.squeeze(-1)
                    batch_end_logits = end_logits.squeeze(-1)
                    # outputs = model(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask)
                    # batch_start_logits, batch_end_logits = outputs

                for i, example_index in enumerate(example_indices):
                    start_logits = batch_start_logits[i].detach().cpu().tolist()
                    end_logits = batch_end_logits[i].detach().cpu().tolist()
                    eval_feature = eval_features[example_index.item()]
                    unique_id = int(eval_feature.unique_id)
                    all_results.append(RawResult(unique_id=unique_id,
                                                start_logits=start_logits,
                                                end_logits=end_logits))

                    if args.visualize_prediction:
                        start_id = torch.tensor(start_logits).argmax(dim=-1).item()
                        end_id = torch.tensor(end_logits).argmax(dim=-1).item()
                        answer_correct = False
                        for start, end in zip(all_start_positions[example_index.item()], all_end_positions[example_index.item()]):
                            if start == start_id and end == end_id:
                                answer_correct = True
                                break
                        
                        if len(list(all_answer_positions[example_index.item()])) == 1:
                            if list(all_answer_positions[example_index.item()])[0] == 0:
                                continue

                        question_ids = list(range(1, segment_ids[i].detach().cpu().tolist().index(1) - 1))
                        answer_ids = list(all_answer_positions[example_index.item()])
                        context_ids = list(range(segment_ids[i].detach().cpu().tolist().index(1), len(segment_ids[i]) - segment_ids[i].detach().cpu().tolist()[::-1].index(1) - 1))
                        context_ids = list(set(context_ids) - all_answer_positions[example_index.item()])

                        if answer_correct and len(visualization_data_correct) < (args.visualize_amount//2):
                            visualization_sample = {
                                'question_ids': question_ids,
                                'context_ids': context_ids,
                                'answer_ids': answer_ids,
                                'text': all_tokens[example_index.item()],
                                'bert_output': sequence_output[i].detach().cpu().tolist(),
                                'predicted_answer': [torch.tensor(start_logits).argmax(dim=-1).item(), torch.tensor(end_logits).argmax(dim=-1).item()]
                            }
                            visualization_data_correct.append(visualization_sample)

                        elif not answer_correct and len(visualization_data_incorrect) < (args.visualize_amount//2):
                            visualization_sample = {
                                'question_ids': question_ids,
                                'context_ids': context_ids,
                                'answer_ids': answer_ids,
                                'text': all_tokens[example_index.item()],
                                'bert_output': sequence_output[i].detach().cpu().tolist(),
                                'predicted_answer': [torch.tensor(start_logits).argmax(dim=-1).item(), torch.tensor(end_logits).argmax(dim=-1).item()]
                            }
                            visualization_data_incorrect.append(visualization_sample)

            if not args.output_name:
                args.output_name = 'squad'
            
            if args.visualize_prediction:
                correct_samples_file = os.path.join(export_root, '{}_correct_samples.json'.format(args.output_name))
                with open(correct_samples_file, 'w') as f:
                    json.dump(visualization_data_correct, f)
                incorrect_samples_file = os.path.join(export_root, '{}_incorrect_samples.json'.format(args.output_name))
                with open(incorrect_samples_file, 'w') as f:
                    json.dump(visualization_data_incorrect, f)

            if args.test_time_dropout:
                dropout_path = os.path.join(export_root, 'dropout_real')
                if not os.path.exists(dropout_path):
                    os.makedirs(dropout_path)
                output_prediction_file = os.path.join(dropout_path, '{}{}_predictions.json'.format(args.output_name, args.dropout_seed))
                output_nbest_file = os.path.join(dropout_path, '{}{}_nbest_predictions.json'.format(args.output_name, args.dropout_seed))
            else:
                output_prediction_file = os.path.join(export_root, '{}_predictions.json'.format(args.output_name))
                output_nbest_file = os.path.join(export_root, '{}_nbest_predictions.json'.format(args.output_name))

            output_null_log_odds_file = os.path.join(export_root, '{}_null_odds.json'.format(args.output_name))
            print('Writing to {}'.format(output_nbest_file))
            write_predictions(eval_examples, eval_features, all_results,
                                args.n_best_size, args.max_answer_length,
                                args.do_lower_case, output_prediction_file,
                                output_nbest_file, output_null_log_odds_file, args.verbose_logging,
                                args.version_2_with_negative, args.null_score_diff_threshold)

            metrics = evaluate_squad(args.predict_file, output_prediction_file)
            print(metrics)
            with open(os.path.join(export_root, args.output_name+'_prediction_metrics.json'), 'w') as f:
                json.dump(metrics, f, indent=4)


if __name__ == '__main__':
    main()