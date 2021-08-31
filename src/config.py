import argparse


EXPERIMENT_ROOT_FOLDER = 'experiments'

COLOR_LABEL_MAPPING = {
    'default': '0.5',
    'question': 'cyan',
    'context': 'orange',
    'answer': 'red'
}

MARKER_LABEL_MAPPING = {
    'default': 'o',
    'question': 'o',
    'context': 'o',
    'answer': 'd'
}


parser = argparse.ArgumentParser()

parser.add_argument("--bert_model", default='bert-base-uncased', type=str,
                    help="Bert pre-trained model selected in the list: bert-base-uncased, "
                    "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                    "bert-base-multilingual-cased, bert-base-chinese.")
parser.add_argument("--output_dir", default=None, type=str,
                    help="The output directory where the model checkpoints and predictions will be written.")

parser.add_argument("--max_seq_length", default=384, type=int,
                    help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                            "longer than this will be truncated, and sequences shorter than this will be padded.")
parser.add_argument("--doc_stride", default=128, type=int,
                    help="When splitting up a long document into chunks, how much stride to take between chunks.")
parser.add_argument("--max_query_length", default=64, type=int,
                    help="The maximum number of tokens for the question. Questions longer than this will "
                            "be truncated to this length.")

parser.add_argument("--pretrained_dir", default=None, type=str,
                    help="The pretrained directory where the model weights could be loaded.")
parser.add_argument("--train_file", default='./data/squad/train-v1.1.json', type=str, help="SQuAD json for training. E.g., train-v1.1.json")
parser.add_argument("--mrqa_train_file", default=None, type=str, help="MRQA json for training for both, else put in train_file.")
parser.add_argument("--do_train", action='store_true', help="Whether to run training.")
parser.add_argument("--train_both", action='store_true', help="Whether to train on both train data")
parser.add_argument("--train_squad", action='store_true', help="Whether to only train on squad data")
parser.add_argument("--squad_samples", default=0, type=int, help="How many samples to train on squad data")
parser.add_argument("--train_mrqa", action='store_true', help="Whether to only train on mrqa data")
parser.add_argument("--mrqa_samples", default=0, type=int, help="How many samples to train on mrqa data")
parser.add_argument("--amount_train_both", default=0, type=int, help="How much OOD data to include in train")

parser.add_argument("--predict_file", default='./data/squad/dev-v1.1.json', type=str, help="SQuAD json for predictions. E.g., dev-v1.1.json.")
parser.add_argument("--mrqa_predict_file", default=None, type=str, help="MRQA json for predictions. E.g., HotpotQA.json.")
parser.add_argument("--do_predict", action='store_true', help="Whether to run eval on the dev set.")
parser.add_argument("--use_pretrained", action='store_true', help="Whether to use a pretrained model from out_dir.")
parser.add_argument("--predict_squad", action='store_true', help="Whether to run eval on SQuAD dev data.")
parser.add_argument("--predict_mrqa", action='store_true', help="Whether to run eval on MRQA dev data.")
parser.add_argument("--visualize_prediction", action='store_true', help="Whether to visualize results on dev data.")
parser.add_argument("--visualize_amount", default=500, type=int, help="Total visualization amount on dev data.")

parser.add_argument("--output_name", default=None, type=str, help="Prefix for eval output files.")
parser.add_argument("--train_batch_size", default=16, type=int, help="Total batch size for training.")
parser.add_argument("--predict_batch_size", default=16, type=int, help="Total batch size for predictions.")
# This is important
parser.add_argument("--test_time_dropout", action='store_true', help='Activate test time dropout')
parser.add_argument('--dropout_seed', type=int, default=None, help='dropout test time seed')
# If you're predicting on MRQA train
parser.add_argument("--predict_on_train", type=str, default=None, help='Dataset QID PATH if predicting on MRQA train')
parser.add_argument("--learning_rate", default=3e-5, type=float, help="The initial learning rate for Adam.")
parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
parser.add_argument("--max_grad_norm", default=1.0, type=float, help="The maximum norm for backward gradients.")
parser.add_argument("--num_train_epochs", default=2.0, type=float, help="Total number of training epochs to perform.")
parser.add_argument("--warmup_proportion", default=0.0, type=float,
                    help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10%% "
                            "of training.")
parser.add_argument("--n_best_size", default=20, type=int,
                    help="The total number of n-best predictions to generate in the nbest_predictions.json "
                            "output file.")
parser.add_argument("--max_answer_length", default=30, type=int,
                    help="The maximum length of an answer that can be generated. This is needed because the start "
                            "and end predictions are not conditioned on one another.")
parser.add_argument("--verbose_logging", action='store_true',
                    help="If true, all of the warnings related to data processing will be printed. "
                            "A number of warnings are expected for a normal SQuAD evaluation.")
parser.add_argument("--no_cuda",
                    action='store_true',
                    help="Whether not to use CUDA when available")
parser.add_argument('--seed',
                    type=int,
                    default=42,
                    help="random seed for initialization")
parser.add_argument('--gradient_accumulation_steps',
                    type=int,
                    default=1,
                    help="Number of updates steps to accumulate before performing a backward/update pass.")
parser.add_argument("--do_lower_case",
                    action='store_true',
                    help="Whether to lower case the input text. True for uncased models, False for cased models.")
parser.add_argument('--overwrite_output_dir',
                    action='store_true',
                    help="Overwrite the content of the output directory")
parser.add_argument('--version_2_with_negative',
                    action='store_true',
                    help='If true, the SQuAD examples contain some that do not have an answer.')
parser.add_argument('--null_score_diff_threshold',
                    type=float, default=0.0,
                    help="If null_score - best_non_null is greater than the threshold predict null.")

parser.add_argument("--adaptation_method", default=None, type=str, help="Choose methods from can, smoothing and smooth_can")
parser.add_argument('--beta', type=float, default=0.01, help="Scale for contrastive loss")
parser.add_argument('--sigma', type=float, default=0.01, help="Noise scale for feature smoothing")

args = parser.parse_args()