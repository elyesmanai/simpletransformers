from simpletransformers.language_modeling import LanguageModelingModel
import logging
import argparse


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--use_tpu", default=False)
parser.add_argument("--num_epochs", default=100)
parser.add_argument("--batch_size", default=256)
args = parser.parse_args()

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

train_args = {
    "use_tpu": bool(args.use_tpu),
    "output_dir": "outputs/models/",
    "reprocess_input_data": False,
    "overwrite_output_dir": True,
    "num_train_epochs": args.num_epochs,
    "save_eval_checkpoints": True,
    "save_model_every_epoch": False,
    "learning_rate": 4e-4,
    "warmup_steps": 10000,
    "train_batch_size": args.batch_size,
    "eval_batch_size": 128,
    "gradient_accumulation_steps": 1,
    "block_size": 128,
    "max_seq_length": 128,
    "dataset_type": "simple",
    "logging_steps": 100,
    "evaluate_during_training": True,
    "evaluate_during_training_steps": 50000,
    "evaluate_during_training_verbose": True,
    "use_cached_eval_features": True,
    "sliding_window": True,
    "vocab_size": 52000,
    "generator_config": {
        "embedding_size": 128,
        "hidden_size": 256,
        "num_hidden_layers": 3,
    },
    "discriminator_config": {
        "embedding_size": 128,
        "hidden_size": 256,
    },
}

train_file = "../data/train.txt"
test_file = "../data/test.txt"

model = LanguageModelingModel(
    "electra",
    None,
    args=train_args,
    train_files=train_file,
)

model.train_model(train_file, eval_file=test_file,)

model.eval_model(test_file)


model.save_discriminator()

model.save_generator()
