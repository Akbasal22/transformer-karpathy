from tokenizers import Tokenizer, models, trainers, pre_tokenizers



#tokenizer
def tokenizer():
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizers = pre_tokenizers.Whitespace()
    trainer = trainers.BpeTrainer(
        vocab_size=200,  # pick a vocab size (depends on dataset size)
        min_frequency=2,
        special_tokens=["[PAD]", "[UNK]", "[BOS]", "[EOS]"]
    )
    files = ["input.txt"]
    tokenizer.train(files, trainer)
    tokenizer.save("bpe_tokenizer.json")

tokenizer()