import torch

from uelm4.config import load_config
from uelm4.train import train_from_texts
from uelm4.model.uelm4_model import UELM4


def test_train_sample_corpus(tmp_path):
    corpus = (tmp_path / "sample.txt")
    corpus.write_text("To be, or not to be, that is the question.")
    cfg = load_config("small")
    model = UELM4(cfg)
    trained = train_from_texts([line.strip() for line in corpus.read_text().split('.') if line.strip()], config_name="small")
    tokens = torch.randint(0, trained.cfg.model.vocab_size, (8,))
    logits = trained(tokens)
    assert logits.shape[0] == tokens.shape[0]
    assert logits.shape[1] > 0
