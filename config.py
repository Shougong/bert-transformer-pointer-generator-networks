class Config(object):
    '''
        Setting hyper-parameters
    '''
    def __init__(self, vocab):
        super(Config, self).__init__()
        self.hidden_size = 768
        self.num_layers = 6
        self.num_heads = 8
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.pf_dim = 512
        self.dropout_rate = 0.1
        


