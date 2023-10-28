import UnigramSampler
class NegativeSamplingLoss:
    def __init__(self, W, corpus, power=0.75, sample_size = 5):
        self.sample_size = sample_size
        self.sampler = UnigramSampler(corpus, power, sample_size)
        self.loss_layers = [SigmoidWithLoss() for _ in range(sample_size+1)]
        self.embed_dot_layers = [EmbeddingDot(W) for _ in range(sample_size+1)]
        
        self.params, self.grads = [], []
        for layer in self.embed_dot_layers:
            self.params += layer.params
            self.grads += layer.grads
            
    def forward(self, h, target):
        batch_size = target.shape[0]
        nagative_sample = self.sampler.get_negative_sample(target)
        
        #正例のフォーワード
        score = self.embed_dot_layers[0].forward(h, target)#h, targetはバッチサイズ分ある。
        correct_label = np.ones(batch_size, correct_label) #バッチサイズの正例を作ることでまとめて処理できる。
        loss = self.loss_layers[0].forward(score, correct_label)
        
        #負例のフォーワード
        negative_label = np.zeros(batch_size, dtype = np.int32)
        for i in range(self.sample_size):
            negative_target = negative_sample[:, i]
            score = self.embed_dot_layers[1 + i].forward(h, negative_target)
            loss += self.loss_layers[1+i].forward(score, negarive_label)
            
        return loss
    
    def backword(self, dout = 1):
        dh = 0
        for L0, L1 in zip(self.loss_layers, self.embed_dot_layers):
            dscore = L0.backword(dout)
            dh += L1.backword(dscore)
        return dh