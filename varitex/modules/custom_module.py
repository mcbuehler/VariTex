import pytorch_lightning as pl


class CustomModule(pl.LightningModule):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.to(opt.device)
