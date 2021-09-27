class ObjectDict(dict):
    def __init__(self, d):
        super().__init__()
        for k, v in d.items():
            setattr(self, k, v)