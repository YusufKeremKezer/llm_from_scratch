class GPTConfig:
    def __init__(self,**kwargs):
        self.__dict__.update(kwargs)

    def merge_from_dict(self, dict_to_merge):
        self.__dict__.update(dict_to_merge)
    




class TraniningConfig:

    @staticmethod
    def get_default_config():
        cfg = GPTConfig()
        cfg.device="cuda"
        cfg.batch_size=32
        cfg.context_len=16384
        cfg.learning_rate=3e-4
