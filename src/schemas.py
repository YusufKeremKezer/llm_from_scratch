from dataclasses import dataclass

@dataclass
class BPEOutput():
    word_splits:dict[str,list[str]]
    vocab:list
    token_to_id:dict
    merge_rules:list
