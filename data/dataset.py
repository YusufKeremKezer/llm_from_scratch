# %% 

from datasets import load_dataset

# Login using e.g. `huggingface-cli login` to access this dataset
ds = load_dataset("FreedomIntelligence/medical-o1-reasoning-SFT", "en" ,split='train[:100]')
print(ds)

# %%


