from src.val.GPT2 import val_GPT2
from src.val.Llama import val_Llama

def get_val(model_name, data_name, state_dict_full, logger):
    if model_name == 'GPT2':
        val_GPT2(model_name, data_name, state_dict_full, logger)
    elif model_name == 'Llama':
        val_Llama(model_name, data_name, state_dict_full, logger)
    return True
