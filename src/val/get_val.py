from src.val.GPT2 import val_GPT2

def get_val(model_name, data_name, state_dict_full, logger):
    if model_name == 'GPT2':
        val_GPT2(data_name, state_dict_full, logger)
        return True
