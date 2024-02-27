import yaml
import os
import copy

class Config:
    def __init__(self, default_config=None, replace_env_variables=False):
        if default_config:
            self.config = default_config.copy()
        else:
            self.config = {}
        self.replace_env_variables = replace_env_variables
        if self.replace_env_variables:
            self.config = _replace_env_variables_recursively(self.config)

    def set_value(self, name, value):
        keys = name.split('__')
        if len(keys) == 1:
            self.config[keys[0]] = value
        elif len(keys) == 2:
            self.config[keys[0]][keys[1]] = value
        elif len(keys) == 3:
            self.config[keys[0]][keys[1]][keys[2]] = value
        elif len(keys) == 4:
            self.config[keys[0]][keys[1]][keys[2]][keys[3]] = value
        else:
            raise NotImplementedError()

    def save(self, output_fn='config.yaml'):
        with open(output_fn, 'w') as f:
            yaml.safe_dump(self.config, f, indent=4)

    def flat_config(self):
        res = _flatten_dict(self.config)
        new_config = dict([('__'.join(keys), value) for keys, value in res.items()])
        return new_config


def _flatten_dict(d, pre_lst=None, result=None):
    if result is None:
        result = {}
    if pre_lst is None:
        pre_lst = []
    for k,v in d.items():
        if isinstance(v, dict):
            _flatten_dict(v, pre_lst=pre_lst+[k], result=result)
        else:
            result[tuple(pre_lst+[k])] = v
    return result


def _replace_env_variables_recursively(d):
    for k, v in d.items():
        if isinstance(v, dict):
            d[k] = _replace_env_variables_recursively(d.get(k, {}))
        elif isinstance(d[k], list):
            d[k] = v
        elif isinstance(d[k], str):
            d[k] = v.replace('$HOME', os.environ['HOME'])
        else:
            d[k] = v
    return d
    