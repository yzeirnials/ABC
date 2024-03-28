import json
import yaml 
import os
import subprocess
import sys
import time
import socket

yaml_file_name = "CONFIG.yaml"

def json_to_yaml(json_file_path):

    with open(json_file_path, 'r') as json_file:
        json_data = json.load(json_file)
        original_data = json_data
    

    if 'general' not in original_data:
        original_data['general'] = {} 
    if 'model' not in original_data:
        original_data['model'] = {} 
    if 'bab' not in original_data:
        original_data['bab'] = {} 
    if 'data' not in original_data:
        original_data['data'] = {} 
    if 'attack' not in original_data:
        original_data['attack'] = {}  

    general_updated = original_data['general'].copy()
    general_updated['results_file'] = original_data['outputPath']

    model_updated = original_data['model'].copy()
    model_updated['name'] = original_data['modelName']
    model_updated['path'] = original_data['modelPath']
    model_updated['structure'] = original_data['modelStructure']

    data_updated = original_data['data'].copy()
    data_updated['num_outputs'] = original_data['numClasses']
    data_updated['data_path'] = original_data['picPath']
    data_updated['dataset'] = "Customized(\"custom_model_data\", \"image_folder\")"

    bab_updated = original_data['bab'].copy()
    bab_updated['timeout'] = original_data['timeout']

    transformed_data = {
        'general': general_updated,
        'model': model_updated,
        'data': data_updated,
        'bab': bab_updated,
        'specification': original_data['specification'],
        'solver': original_data['solver'],
        'attack': original_data['attack']
    }

    currDir = os.getcwd()
    yaml_file_path = os.path.join(currDir, yaml_file_name)
    
    with open(yaml_file_path, 'w') as yaml_file:
        yaml.dump(transformed_data, yaml_file, allow_unicode=True, default_flow_style=False)


class ABC:
    def __init__(self, args):
        self.ConfigPath = args[0]
        # self.ResultPath = self.args[1]

        json_to_yaml(self.ConfigPath)
    
    def main(self):
        print(f'Experiments at {time.ctime()} on {socket.gethostname()}')
        subprocess.run(["python", "abcrown.py", "--config", yaml_file_name])



if __name__ == '__main__':
    abc = ABC(args = sys.argv[1:])
    abc.main()

