import argparse
import yaml
import os
import subprocess


# 使用argparse解析命令行参数
def parse_command_line_args():
    parser = argparse.ArgumentParser(description='Experiment Setup and Training Script')
    parser.add_argument('--config', '-c', type=str, default='',
                        help='Path to the experiment configuration YAML file.')

    args = parser.parse_args()
    return args.config

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def setup_experiment(experiment_config):
    experiment_name = experiment_config['experiment_name']
    base_path = experiment_config['experiment_base_path']
    train_config_path = experiment_config['train_config_path']

    # 创建实验路径（如果不存在）
    experiment_dir = os.path.join(base_path, experiment_name)
    train_config_copy_path = os.path.join(experiment_dir, 'train_config.yaml')
    os.makedirs(experiment_dir, exist_ok=True)

    print(f'Experiment Name: {experiment_name}')
    print(f'Experiment Path: {experiment_dir}')

    with open(f'{experiment_dir}/experiment_config.yaml', 'w') as f:
        yaml.dump(experiment_config, f, default_flow_style=False)
    # 加载训练文件
    train_config = load_config(train_config_path)

    # 更新或创建子目录路径
    for key in ['weights', 'results', 'buckets']:
        path = os.path.join(experiment_dir, key)
        train_config[f'{key}_path'] = path
        os.makedirs(path, exist_ok=True)

    return train_config, train_config_copy_path

def setup_trainconfig(train_config, train_params, train_config_copy_path):

    for factor, value in train_params.items():
        train_config[factor] = value

    # 将更新后的配置内容写回文件
    with open(train_config_copy_path, 'w') as f:
        yaml.dump(train_config, f, default_flow_style=False)


    print("\nTrain Configuration Path:")
    print(train_config_copy_path)
    print("\nTrain Configuration:")
    print(train_config)

    return train_config

def train(train_script_path, train_config_path):

    train_command = f"python {train_script_path} --config {train_config_path}"
    subprocess.run(train_command, shell=True)


if __name__ == "__main__":
    # 解析命令行参数获取配置文件路径
    config_path = parse_command_line_args() or '../config/experiment_config.yaml'

    # 加载配置文件
    experiment_config = load_config(config_path)

    # 设置实验环境和更新配置中的路径
    train_config, train_config_path = setup_experiment(experiment_config)

    # 设置实验中的训练因子
    train_params = experiment_config['train_params']
    train_config = setup_trainconfig(train_config, experiment_config, train_config_path)

    # 训练
    train_script_path = experiment_config['train_script_path']
    train(train_script_path, train_config_path)
