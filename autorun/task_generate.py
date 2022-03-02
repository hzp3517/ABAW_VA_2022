import os
import csv
import fcntl

def mkdir(path):
    try:
        os.makedirs(path)
    except FileExistsError:
        pass
    
def make_grid(params):
    total_length = 1
    for key, value in params.items():
        total_length *= len(value) #一共有多少种参数组合方式
    
    ans = []
    for _ in range(total_length):
        ans.append({})
    
    combo_num = total_length
    for key, value in params.items():
        combo_num = combo_num // len(value)
        for i in range(0, total_length, combo_num):
            for j in range(combo_num):
                ans[i+j][key] = value[i//combo_num%len(value)]

    return ans

def process_grid(param_grid, order_list): #添加关联参数
    ans = []
    for param in param_grid:
        order_dict = {}
        for key in order_list:
            if key in param.keys():
                order_dict[key] = param[key]
            # else: #自己根据关联参数的逻辑需要修改的部分
            #     assert key in ['norm_features'] #所有关联参数
            #     if key == 'norm_features':
            #         ft_str = param['feature']
            #         features = ft_str.split(',')
            #         norm_ft_list = []
            #         for ft in features:
            #             if ft in norm_features:
            #                 norm_ft_list.append(ft)
            #         ft_str = ','.join(norm_ft_list)
            #         ft_str = 'None' if ft_str == '' else ft_str
            #         order_dict['norm_features'] = ft_str

        ans.append(order_dict)
    return ans

def remove_grid(param_grid): # 去除掉多余的组合
    ans = []
    for param in param_grid:
        # 当不适用预训练模型时，frozen_dense_blocks设置为0以外的组合都将被直接丢弃
        if param['pretrain'] == 'n' and param['frozen_dense_blocks'] > 0:
            continue
        else:
            ans.append(param)
    return ans

def make_task(independent_parameters, param_order_list):
    # tuned hyper-parameters
    param_grid = make_grid(independent_parameters)
    param_grid = process_grid(param_grid, param_order_list)
    param_grid = remove_grid(param_grid)
    template = 'source activate hzp_py37;sh ' + task_script + ' ' + ' '.join(['{' + key + '}' for key in param_order_list])

    total_cmd = []
    for param in param_grid:
        cmd = template.format(**param)
        total_cmd.append(cmd)
    
    # 平均分配gpu
    cmd_with_gpu = []
    for i in range(len(avialable_gpus)):
        task_num = len(total_cmd) / len(avialable_gpus)
        cmds = total_cmd[int(i*task_num):int((i+1)*task_num)]
        for cmd in cmds:
            cmd_with_gpu.append(cmd + ' ' + str(avialable_gpus[i]))
    
    for i in range(num_sessions):
        session_name = '{}_{}'.format(screen_name, i)
        task_file = os.path.join(auto_script_dir, f'{i}_task.sh')
        f = open(task_file, 'w')
        f.write('screen -dmS {}\n'.format(session_name))
        task_num = len(cmd_with_gpu) / num_sessions
        cmds = cmd_with_gpu[int(i*task_num):int((i+1)*task_num)]
        for cmd in cmds:
            _cmd = "screen -x -S {} -p 0 -X stuff '{}\n'\n".format(session_name, cmd)
            f.write(_cmd)
        f.write("screen -x -S {} -p 0 -X stuff 'exit\n'\n".format(session_name))
        # -dmS <作业名称> 新建一个session，但暂不进入
        # -x: 恢复之前离线的screen作业 
        # -S <作业名称> 指定screen作业的名称
        
        
auto_script_dir = 'autorun/auto'           # 生成脚本路径
auto_csv_dir = 'autorun/csv_results/debug'       # 生成结果csv文件路径，注意train_csv.py文件中也需要同步修改！！！
# 当需要在同一组设定下跑多次时候最好在这里开个子目录，而不是改下面的"name"

task_script = 'scripts/train_denseface_auto.sh'     # 执行script路径
avialable_gpus = [1, 2, 3]                 # 可用GPU有哪些
num_sessions = 6                        # 一共开多少个session同时执行（即开几个screen的会话）
avialable_gpus = avialable_gpus[:num_sessions]
screen_name = 'hzp_abaw_train'
independent_parameters = {                              # 一共有哪些非关联参数
    # bash scripts/train_denseface_toy.sh debug y 0 512 0.4 1e-4 64 1 3

    'name': ['debug'], #注意：此列表中只能有一个元素，这个名字与log文件名最前面一部分也是关联的
    'pretrain': ['y', 'n'],
    # 'pretrain': ['y'],
    'frozen_dense_blocks': [0, 1, 2, 3],
    'clf_hidden_size': [512],
    'dropout_rate': [0.3],
    'lr': [1e-4],
    'batch_size': [256],
    'run_idx': [1, 2]
}
param_order_list = ['name', 'pretrain', 'frozen_dense_blocks', 'clf_hidden_size', 'dropout_rate', 'lr', 'batch_size', 'run_idx'] #除gpu外所有参数的顺序

mkdir(auto_script_dir)


# 创建csv结果文件并写入表头和列标题（所有特征名称）
mkdir(auto_csv_dir)
csv_path = os.path.join(auto_csv_dir, independent_parameters['name'][0] + '.csv')
with open(csv_path, 'w') as f:
    fcntl.flock(f.fileno(), fcntl.LOCK_EX) #加锁
    writer = csv.writer(f)
    #写入表头
    file_head = ['param']
    for lr in independent_parameters['lr']:
        for run in independent_parameters['run_idx']:
            file_head.append(str(lr) + '_run' + str(run) + '_f1')
            file_head.append(str(lr) + '_run' + str(run) + '_acc')
    writer.writerow(file_head)
    #写入列标题
    for pretrain in independent_parameters['pretrain']:
        row_title = 'pretrain' if pretrain=='y' else 'scratch'
        if pretrain == 'n':
            for frozen in independent_parameters['frozen_dense_blocks']:
                if frozen == 0:
                    line = [row_title]
                    for _ in range(len(independent_parameters['lr']) * len(independent_parameters['run_idx']) * 2):
                        line.append('-') #在应该填入实验结果的位置先以'-'补上
                    writer.writerow(line)
        else:
            for frozen in independent_parameters['frozen_dense_blocks']:
                tmp_row_title = row_title + '-{}'.format(frozen)
                line = [tmp_row_title]
                for _ in range(len(independent_parameters['lr']) * len(independent_parameters['run_idx']) * 2):
                    line.append('-') #在应该填入实验结果的位置先以'-'补上
                writer.writerow(line)
    fcntl.flock(f.fileno(), fcntl.LOCK_UN) #解锁


make_task(independent_parameters, param_order_list)

for i in range(num_sessions):
    cmd = 'sh {}/{}_task.sh'.format(auto_script_dir, i)
    print(cmd)
    os.system(cmd)