# 说明
1. 工具运行需求参见 ``./complete_verifier/environment.yaml`` ，使用`conda`创建虚拟环境：
```shell
conda env create -f complete_verifier/environment.yaml --name abcrown
conda activate abcrown
```
2. 使用时如果出现auto_LiPRA相关库缺失的情况，请先编译`auto_LiRPA`，如果没有直接跳转至步骤3即可
```shell
# git clone the original repository when needed
git clone https://github.com/Verified-Intelligence/auto_LiRPA

cd auto_LiRPA
python setup.py install
```

3.  使用
```shell
cd complete_verifier
python abc.py [PATH to config.json]
```

4. `config.json`参数说明：
我将config.json中的参数转换为符合原始工具要求的参数。原始工具参数说明详见[完整参数](https://github.com/Verified-Intelligence/alpha-beta-CROWN/blob/main/complete_verifier/docs/abcrown_all_params.yaml)，对应关系如下表所示。

| 参数    |   对应项 | 说明     |
| :--------- | :------- | :--------- |
| picPath | ['data', 'data_path'] | 原始参数列表中没有此项，额外添加 |
| modelPath | ['model', 'path'] |  |
| modelName | ['model', 'name'] |  |
| modelStructure | ['model', ''] |  |
| numClasses | ['data', 'num_outputs'] |  |
| outputPath | ['general', 'results_file'] | 原始工具将结果存为txt |

下表是额外添加的一些比较有用的参数：
| 参数    |   对应项 | 说明     |
| :--------- | :------- | :--------- |
| timeout | ['bab', 'timeout'] | 每个样本的超时时间 |
| ['attack', 'pgd_order'] | ['attack', 'pgd_order'] | 是否添加pgd_attack以增加鲁棒性，可选before, after, skip



