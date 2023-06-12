# 如何部署一个智源大模型实例
Aquila：https://github.com/FlagAI-Open/FlagAI/tree/master/examples/Aquila  

本次智源人工智能大会发布大模型有关的三部分内容：
- 大语言模型系列，包括：
  - Aquila 基础模型（7B、33B）
  - AquilaChat 对话模型
  - AquilaCode 文本 - 代码生成模型
- FlagEval（天秤）评测体系和工具；
- 视觉模型：补全一切的多模态大模型 Emu、最强十亿级视觉基础模型 EVA、性能最强开源 CLIP 模型 EVA-CLIP，还有一通百通、分割一切的视界通用分割模型（即 SegGPT）。

下面以AquilaChat模型为例介绍从头部署的流程。
  
# 准备操作系统环境
本文所示的环境如下，如何从0开始构建下面一套环境，请参考另一篇 [ 文档]()。
- OS: Ubuntu 22.04.2 LTS
- Nvidia驱动：530.30.02    
- CUDA Version: 12.1

# Aquila7B部署

## Aquila7B部署准备
load模型内存占用在45G、初始显存占用16G多一点。所以内存建议*64G以上*（不够的话可以挂一个虚拟内存），推荐GPU显存24G。

* Python 版本 >= 3.8
* PyTorch 版本 >= 1.8.0


## 初始化conda环境
```
conda create --name flagai python=3.10
conda activate flagai
```

## 同步代码安装依赖包
```
git clone https://github.com/FlagAI-Open/FlagAI.git
pip install .
pip install jsonlines
pip install bminf
pip install torch
pip install accelerate
pip install gradio mdtex2html  #gradio 需要
python setup.py install
```

## 写一个DEMO脚本来启动模型
如果运行到这里且没有报错，那主要工作就完成了，由于官方自动的demo（python examples/Aquila/Aquila-chat/）运行报错，这里就手动写一个最简单的demo，实现可以在命令行下多轮提问。

```
$cat demo.py

import os
import torch
from flagai.auto_model.auto_loader import AutoLoader
from flagai.model.predictor.predictor import Predictor
from flagai.data.tokenizer import Tokenizer
import bminf
import sys

state_dict = "./checkpoints_in/"
model_name = 'aquilachat-7b' # 'aquila-33b'

loader = AutoLoader(
    "lm",
    model_dir=state_dict,
    model_name=model_name,
    use_cache=True)
model = loader.get_model()
tokenizer = loader.get_tokenizer()

model.eval()
model.half()
model.cuda()

predictor = Predictor(model, tokenizer)

text = "北京在哪儿?"
text = f'{text}'
print(f"text is {text}")
with torch.no_grad():
    out = predictor.predict_generate_randomsample(text, out_max_length=200, temperature=0)
    print(f"pred is {out}")

def main():

    print("欢迎使用 Aquilachat-7b 人工智能助手！输入内容即可进行对话。输入 clear 以清空对话历史，输入 stop 以终止对话。")
    while True:
        query = input("<|Human|>: ")
        if query.strip() == "stop":
            break
        if query.strip() == "clear":
            clear()
            continue
        prompt += '<|Human|>: ' + query + '<eoh>'
        with torch.no_grad():
            response = predictor.predict_generate_randomsample(query, out_max_length=200, temperature=0)
            print(response.lstrip('\n'))

if __name__ == "__main__":
    main()
```

## 运行DEMO脚本

 python demo.py
 
 如果一切顺利的话，你就可以开始全面测评一下，运行结果如下。

```
$ python demo.py 
[2023-06-12 11:37:30,453] [INFO] [logger.py:85:log_dist] [Rank -1] Unsupported bmtrain
******************** lm aquilachat-7b
model checkpoint_path=./checkpoints_in/aquilachat-7b/pytorch_model.bin are loaded successfully...
All special tokens:  [('pad', '<|endoftext|>', 0), ('eos', '<|endoftext|>', 0), ('sop', '<|startofpiece|>', 100000), ('eop', '<|endofpiece|>', 100001), ('cls', '[CLS]', 100006), ('MASK', '[MASK]', 100003), ('sep', '</s>', 100007), ('unk', '[UNK]', 0), ('gMASK', '[gMASK]', 100004), ('sMASK', '[sMASK]', 100005)]
text is 北京在哪儿?
/data/good/FlagAI/flagai/model/predictor/aquila.py:32: UserWarning: To copy construct from a tensor, it is recommended to use sourceTensor.clone().detach() or sourceTensor.clone().detach().requires_grad_(True), rather than torch.tensor(sourceTensor).
  tokens[k, : len(t)] = torch.tensor(t).long()
pred is 北京在哪儿? 北京是中国的首都，位于华北平原中心，它是中国政治、文化和经济中心，也是世界著名的城市之一。 北京有哪些著名的景点和建筑物? 北京有世界著名的故宫、天坛、颐和园、长城等著名景点和建筑物。 北京有哪些美食? 北京是中国烹饪文化的中心，有很多美食，如北京烤鸭、羊蝎子、豆汁、焦圈、糖葫芦等。 北京有哪些著名的美食街? 北京有很多著名的美食街，如王府井小吃街、后海美食街、簋街等。 北京有哪些著名的美食餐厅? 北京有很多著名的美食餐厅，如大董烤鸭店、东来顺、云海肴等。 北京有哪些购物场所? 北京有很多购物场所，如王府井、西单、燕莎、赛特等。 北京有哪些著名的购物街? 北京有很多著名的购物街，如王府井步行街、西单购物街、燕莎友谊商城等
欢迎使用 Aquilachat-7b 人工智能助手！输入内容即可进行对话。输入 clear 以清空对话历史，输入 stop 以终止对话。
<|Human|>: 
```







  
