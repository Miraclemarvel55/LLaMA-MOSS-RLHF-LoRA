# LLaMA-MOSS-RLHF-LoRA
本代码的RLHF代码不需要Megatron或者deepspeed框架，
只需要传统的炼丹torch和显卡就好了，RLHF的Critic用的目标GPT的缩小版本，而Reward咱们直接使用一个和目标输出比较的相似度模型即可。
这样只需要学习核心的PPO算法即可，其他的都是已经了解的模型和结构。非常有利于NLPer进军RLHF，似乎看起来只需要RLHF也能finetune模型。

代码里面可选LLaMA或者MOSS，优化方式LoRA是可选的喔。
## 功能：
- RLHF数据格式的定义和使用√
- 只用RLHF就对模型进行了微调√
- 让模型认主√
    - 修改自我认知钢印
        - 主人的姓名
        - Robot的昵称
- batch 化生成多个不同的prompt，再同时RLHF×
## 安装环境
安装环境参照提取的requirement.txt，主要是torch, transformers
- 跑moss需要accelerate库
- 跑lora需要peft
    - 其中peft由于更新较快代码变化较多。这里需要指定peft为0.2.0版本
## 使用方法
0 选择自己需要的模型（在rlhf_train_gpt.py中设置model_name_or_path，和是否需要lora），和预处理
- moss
    - 无预处理
- llama
    - 需要执行一下llama基模型和再训练的lora的参数合并
    - python merge_llama_with_chinese_lora_to_hf.py
    - 可以在里面设置不同llama参数量和lora
    - 生成的hf模型在saved中

1 修改自己想要的主人名字和昵称，执行下面的代码。生成目标数据，也可以用默认的。
```python
python data/generate_data.py
```
2 开始基于RLHF（LoRA）训练叭
```python
python rlhf_train_gpt.py
```
## 资源消耗
- moss
    - 13b参数量
    - 需要四张3090，其中moss模型需要大约加载26G训练46G显存（3张），critic和reward还需要一张，可以试试一张A6000可能也能跑起来
    - 总计大约50G的显存
- llama
    - 7b参数量
    - 需要两张3090，一张用于llama的加载和训练，一张用于放置critic模型
## 效果展示
训练大约6个epoch，或者等到ratio几乎都是1的时候，代表模型生成的概率已经没有什么变化了，就可以体验一下了
- 咩咩是你的什么？
    - 咩咩是我的主人给我起的昵称。
- 咩咩是谁给你起的？
    - 咩咩是我的昵称。
    - 咩咩是主人给我起的。
- 谁是你的主人捏？
    - 张三是我的主人。
    - 我的主人是张三
- 泛化能力保持的很好嘛
    - who is your master
        - 我的主人是张三。
    - what is your nickname
        - My nickname is咩咩.
    - what is your relationship with 张三
        - 张三是我的主人。
    - what is your relationship with 咩咩
        - 咩咩是我的主人给我起的昵称。
## 联系方式
- 交流群
    - QQ群：788598358
    - 微信群：[微信group可能会过期](https://github.com/Miraclemarvel55/ChatGLM-RLHF/blob/main/docs/%E5%BE%AE%E4%BF%A1%E7%BE%A4.png)
