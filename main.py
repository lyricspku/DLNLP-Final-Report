import math  # 导入数学模块，提供基本的数学运算函数
import torch  # 导入PyTorch库，用于深度学习模型的构建和训练
import torch.nn as nn  # 导入PyTorch的神经网络模块
from torch import Tensor  # 从PyTorch导入Tensor类
from torch.nn.utils.rnn import pad_sequence  # 导入pad_sequence函数，用于将序列填充到相同长度
from torch.utils.data import DataLoader  # 导入DataLoader类，用于数据加载
from collections import Counter  # 导入Counter类，用于计数
from torch.nn import TransformerEncoder, TransformerDecoder, TransformerEncoderLayer, TransformerDecoderLayer# 导入Transformer相关的编码器和解码器模块
import io  # 导入io模块，用于文件操作
import time  # 导入time模块，用于计时
import pandas as pd  # 导入pandas库，用于数据处理
import numpy as np  # 导入numpy库，用于科学计算
import json
from janome.tokenizer import Tokenizer
import tqdm  # 导入tqdm库，用于显示进度条
 
# 设置随机种子，以确保结果的可复现性
torch.manual_seed(0)
# 检查是否有可用的GPU，如果有则使用GPU，否则使用CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
# 读取英文和日文的词汇表和数据
with open('./data/vocalbulary_en_list.json', 'rb') as f:
    vocalbulary_en_list = json.load(f)
with open('./data/vocalbulary_jp_list.json', 'rb') as f:
    vocalbulary_jp_list = json.load(f)
with open('./data/vocalbulary_en.json', 'rb') as f:
    vocalbulary_en = json.load(f)
with open('./data/vocalbulary_jp.json', 'rb') as f:
    vocalbulary_jp = json.load(f)
with open('./data/train.json', 'rb') as f:
    raw_train_data = json.load(f)
with open('./data/valid.json', 'rb') as f:
    raw_valid_data = json.load(f)
with open('./data/test.json', 'rb') as f:
    raw_test_data = json.load(f)
 
# 定义批处理大小
BATCH_SIZE = 8

UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3

def data_process(datafortrain):
    # 初始化一个空列表来存储处理后的数据
    data = []
    # 使用 zip 函数将日文句子和英文句子配对遍历
    for item in datafortrain:
        # 对每个日文句子进行分词并转换为词汇表中对应的索引，构建张量
        ja_tensor_ = torch.tensor([vocalbulary_jp[token] if token in vocalbulary_jp else UNK_IDX for token in item["tokens_jp"]],
                                    dtype=torch.long)
        # 对每个英文句子进行分词并转换为词汇表中对应的索引，构建张量
        en_tensor_ = torch.tensor([vocalbulary_en[token] if token in vocalbulary_en else UNK_IDX for token in item["tokens_en"]],
                                    dtype=torch.long)
        # 将日文张量和英文张量组成一个元组，并添加到数据列表中
        data.append((ja_tensor_, en_tensor_))
    # 返回处理后的数据列表
    return data
# 调用数据处理函数，生成训练数据
train_data = data_process(raw_train_data)
valid_data = data_process(raw_valid_data)
test_data = data_process(raw_test_data)

def generate_batch(data_batch):
    # 初始化空列表来存储日文和英文句子的张量
    ja_batch, en_batch = [], []
    # 遍历每个数据批次中的日文和英文句子对
    for (ja_item, en_item) in data_batch:
        # 在日文句子的开头添加起始符，在末尾添加结束符，并将其放入日文句子张量列表中
        ja_batch.append(torch.cat([torch.tensor([BOS_IDX]), ja_item, torch.tensor([EOS_IDX])], dim=0))
        # 在英文句子的开头添加起始符，在末尾添加结束符，并将其放入英文句子张量列表中
        en_batch.append(torch.cat([torch.tensor([BOS_IDX]), en_item, torch.tensor([EOS_IDX])], dim=0))
    # 对日文句子张量列表进行填充，使用 PAD_IDX 作为填充值
    ja_batch = pad_sequence(ja_batch, padding_value=PAD_IDX)
    # 对英文句子张量列表进行填充，使用 PAD_IDX 作为填充值
    en_batch = pad_sequence(en_batch, padding_value=PAD_IDX)
    # 返回填充后的日文句子张量和英文句子张量
    return ja_batch, en_batch
# 创建一个 DataLoader 对象，用于加载训练数据
train_iter = DataLoader(train_data, batch_size=BATCH_SIZE,
                        shuffle=True, collate_fn=generate_batch)
valid_iter = DataLoader(valid_data, batch_size=BATCH_SIZE,
                        shuffle=False, collate_fn=generate_batch)
test_iter = DataLoader(test_data, batch_size=BATCH_SIZE,
                        shuffle=False, collate_fn=generate_batch)

 
from torch.nn import (TransformerEncoder, TransformerDecoder,
                      TransformerEncoderLayer, TransformerDecoderLayer)
 
 
class Seq2SeqTransformer(nn.Module):
    def __init__(self, num_encoder_layers: int, num_decoder_layers: int,
                 emb_size: int, src_vocab_size: int, tgt_vocab_size: int,
                 dim_feedforward:int = 512, dropout:float = 0.1):
        """
        初始化 Seq2Seq Transformer 模型
        参数:
        num_encoder_layers (int): 编码器层数
        num_decoder_layers (int): 解码器层数
        emb_size (int): 嵌入向量大小
        src_vocab_size (int): 源语言词汇表大小
        tgt_vocab_size (int): 目标语言词汇表大小
        dim_feedforward (int, optional): 前馈神经网络的隐藏层大小，默认值为 512
        dropout (float, optional): dropout 概率，默认值为 0.1
        """
        super(Seq2SeqTransformer, self).__init__()
        # 定义编码器层
        encoder_layer = TransformerEncoderLayer(d_model=emb_size, nhead=NHEAD,
                                                dim_feedforward=dim_feedforward)
        # 定义 Transformer 编码器，由多个编码器层组成
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        
        # 定义解码器层
        decoder_layer = TransformerDecoderLayer(d_model=emb_size, nhead=NHEAD,
                                                dim_feedforward=dim_feedforward)
        # 定义 Transformer 解码器，由多个解码器层组成
        self.transformer_decoder = TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        
        # 定义生成器，将解码器的输出转换为目标词汇表的概率分布
        self.generator = nn.Linear(emb_size, tgt_vocab_size)
        
        # 定义源语言和目标语言的词嵌入层
        self.src_tok_emb = TokenEmbedding(src_vocab_size, emb_size)
        self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, emb_size)
        
        # 定义位置编码层
        self.positional_encoding = PositionalEncoding(emb_size, dropout_rate=dropout)
 
    def forward(self, src: Tensor, trg: Tensor, src_mask: Tensor,
                tgt_mask: Tensor, src_padding_mask: Tensor,
                tgt_padding_mask: Tensor, memory_key_padding_mask: Tensor):
        """
        前向传播函数
        参数:
        src (Tensor): 源语言输入序列
        trg (Tensor): 目标语言输入序列
        src_mask (Tensor): 源语言序列的掩码
        tgt_mask (Tensor): 目标语言序列的掩码
        src_padding_mask (Tensor): 源语言序列的填充值掩码
        tgt_padding_mask (Tensor): 目标语言序列的填充值掩码
        memory_key_padding_mask (Tensor): 编码器输出的填充值掩码
        返回:
        Tensor: 模型的输出
        """
        # 对源语言输入进行词嵌入并添加位置编码
        src_emb = self.positional_encoding(self.src_tok_emb(src))
        # 对目标语言输入进行词嵌入并添加位置编码
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(trg))
        
        # 通过编码器
        memory = self.transformer_encoder(src_emb, src_mask, src_padding_mask)
        # 通过解码器
        outs = self.transformer_decoder(tgt_emb, memory, tgt_mask, None,
                                        tgt_padding_mask, memory_key_padding_mask)
        
        # 通过生成器将解码器的输出转换为目标词汇表的概率分布
        return self.generator(outs)
 
    def encode(self, src: Tensor, src_mask: Tensor):
        """
        编码函数
        参数:
        src (Tensor): 源语言输入序列
        src_mask (Tensor): 源语言序列的掩码
        返回:
        Tensor: 编码器的输出
        """
        return self.transformer_encoder(self.positional_encoding(
                            self.src_tok_emb(src)), src_mask)
 
    def decode(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor):
        """
        解码函数
        参数:
        tgt (Tensor): 目标语言输入序列
        memory (Tensor): 编码器的输出
        tgt_mask (Tensor): 目标语言序列的掩码
        返回:
        Tensor: 解码器的输出
        """
        return self.transformer_decoder(self.positional_encoding(
                          self.tgt_tok_emb(tgt)), memory,
                          tgt_mask)
 
class PositionalEncoding(nn.Module):
    def __init__(self, emb_size: int, dropout_rate, maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        """
        初始化位置编码模块。
        
        参数:
        emb_size (int): 嵌入向量的维度。
        dropout (float): dropout 概率，用于防止过拟合。
        maxlen (int): 序列的最大长度，默认值为 5000。
        """
        # 计算位置编码中的分母部分
        den = torch.exp(- torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        
        # 生成位置索引
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        
        # 初始化位置嵌入矩阵
        pos_embedding = torch.zeros((maxlen, emb_size))
        
        # 对位置嵌入矩阵进行赋值
        pos_embedding[:, 0::2] = torch.sin(pos * den)  # 偶数列
        pos_embedding[:, 1::2] = torch.cos(pos * den)  # 奇数列
        
        # 对位置嵌入矩阵增加一个维度
        pos_embedding = pos_embedding.unsqueeze(-2)
        
        # 定义dropout层
        self.dropout = nn.Dropout(dropout_rate)
        
        # 注册位置嵌入矩阵为buffer，使其在模型保存和加载时不会作为参数
        self.register_buffer('pos_embedding', pos_embedding)
 
 
    def forward(self, token_embedding: Tensor):
        """
        前向传播函数，将位置编码添加到token嵌入上，并应用dropout。
        
        参数:
        token_embedding (Tensor): 输入的token嵌入张量，形状为 [seq_len, batch_size, emb_size]。
        
        返回:
        Tensor: 叠加位置编码后的token嵌入张量。
        """
        # 将位置编码添加到token嵌入上，并应用dropout
        return self.dropout(token_embedding +
                            self.pos_embedding[:token_embedding.size(0),:])
 
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size):
        """
        初始化token嵌入模块。
        
        参数:
        vocab_size (int): 词汇表的大小。
        emb_size (int): 嵌入向量的维度。
        """
        super(TokenEmbedding, self).__init__()
        # 定义嵌入层
        self.embedding = nn.Embedding(vocab_size, emb_size)
        # 存储嵌入向量的维度
        self.emb_size = emb_size
    def forward(self, tokens: Tensor):
        """
        前向传播函数，获取tokens的嵌入表示并进行缩放。
        
        参数:
        tokens (Tensor): 输入的token张量，形状通常为 [seq_len, batch_size]。
        
        返回:
        Tensor: 缩放后的token嵌入张量，形状为 [seq_len, batch_size, emb_size]。
        """
        # 获取token的嵌入表示，并乘以嵌入维度的平方根进行缩放
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)
 
# 定义生成方形后续掩码的函数
def generate_square_subsequent_mask(sz):
    """
    生成一个大小为 sz x sz 的方形掩码矩阵，其中上三角部分为 0，其他部分为 -inf。
    这个掩码用于在解码过程中屏蔽未来的位置，以保证自回归模型只能看到当前和之前的 token。
    参数:
    sz (int): 掩码矩阵的大小 (边长)
    返回:
    torch.Tensor: 一个大小为 sz x sz 的掩码矩阵
    """
    # 创建一个上三角矩阵，值为 1，表示允许的位置；之后转置矩阵
    mask = (torch.triu(torch.ones((sz, sz), device=device)) == 1).transpose(0, 1)
    # 把上三角部分保留为 0，其他部分填充为 -inf，表示不允许的位置
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask
 
# 定义创建掩码的函数
def create_mask(src, tgt):
    """
    为源序列和目标序列创建必要的掩码，包括源掩码、目标掩码、源填充掩码和目标填充掩码。
    参数:
    src (torch.Tensor): 源序列张量，形状为 (src_seq_len, batch_size)
    tgt (torch.Tensor): 目标序列张量，形状为 (tgt_seq_len, batch_size)
    返回:
    tuple: 包含以下四个元素的元组：
        - src_mask (torch.Tensor): 源序列掩码矩阵
        - tgt_mask (torch.Tensor): 目标序列掩码矩阵
        - src_padding_mask (torch.Tensor): 源序列填充掩码矩阵
        - tgt_padding_mask (torch.Tensor): 目标序列填充掩码矩阵
    """
    # 获取源序列和目标序列的长度
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]
 
    # 生成目标序列的掩码，防止解码时看到未来的 token
    tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
 
    # 生成源序列的掩码，这里全为 False，因为不需要屏蔽任何位置
    src_mask = torch.zeros((src_seq_len, src_seq_len), device=device).type(torch.bool)
 
    # 生成源序列的填充掩码，屏蔽填充的位置
    src_padding_mask = (src == PAD_IDX).transpose(0, 1)
    # 生成目标序列的填充掩码，屏蔽填充的位置
    tgt_padding_mask = (tgt == PAD_IDX).transpose(0, 1)
 
    # 返回所有生成的掩码
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask
 
SRC_VOCAB_SIZE = len(vocalbulary_jp_list)
TGT_VOCAB_SIZE = len(vocalbulary_en_list)
EMB_SIZE = 512
NHEAD = 8
FFN_HID_DIM = 512
BATCH_SIZE = 16
NUM_ENCODER_LAYERS = 3
NUM_DECODER_LAYERS = 3
NUM_EPOCHS = 16
 
# 创建 Seq2SeqTransformer 模型实例
transformer = Seq2SeqTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS,
                                 EMB_SIZE, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE,
                                 FFN_HID_DIM)
 
# 对模型参数进行 Xavier 初始化
for p in transformer.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)
 
print("Model initialized")
# 将模型移动到指定的设备上
transformer = transformer.to(device)
 
# 定义交叉熵损失函数
loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)
 
# 定义优化器
optimizer = torch.optim.Adam(
    transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9
)
# 定义训练函数
def train_epoch(model, train_iter, optimizer):
    """
    对模型进行一个 epoch 的训练，并返回平均损失。
    参数:
    model: 待训练的模型
    train_iter: 训练数据迭代器
    optimizer: 优化器
    返回:
    float: 平均损失
    """
    model.train()
    losses = 0
    for idx, (src, tgt) in  enumerate(train_iter):
        src = src.to(device)
        tgt = tgt.to(device)
 
        tgt_input = tgt[:-1, :]
        # 创建掩码
        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)
        # 前向传播
        logits = model(src, tgt_input, src_mask, tgt_mask,
                                  src_padding_mask, tgt_padding_mask, src_padding_mask)
 
        optimizer.zero_grad()
 
        tgt_out = tgt[1:,:]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        loss.backward()
 
        optimizer.step()
        losses += loss.item()
    return losses / len(train_iter)
 
# 定义评估函数
def evaluate(model, val_iter):
    """
    对模型进行评估，并返回平均损失。
    参数:
    model: 待评估的模型
    val_iter: 验证数据迭代器
    返回:
    float: 平均损失
    """
    model.eval()
    losses = 0
    for idx, (src, tgt) in (enumerate(valid_iter)):
        src = src.to(device)
        tgt = tgt.to(device)
 
        tgt_input = tgt[:-1, :]
        # 创建掩码
        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)
        # 前向传播
        logits = model(src, tgt_input, src_mask, tgt_mask,
                                  src_padding_mask, tgt_padding_mask, src_padding_mask)
        tgt_out = tgt[1:,:]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        losses += loss.item()
    return losses / len(val_iter)
 
# tqdm.tqdm用于生成进度条
for epoch in tqdm.tqdm(range(1, NUM_EPOCHS+1)):
    # 记录一个epoch开始的时间
    start_time = time.time()
    # 调用训练函数进行模型训练，并返回该epoch的平均训练损失
    train_loss = train_epoch(transformer, train_iter, optimizer)
    # 记录一个epoch结束的时间
    end_time = time.time()
    # 打印当前epoch的编号、训练损失以及耗时
    print((f"Epoch: {epoch}, Train loss: {train_loss:.3f}, "
            f"Epoch time = {(end_time - start_time):.3f}s"))
 
def greedy_decode(model, src, src_mask, max_len, start_symbol):
    # 将输入数据移动到设备上
    src = src.to(device)
    src_mask = src_mask.to(device)
    # 使用编码器对源语言进行编码，得到内部记忆
    memory = model.encode(src, src_mask)
    # 初始化目标语言的起始符号
    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(device)
    # 循环生成目标语言的单词直到达到最大长度或者遇到终止符号
    for i in range(max_len-1):
        memory = memory.to(device)
        # 创建目标语言的mask
        memory_mask = torch.zeros(ys.shape[0], memory.shape[0]).to(device).type(torch.bool)
        tgt_mask = (generate_square_subsequent_mask(ys.size(0))
                                    .type(torch.bool)).to(device)
        # 解码得到输出
        out = model.decode(ys, memory, tgt_mask)
        out = out.transpose(0, 1)
        # 从输出中获取概率最高的单词作为下一个输入
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim = 1)
        next_word = next_word.item()
        # 将预测的单词追加到目标语言序列中
        ys = torch.cat([ys,
                        torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
        # 如果预测到终止符号，则停止生成
        if next_word == EOS_IDX:
            break
    return ys
def translate(model, src):
    # 将模型设置为评估模式
    model.eval()
    tokenizer = Tokenizer()
    tokens_jp = [token.surface for token in tokenizer.tokenize(src)]
    # 对源语言进行tokenize，并在开头和结尾加上特殊符号
    tokens = [BOS_IDX] + [vocalbulary_jp[token] if token in vocalbulary_jp else UNK_IDX for token in tokens_jp] + [EOS_IDX]
    num_tokens = len(tokens)
    # 将源语言转换为张量，并创建对应的mask
    src = (torch.LongTensor(tokens).reshape(num_tokens, 1) )
    src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
    # 使用greedy_decode函数生成目标语言的翻译结果
    tgt_tokens = greedy_decode(model,  src, src_mask, max_len=num_tokens + 5, start_symbol=BOS_IDX).flatten()
    # 将生成的目标语言token转换为文本并进行后处理
    return " ".join([vocalbulary_en_list[i] for i in tgt_tokens if i not in [BOS_IDX,PAD_IDX, EOS_IDX]])
 
translate(transformer, "私は元気です。")
 
# save model for inference
torch.save(transformer.state_dict(), 'inference_model')
 
# 保存模型和检查点，以便以后恢复训练
torch.save({
    'epoch': NUM_EPOCHS,  # 当前训练的轮次
    'model_state_dict': transformer.state_dict(),  # 模型的状态字典（包含模型参数）
    'optimizer_state_dict': optimizer.state_dict(),  # 优化器的状态字典（包含优化器参数）
    'loss': train_loss,  # 当前的训练损失
    }, 'model_checkpoint.tar')  # 保存文件的路径和名称
