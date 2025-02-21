import re
import tensorflow as tf
from tensorflow_text.tools.wordpiece_vocab import bert_vocab_from_dataset as bert_vocab
import tensorflow_text as text
import pandas as pd
import random
import numpy as np
import matplotlib.pyplot as plt
import time


"""
https://blog.csdn.net/gzroy/article/details/124460547
"""

corpus_path = 'data/transformer_demo/corpus/fra.txt'
vocab_eng_path = 'data/transformer_demo/vocab/en_vocab.txt'
vocab_fra_path = 'data/transformer_demo/vocab/fra_vocab.txt'
checkpoint_path = 'data/transformer_demo/checkpoints/train'

d_model = 128  # 词嵌入,位置嵌入的维度, 将词从高维映射到的低维
num_heads = 8

"""
读取原始数据
"""
fra = []
eng = []
with open(corpus_path, 'r', encoding='utf8') as f:
    content = f.readlines()
    for line in content:
        temp = line.split(sep='\t')  # 读取语料数据，语料数据用\t分隔开英语和法语
        eng.append(temp[0])
        fra.append(temp[1])

new_fra = []
new_eng = []
for item in fra:
    new_fra.append(re.sub('\s', ' ', item).strip().lower())
for item in eng:
    new_eng.append(re.sub('\s', ' ', item).strip().lower())
"""
将单词处理成Token, 使用Bert
"""
"""
先建立dataset
"""
ds_fra = tf.data.Dataset.from_tensor_slices(new_fra)
ds_eng = tf.data.Dataset.from_tensor_slices(new_eng)
"""
根据dataset产生bert的词汇表
"""
bert_tokenizer_params = dict(lower_case=True)
reserved_tokens = ["[PAD]", "[UNK]", "[START]", "[END]"]

bert_vocab_args = dict(
    # The target vocabulary size
    vocab_size=8000,
    # Reserved tokens that must be included in the vocabulary
    reserved_tokens=reserved_tokens,
    # Arguments for `text.BertTokenizer`
    bert_tokenizer_params=bert_tokenizer_params,
    # Arguments for `wordpiece_vocab.wordpiece_tokenizer_learner_lib.learn`
    learn_params={},
)

fra_vocab = bert_vocab.bert_vocab_from_dataset(
    ds_fra.batch(1000).prefetch(2),  # batch表示将数据集平均分成1000组. prefetch建立了预取的缓存
    **bert_vocab_args
)

en_vocab = bert_vocab.bert_vocab_from_dataset(
    ds_eng.batch(1000).prefetch(2),
    **bert_vocab_args
)
"""
词汇表不是严格按照每个英语单词来划分的，例如'##ers'表示某个单词如果以ers结尾，则会划分出一个'##ers'的token
"""
print(en_vocab[:10])
print(en_vocab[100:110])
print(en_vocab[1000:1010])
print(en_vocab[-10:])


def write_vocab_file(filepath, vocab):
    with open(filepath, 'w', encoding='utf8') as f:
        for token in vocab:
            print(token, file=f)


"""
先生成词汇表txt再导入，得到BertTokenizer，可以对Token处理
"""
write_vocab_file(vocab_fra_path, fra_vocab)
write_vocab_file(vocab_eng_path, en_vocab)

fra_tokenizer = text.BertTokenizer(vocab_fra_path, **bert_tokenizer_params)
en_tokenizer = text.BertTokenizer(vocab_eng_path, **bert_tokenizer_params)

START = tf.argmax(tf.constant(reserved_tokens) == "[START]")
END = tf.argmax(tf.constant(reserved_tokens) == "[END]")


def add_start_end(ragged):
    count = ragged.bounding_shape()[0]
    starts = tf.fill([count, 1], START)
    ends = tf.fill([count, 1], END)
    return tf.concat([starts, ragged, ends], axis=1)


"""
测试token化后的结果。产生的都是数字
"""
sentences = ["Hello Roy!", "The sky is blue.", "Nice to meet you!"]

print(add_start_end(en_tokenizer.tokenize(sentences).merge_dims(1, 2)).to_tensor())

df = pd.DataFrame(data={'fra': new_fra, 'eng': new_eng})

"""
构建数据集
将80%的数据作为训练集，20%的数据作为测试集
"""
# Shuffle the Dataframe
recordnum = df.count()['fra']
indexlist = list(range(recordnum - 1))
random.shuffle(indexlist)
df_train = df.loc[indexlist[:int(recordnum * 0.8)]]
df_val = df.loc[indexlist[int(recordnum * 0.8):]]
"""
区分为训练集和测试集
"""
ds_train = tf.data.Dataset.from_tensor_slices((df_train.fra.values, df_train.eng.values))
ds_val = tf.data.Dataset.from_tensor_slices((df_val.fra.values, df_val.eng.values))
"""
查看训练集的句子最多包含多少个token
"""
lengths = []

for fr_examples, en_examples in ds_train.batch(1024):
    fr_tokens = fra_tokenizer.tokenize(fr_examples)
    lengths.append(fr_tokens.row_lengths())

    en_tokens = en_tokenizer.tokenize(en_examples)
    lengths.append(en_tokens.row_lengths())
    print('.', end='', flush=True)

all_lengths = np.concatenate(lengths)

plt.hist(all_lengths, np.linspace(0, 100, 11))
plt.ylim(plt.ylim())
max_length = max(all_lengths)  # 查看每个句子中包含的最多token数，从而作为MAX_TOKENS的参数
plt.plot([max_length, max_length], plt.ylim())
plt.title(f'Max tokens per example: {max_length}');
plt.show()

"""
生成Batch
"""
BUFFER_SIZE = 20000
BATCH_SIZE = 64
MAX_TOKENS = 67


def filter_max_tokens(fr, en):
    num_tokens = tf.maximum(tf.shape(fr)[1], tf.shape(en)[1])
    return num_tokens < MAX_TOKENS


def tokenize_pairs(fr, en):
    fr = add_start_end(fra_tokenizer.tokenize(fr).merge_dims(1, 2))
    # Convert from ragged to dense, padding with zeros.
    fr = fr.to_tensor()

    en = add_start_end(en_tokenizer.tokenize(en).merge_dims(1, 2))
    # Convert from ragged to dense, padding with zeros.
    en = en.to_tensor()
    return fr, en


def make_batches(ds):
    return (
        ds
        .cache()
        .shuffle(BUFFER_SIZE)
        .batch(BATCH_SIZE)
        .map(tokenize_pairs, num_parallel_calls=tf.data.AUTOTUNE)
        .filter(filter_max_tokens)
        .prefetch(tf.data.AUTOTUNE))


train_batches = make_batches(ds_train)  # 构造BATCH，每个BATCH包括2个tensor，分别代表法语和英语，每个BATCH有64个句子对，并追加开始和结束标识
val_batches = make_batches(ds_val)
"""
生成一个Batch查看下,每个batch包含两个tensor，分别对应法语和英语64个句子转化为token之后的向量，每个句子以token 2开头，以token 0结尾
"""
for a in train_batches.take(1):
    print(a)

"""
添加位置信息
"""


def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    return pos * angle_rates


def positional_encoding(position, d_model):
    """
    对于偶数位置,PE(2i)=sin(pos/10000^(2i/d))
    对于奇数位置,PE(2i+1)=cos(pos/10000^(2i/d))
    Args:
        position: 传入MAX_TOKENS=67, 即每句话最多的TOKEN数量
        d_model: 词嵌入维度=128

    Returns:

    """
    angle_rads = get_angles(np.arange(position)[:, np.newaxis], # 使用np.newaxis将position的维度转换为(67, 1)
                            np.arange(d_model)[np.newaxis, :],  # 使用np.newaxis将d_model的维度转换为(1, 128)
                            d_model)

    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)


"""
创建padding掩码和look ahead掩码
"""


def create_padding_mask(seq):
    """
    padding掩码是为了批量处理中处理不同长度的序列，通常会对较短的序列进行填充（Padding），使其长度与最长序列一致
    1表示需要掩盖掉
    """
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)

    # add extra dimensions to add the padding
    # to the attention logits.
    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)


def create_look_ahead_mask(size):
    """
    1表示需要掩盖掉
    """
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)  # 生成全为1的上三角矩阵(右上部分为1,不含对角线,左下部分为0)
    return mask  # (seq_len, seq_len)


"""
Attention的计算, mask是一个参数
q, k相乘,防止点积过大除以一个数,加上padding(如有),过softmax,乘v
"""


def scaled_dot_product_attention(q, k, v, mask):
    """Calculate the attention weights.
    q, k, v must have matching leading dimensions.
    k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
    The mask has different shapes depending on its type(padding or look ahead)
    but it must be broadcastable for addition.
    Args:
    q: query shape == (..., seq_len_q, depth)
    k: key shape == (..., seq_len_k, depth)
    v: value shape == (..., seq_len_v, depth_v)
    mask: Float tensor with shape broadcastable
          to (..., seq_len_q, seq_len_k). Defaults to None.
    Returns:
    output, attention_weights
    """

    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)
    # transpose_b=True表示对第二个矩阵做转置,因此(seq_len_q,depth)与矩阵(depth,seq_len_k)相乘得到(seq_len_q,seq_len_k)
    # scale matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)  # 矩阵相乘结果除以根号16(128/8)

    # add the mask to the scaled tensor.
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)  # mask的位置乘比较小的数

    # softmax is normalized on the last axis (seq_len_k) so that the scores
    # add up to 1.  最后一个维度的分数加起来为1
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

    return output, attention_weights


"""
多头Attention, 封装成了一个Keras的Layer.调用了单头Attention
输入乘以每个头的权重矩阵线性变化，qk相乘，防止点积过大除以一个数，padding(如有),过softmax,乘v,拼接,线性变化
实际训练中每个头的权重放在一起训练的,后续将不同的头分离出来
"""


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, *, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(d_model)  # 无激活函数,因此是线性变换
        self.wk = tf.keras.layers.Dense(d_model)  # 无激活函数,因此是线性变换
        self.wv = tf.keras.layers.Dense(d_model)  # 无激活函数,因此是线性变换

        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        """
        最后一维度是128,分成维度数*深度
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])  # tf.transpose修改维度的顺序

    def call(self, v, k, q, mask):
        # 在Encoder中v,k,q是一样的,在Decoder中mh1是一样的,mh2的v,k是一样的。但是注意他们经过了不同的线性变换.
        batch_size = tf.shape(q)[0]  # query的第一个维度是batch号

        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth) 简单的reshape操作
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, mask)

        scaled_attention = tf.transpose(scaled_attention,
                                        perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

        concat_attention = tf.reshape(scaled_attention,
                                      (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model) 将最后2个维度拼接起来

        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model) 经过一个dense网络,无激活函数,线性变换

        return output, attention_weights


"""
对于比较简单的网络,直接用函数来定义了
"""


def point_wise_feed_forward_network(d_model, dff):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
        tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
    ])


"""
封装一个encoder层,输入是向量[64(每批次包含的句子如),32(每个句子最多的token,应建模成最多句子的维度),128(每个token建模的维度)]
"""


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, *, d_model, num_heads, dff, rate=0.1):
        super(EncoderLayer, self).__init__()
        self.mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)  # feed forward network
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)  # LN而非BN。BN:同一特征不同样本做归一化 例如1*10*128,代表一句话有10个token,每个token有128维度,对每个token128维度做归一化操作
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)  # LN而非BN。LN:同一样本不同特征做归一化。如果这128维度某个维度有异常数据，计算attention时候可能有影响
        # 因为各样本的序列长度可能不同，不好在样本之间做归一化
        # 由于每一个特征的gamma和beta不一样,所以归一化前后的排序不同
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):
        attn_output, _ = self.mha(x, x, x, mask)  # (batch_size, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)  # 将x加入,即ResNet的实现. Add+LN

        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model) 减均值除以sqrt(方差+epsilon),然后放缩gamma和beta
        return out2


"""
编码器, 包含N个EncoderLayer,在此基础上新定义了:encoder的层数,以及embedding的步骤
embedding+乘一个数+position_encoding+encode_layer*6+dropout
encode_layer:多头注意力+dropout&Add&LN+ffn+dropout+Add&LN
"""


class Encoder(tf.keras.layers.Layer):
    def __init__(self, *, num_layers, d_model, num_heads, dff, input_vocab_size, rate=0.1):
        super(Encoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)
        self.pos_encoding = positional_encoding(MAX_TOKENS, self.d_model)

        self.enc_layers = [
            EncoderLayer(d_model=d_model, num_heads=num_heads, dff=dff, rate=rate)
            for _ in range(num_layers)]

        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):
        seq_len = tf.shape(x)[1]  # 第二个维度表示该句有几个token

        # adding embedding and position encoding.
        x = self.embedding(x)  # (batch_size, input_seq_len, d_model)  # 每个token转变成128维度
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))  # embedding后的结果乘根号128
        x += self.pos_encoding[:, :seq_len, :]  # 加上位置padding,注意pos_encoding是个常量,该句有多长就只需要取多少即可

        x = self.dropout(x, training=training)  # 附带了一个dropout层,当training=True时做dropout,当为False时不做dropout

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)

        return x  # (batch_size, input_seq_len, d_model)


"""
解码层
"""


class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, *, d_model, num_heads, dff, rate=0.1):
        super(DecoderLayer, self).__init__()
        self.mha1 = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
        self.mha2 = MultiHeadAttention(d_model=d_model, num_heads=num_heads)

        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.dropout3 = tf.keras.layers.Dropout(rate)

    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
        # enc_output.shape == (batch_size, input_seq_len, d_model)

        attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask)  # (batch_size, target_seq_len, d_model)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)

        attn2, attn_weights_block2 = self.mha2(
            enc_output, enc_output, out1, padding_mask)  # (batch_size, target_seq_len, d_model)  交叉注意力
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(attn2 + out1)  # (batch_size, target_seq_len, d_model)

        ffn_output = self.ffn(out2)  # (batch_size, target_seq_len, d_model)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(ffn_output + out2)  # (batch_size, target_seq_len, d_model)

        return out3, attn_weights_block1, attn_weights_block2


"""
解码器
Embedding+乘一个数+pos_encoding+dropout+decode_layer*6
decode_layer:多头注意力+dropout&Add&LN记为q+多头注意力(k和v是encoder的输入)+dropout&Add&LN+fnn+drouput&Add&LN
"""


class Decoder(tf.keras.layers.Layer):
    def __init__(self, *, num_layers, d_model, num_heads, dff, target_vocab_size,
                 rate=0.1):
        super(Decoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Embedding(target_vocab_size, d_model)
        self.pos_encoding = positional_encoding(MAX_TOKENS, d_model)

        self.dec_layers = [
            DecoderLayer(d_model=d_model, num_heads=num_heads, dff=dff, rate=rate)
            for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
        seq_len = tf.shape(x)[1]  # 在预测阶段,x是已经生成的文本;在训练阶段,x是目标文本
        attention_weights = {}

        x = self.embedding(x)  # (batch_size, target_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x, block1, block2 = self.dec_layers[i](x, enc_output, training, look_ahead_mask, padding_mask)

            attention_weights[f'decoder_layer{i + 1}_block1'] = block1
            attention_weights[f'decoder_layer{i + 1}_block2'] = block2

        # x.shape == (batch_size, target_seq_len, d_model)
        return x, attention_weights


"""
组装为整体的模型
"""


class Transformer(tf.keras.Model):
    def __init__(self, *, num_layers, d_model, num_heads, dff, input_vocab_size,
                 target_vocab_size, rate=0.1):
        super().__init__()
        self.encoder = Encoder(num_layers=num_layers, d_model=d_model,
                               num_heads=num_heads, dff=dff,
                               input_vocab_size=input_vocab_size, rate=rate)

        self.decoder = Decoder(num_layers=num_layers, d_model=d_model,
                               num_heads=num_heads, dff=dff,
                               target_vocab_size=target_vocab_size, rate=rate)

        self.final_layer = tf.keras.layers.Dense(target_vocab_size)  # 输出维度是词汇表的大小

    def call(self, inputs, training):
        # Keras models prefer if you pass all your inputs in the first argument
        inp, tar = inputs  # input中含有输入和target输出

        padding_mask, look_ahead_mask = self.create_masks(inp, tar)

        enc_output = self.encoder(inp, training, padding_mask)  # (batch_size, inp_seq_len, d_model)
        # 为啥enc_output里边每一列几乎都是一样的
        # dec_output.shape == (batch_size, tar_seq_len, d_model)
        dec_output, attention_weights = self.decoder(
            tar, enc_output, training, look_ahead_mask, padding_mask)

        final_output = self.final_layer(dec_output)  # (batch_size, tar_seq_len, target_vocab_size)

        return final_output, attention_weights

    def create_masks(self, inp, tar):
        # Encoder padding mask (Used in the 2nd attention block in the decoder too.)
        padding_mask = create_padding_mask(inp)  # 查看每一个token是否等于0

        # Used in the 1st attention block in the decoder.
        # It is used to pad and mask future tokens in the input received by
        # the decoder.
        look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])  # 针对target生成look ahead mask
        dec_target_padding_mask = create_padding_mask(tar)  # 查看每一个token是否等于0
        look_ahead_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)  # 两个mask取较大值,只有2个掩码都是0,混合掩码才是0. 只要有一个是1,那么结果就是1,1表示需要掩盖

        return padding_mask, look_ahead_mask


"""
自定义的learning rate
"""


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


learning_rate = CustomSchedule(d_model)
"""
采用Adam优化器
"""
optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
"""
模型的预测值是token的序号，可以理解为类别。因此采用类别的交叉熵来计算Loss值
"""
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')


def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_sum(loss_) / tf.reduce_sum(mask)


def accuracy_function(real, pred):
    accuracies = tf.equal(real, tf.argmax(pred, axis=2))  # 真实值与预测中最大概率的词相等的数量

    mask = tf.math.logical_not(tf.math.equal(real, 0))  # 真实值中不是0的词数量，即总数
    accuracies = tf.math.logical_and(mask, accuracies)

    accuracies = tf.cast(accuracies, dtype=tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    return tf.reduce_sum(accuracies) / tf.reduce_sum(mask)


train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.Mean(name='train_accuracy')

input_vocab_size = 0  # Encoder的Embedding层维度，等于法语词汇表大小
target_vocab_size = 0  # Decoder的Embedding层维度，等于英语词汇表大小
num_layers = 6  # Attention层的个数
dff = 128  # FFN中间层的维度
dropout_rate = 0.1

with open(vocab_fra_path, 'r', encoding='utf8') as f:
    input_vocab_size = len(f.readlines())
with open(vocab_eng_path, 'r', encoding='utf8') as f:
    target_vocab_size = len(f.readlines())

"""
实例化一个transformer
"""
transformer = Transformer(
    num_layers=num_layers,
    d_model=d_model,
    num_heads=num_heads,
    dff=dff,
    input_vocab_size=input_vocab_size,
    target_vocab_size=target_vocab_size,
    rate=dropout_rate)
"""
定义checkpoint在训练过程中保存模型
"""


# 定义两个trackable object需要保存
"""
这里Checkpoint的参数喂什么就存什么，即transformer和optimizer
"""
ckpt = tf.train.Checkpoint(transformer=transformer, optimizer=optimizer)

ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

# if a checkpoint exists, restore the latest checkpoint.
if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)
    print('Latest checkpoint restored!!')
"""
定义训练函数
"""
EPOCHS = 0

# The @tf.function trace-compiles train_step into a TF graph for faster
# execution. The function specializes to the precise shape of the argument
# tensors. To avoid re-tracing due to the variable sequence lengths or variable
# batch sizes (the last batch is smaller), use input_signature to specify
# more generic shapes.

train_step_signature = [
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
]


@tf.function(input_signature=train_step_signature)
def train_step(inp, tar):
    tar_inp = tar[:, :-1]
    tar_real = tar[:, 1:]
    print(tar_real)
    with tf.GradientTape() as tape:
        predictions, _ = transformer([inp, tar_inp], training=True)
        loss = loss_function(tar_real, predictions)

    gradients = tape.gradient(loss, transformer.trainable_variables)
    optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))

    train_loss(loss)
    train_accuracy(accuracy_function(tar_real, predictions))


"""
开始训练
"""
for epoch in range(EPOCHS):
    start = time.time()

    train_loss.reset_states()
    train_accuracy.reset_states()

    # inp -> portuguese, tar -> english
    for (batch, (inp, tar)) in enumerate(train_batches):
        try:
            train_step(inp, tar)
        except ValueError:
            print(inp)
            print('-------')
            print(tar)

        if batch % 50 == 0:
            print(
                f'Epoch {epoch + 1} Batch {batch} Loss {train_loss.result():.4f} Accuracy {train_accuracy.result():.4f}')

    if (epoch + 1) % 5 == 0:
        ckpt_save_path = ckpt_manager.save()
        print(f'Saving checkpoint for epoch {epoch + 1} at {ckpt_save_path}')

    print(f'Epoch {epoch + 1} Loss {train_loss.result():.4f} Accuracy {train_accuracy.result():.4f}')

    print(f'Time taken for 1 epoch: {time.time() - start:.2f} secs\n')

"""
模型预测
"""


class Translator(tf.Module):
    START = tf.argmax(tf.constant(reserved_tokens) == "[START]")  # 等价于print(tf.constant(2, dtype=tf.int64) == START)
    END = tf.argmax(tf.constant(reserved_tokens) == "[END]")  # 从保留token中找到开始和结束

    def __init__(self, fr_tokenizer, en_tokenizer, transformer):
        self.fr_tokenizer = fr_tokenizer
        self.en_tokenizer = en_tokenizer
        self.transformer = transformer

    def _add_start_end(self, ragged):
        """
        填充开始和结束
        """
        count = ragged.bounding_shape()[0]  # 有几行,一般输入只有1行
        starts = tf.fill([count, 1], START)
        ends = tf.fill([count, 1], END)
        return tf.concat([starts, ragged, ends], axis=1)

    def __call__(self, sentence, max_length=MAX_TOKENS):
        # input sentence is french, hence adding the start and end token
        assert isinstance(sentence, tf.Tensor)
        if len(sentence.shape) == 0:
            sentence = sentence[tf.newaxis]  # shape转换为1
        # print(sentence)
        # print(self.fr_tokenizer.tokenize(sentence))
        # print(self.fr_tokenizer.tokenize(sentence).merge_dims(1,2))
        sentence = self._add_start_end(self.fr_tokenizer.tokenize(sentence).merge_dims(1, 2)).to_tensor()  # 将输入的句子做分词

        encoder_input = sentence

        # As the output language is english, initialize the output with the
        # english start token.
        # start_end = self.en_tokenizer.tokenize([''])[0]
        start_end = self._add_start_end(en_tokenizer.tokenize(['']).merge_dims(1, 2))[0]
        start = start_end[0][tf.newaxis]
        end = start_end[1][tf.newaxis]

        # `tf.TensorArray` is required here (instead of a python list) so that the
        # dynamic-loop can be traced by `tf.function`.
        output_array = tf.TensorArray(dtype=tf.int64, size=0, dynamic_size=True)
        output_array = output_array.write(0, start)  # 先将英语的开始标志加入输出文本中

        for i in tf.range(max_length):
            """
            循环最大长度(67),每次都基于已生成的结果调用transformer生成预测的下一个词语
            """
            output = tf.transpose(output_array.stack())  # 将output_array合并成一句话，即已预测出的token
            predictions, _ = self.transformer([encoder_input, output], training=False)

            # select the last token from the seq_len dimension. 取产生的最后一个token
            predictions = predictions[:, -1:, :]  # (batch_size, 1, vocab_size)

            predicted_id = tf.argmax(predictions, axis=-1)  # 理论上这里应该有个softmax概率分布,不过这里直接取概率最高的词了

            # concatentate the predicted_id to the output which is given to the decoder
            # as its input.
            output_array = output_array.write(i + 1, predicted_id[0])

            if predicted_id == end:  # 如果预测的词是结束则退出
                break

        output = tf.transpose(output_array.stack())
        # output.shape (1, tokens)
        text = en_tokenizer.detokenize(output)[0]  # shape: ()

        # tokens = en_tokenizer.lookup(output)[0]

        # `tf.function` prevents us from using the attention_weights that were
        # calculated on the last iteration of the loop. So recalculate them outside
        # the loop.
        _, attention_weights = self.transformer([encoder_input, output[:, :-1]], training=False)  # 最后一个词是结束标志, 忽略

        # return text, tokens, attention_weights
        return text, attention_weights


translator = Translator(fra_tokenizer, en_tokenizer, transformer)

"""
辅助函数, 打印输入的句子、输出和预测的句子
"""


def print_translation(sentence, tokens, ground_truth):
    prediction_text = []
    tokens_numpy = tokens.numpy()
    for i in range(1, tokens_numpy.shape[0] - 1):
        prediction_text.append(tokens_numpy[i].decode("utf-8"))
    prediction_text = ' '.join(prediction_text)
    print(f'{"Input:":15s}: {sentence}')
    print(f'{"Prediction":15s}: {prediction_text}')
    print(f'{"Ground truth":15s}: {ground_truth}')


sentence = "c’est une histoire tellement triste."
ground_truth = "this is such a sad story."

translated_text, attention_weights = translator(
    tf.constant(sentence))
print_translation(sentence, translated_text, ground_truth)

sentence = "Ces pratiques sont essentiellement inefficaces et peuvent entraîner des risques pour la santé et la pollution de l'environnement."
ground_truth = "These practices are essentially ineffective, and can cause health hazards and environmental pollution."

translated_text, attention_weights = translator(
    tf.constant(sentence))
print_translation(sentence, translated_text, ground_truth)

sentence = "Il fait beau aujourd'hui."
ground_truth = "It's a fine day today."

translated_text, attention_weights = translator(
    tf.constant(sentence))
print_translation(sentence, translated_text, ground_truth)


sentence = "Les étudiants doivent bien étudier."
ground_truth = "Students should study hard."

translated_text, attention_weights = translator(
    tf.constant(sentence))
print_translation(sentence, translated_text, ground_truth)