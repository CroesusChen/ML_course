import torch
from torchtext import data

SEED = 1234
torch.manual_seed(SEED)  # 为CPU设置随机种子
torch.cuda.manual_seed(SEED)  #为GPU设置随机种子
# 在程序刚开始加这条语句可以提升一点训练速度，没什么额外开销
torch.backends.cudnn.deterministic = True

# torchtext.data.Field : 用来定义字段的处理方法（文本字段，标签字段）
TEXT = data.Field(tokenize='spacy',tokenizer_language='en_core_web_sm')
#LabelField是Field类的一个特殊子集，专门用于处理标签。
LABEL = data.LabelField(dtype=torch.float)

# 加载IMDB电影评论数据集
from torchtext import datasets
train_data, test_data = datasets.IMDB.splits(TEXT, LABEL)

import random
# 默认split_ratio=0.7
train_data, valid_data = train_data.split(random_state=random.seed(SEED))

# 从预训练的词向量（vectors）中，将当前(corpus语料库)词汇表的词向量抽取出来，构成当前 corpus 的 Vocab（词汇表）
# 预训练的 vectors 来自glove模型，每个单词有100维。glove模型训练的词向量参数来自很大的语料库
# 而我们的电影评论的语料库小一点，所以词向量需要更新，glove的词向量适合用做初始化参数。
TEXT.build_vocab(train_data, max_size=25000, vectors="glove.6B.100d", unk_init=torch.Tensor.normal_)
LABEL.build_vocab(train_data)

BATCH_SIZE = 64
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 相当于把样本划分batch，知识多做了一步，把相等长度的单词尽可能的划分到一个batch，不够长的就用padding。
train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
    (train_data, valid_data, test_data),
    batch_size = BATCH_SIZE,
    device = device
)
import torch.nn as nn
import torch.nn.functional as F

class WordAVGModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, output_dim, pad_idx):
        # 初始化参数
        super().__init__()

        # embedding的作用就是将每个单词变成一个词向量
        # vocab_size=词汇表长度，embedding_dim=每个单词的维度
        # padding_idx：如果提供的话，输出遇到此下标时用零填充。这里如果遇到padding的单词就用0填充。
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)

        # output_dim输出的维度，一个数就可以了，=1
        self.fc = nn.Linear(embedding_dim, output_dim)

    def forward(self, text):  # text维度为(sent_len, 1)
        embedded = self.embedding(text)
        # text 下面会指定，为一个batch的数据
        # embedded = [sent_len, batch_size, emb_dim]
        # sen_len 一条评论的单词数
        # batch_size 一个batch有多少条评论
        # emb_dim 一个单词的维度
        # 假设[sent_len, batch_size, emb_dim] = (1000, 64, 100)
        # 则进行运算: (text: 1000, 64, 25000)*(self.embedding: 1000, 25000, 100) = (1000, 64, 100)

        # [batch_size, sent_len, emb_dim] 更换顺序
        embedded = embedded.permute(1, 0, 2)

        # [batch_size, embedding_dim]把单词长度的维度压扁为1，并降维
        # embedded 为input_size，(embedded.shape[1], 1)) 为kernel_size
        # squeeze(1)表示删除索引为1的那个维度
        pooled = F.avg_pool2d(embedded, (embedded.shape[1], 1)).squeeze(1)

        # (batch_size, embedding_dim)*(embedding_dim, output_dim) = (batch_size, output_dim)
        return self.fc(pooled)


class RNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim,
                 n_layers, bidirectional, dropout, pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)

        # embedding_dim: 每个词向量的维度
        # hidden_dim: 隐藏层的维度
        # num_layers: 神经网络深度，纵向深度
        # bidrectional: 是否双向循环RNN
        # dropout是指在深度学习网络的训练过程中，对于神经网络单元，按照一定的概率将其暂时从网络中丢弃。
        # 经过交叉验证，隐含节点dropout率等于0.5的时候效果最好，原因是0.5的时候dropout随机生成的网络结构最多。
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers,
                           bidirectional=bidirectional, dropout=dropout)

        self.fc = nn.Linear(hidden_dim * 2, output_dim)  # *2是因为BiLSTM
        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        embedded = self.dropout(self.embedding(text))  # [sent len, batch size, emb dim]

        # output = [sent len, batch size, hid dim * num directions]
        # hidden = [num layers * num directions, batch size, hid dim]
        # cell = [num layers * num directions, batch size, hid dim]
        output, (hidden, cell) = self.rnn(embedded)

        # concat the final forward (hidden[-2,:,:]) and backward (hidden[-1,:,:]) hidden layers
        # and apply dropout
        # [batch size, hid dim * num directions], 横着拼接的
        # 倒数第一个和倒数第二个是BiLSTM最后要保留的状态
        hidden = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))

        return self.fc(hidden.squeeze)

INPUT_DIM = len(TEXT.vocab)  # 25002
EMBEDDING_DIM = 100
HIDDEN_DIM = 256
OUTPUT_DIM = 1
N_LAYERS = 2
BIDIRECTIONAL = True
DROPOUT = 0.5

# PAD_IDX = 1 为pad的索引
PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]

model = RNN(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM,
            N_LAYERS, BIDIRECTIONAL, DROPOUT, PAD_IDX)

# model = WordAVGModel(INPUT_DIM, EMBEDDING_DIM, OUTPUT_DIM, PAD_IDX)
# 统计参数数量
def count_parameters(model):
  # numel()函数：返回数组中元素的个数
  return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f'The model has {count_parameters(model):,} trainable parameters')

# 把上面vectors="glove.6B.100d"取出的词向量作为初始化参数
# 数量为25000*100个参数，25000个单词，每个单词的词向量维度为100
pretrained_embeddings = TEXT.vocab.vectors
model.embedding.weight.data.copy_(pretrained_embeddings)
UNK_IDX = TEXT.vocab.stoi[TEXT.unk_token]  # UNK_IDX = 0

# 词汇表25002个单词，前两个unk和pad也需要初始化，把它们初始化为0
model.embedding.weight.data[UNK_IDX] = torch.zeros(EMBEDDING_DIM)
model.embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)


print("train model start...")
import torch.optim as optim
# 定义优化器
optimizer = optim.Adam(model.parameters())
# 定义损失函数，这个BCEWithLogitsLoss特殊情况，二分类损失函数
criterion = nn.BCEWithLogitsLoss()
# 送到GPU上去
model = model.to(device)
criterion = criterion.to(device)


# 计算预测的准确率

def binary_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """

    # .round函数 四舍五入，rounded_preds要么为0，要么为1
    # neg为0, pos为1
    rounded_preds = torch.round(torch.sigmoid(preds))

    # convert into float for division
    """
    a = torch.tensor([1, 1])
    b = torch.tensor([1, 1])
    print(a == b)
    output: tensor([1, 1], dtype=torch.uint8)
  
    a = torch.tensor([1, 0])
    b = torch.tensor([1, 1])
    print(a == b)
    output: tensor([1, 0], dtype=torch.uint8)
    """
    correct = (rounded_preds == y).float()
    acc = correct.sum() / len(correct)

    return acc

def train(model, iterator, optimizer, criterion):
    epoch_loss = 0
    epoch_acc = 0
    total_len = 0

    # model.train()代表了训练模式
    # model.train() ：启用 BatchNormalization 和 Dropout
    # model.eval() ：不启用 BatchNormalization 和 Dropout
    model.train()

    # iterator为train_iterator
    for batch in iterator:
        # 梯度清零，加这步防止梯度叠加
        optimizer.zero_grad()

        # batch.text 就是上面forward函数的参数text
        # 压缩维度，不然跟 batch.label 维度对不上
        predictions = model(batch.text).squeeze(1)

        loss = criterion(predictions, batch.label)
        acc = binary_accuracy(predictions, batch.label)

        loss.backward()  # 反向传播
        optimizer.step()  # 梯度下降

        # loss.item() 以及本身除以了 len(batch.label)
        # 所以得再乘一次，得到一个batch的损失，累加得到所有样本损失
        epoch_loss += loss.item() * len(batch.label)

        # (acc.item(): 一个batch的正确率) * batch数 = 正确数
        # train_iterator 所有batch的正确数累加
        epoch_acc += acc.item() * len(batch.label)

        # 计算 train_iterator 所有样本的数量，应该是17500
        total_len += len(batch.label)

    # epoch_loss / total_len ：train_iterator所有batch的损失
    # epoch_acc / total_len ：train_iterator所有batch的正确率
    return epoch_loss / total_len, epoch_acc / total_len

# 不用优化器了
def evaluate(model, iterator, criterion):
    epoch_loss = 0
    epoch_acc = 0
    total_len = 0

    # 转成测试模式，冻结dropout层或其他层
    model.eval()

    with torch.no_grad():
        # iterator为valid_iterator
        for batch in iterator:
            # 没有反向传播和梯度下降

            predictions = model(batch.text).squeeze(1)
            loss = criterion(predictions, batch.label)
            acc = binary_accuracy(predictions, batch.label)

            epoch_loss += loss.item() * len(batch.label)
            epoch_acc += acc.item() * len(batch.label)
            total_len += len(batch.label)

    # 调回训练模式
    model.train()

    return epoch_loss / total_len, epoch_acc / total_len
import time

# 查看每个epoch的时间
def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


N_EPOCHS = 10
best_valid_loss = float('inf')  # 初试的验证集loss设置为无穷大
for epoch in range(N_EPOCHS):
    start_time = time.time()

    train_loss, train_acc = train(model, train_iterator, optimizer, criterion)
    valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)

    end_time = time.time()

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    # 只要模型效果变好，就存模型(参数)
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'wordavg-model.pt')

    print(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
    print(f'\t Val. Loss: {valid_loss:.3f} | Val. Acc: {valid_acc * 100:.2f}%')

# 用保存的模型参数预测数据
model.load_state_dict(torch.load("wordavg-model.pt"))
# spacy是分词工具，跟NLTK类似
import spacy

nlp = spacy.load('en_core_web_sm')

def predict_sentiment(sentence):
    # 分词
    tokenized = [tok.text for tok in nlp.tokenizer(sentence)]
    # sentence 的索引
    indexed = [TEXT.vocab.stoi[t] for t in tokenized]

    tensor = torch.LongTensor(indexed).to(device)  # seq_len
    tensor = tensor.unsqueeze(1)  # seq_len * batch_size (1)

    # tensor与text一样的tensor
    prediction = torch.sigmoid(model(tensor))

    return prediction.item()

predict_sentiment("I love this film bad")
test_loss, test_acc = evaluate(model, test_iterator, criterion)
print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%')






