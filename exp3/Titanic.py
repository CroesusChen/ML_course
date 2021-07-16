# 数据分析
import pandas as pd
import numpy as np
# 绘图
import matplotlib.pyplot as plt
import seaborn as sns

df_train = pd.read_csv('./data/titanic/train.csv')
df_test = pd.read_csv('./data/titanic/test.csv')

# 填充数据值
def fillna_data(df_train, df_test):
    # 对训练集和测试集中的"Age"数据进行平均值填充
    df_train['Age'] = df_train['Age'].fillna(df_train['Age'].mean())
    df_test['Age'] = df_test['Age'].fillna(df_test['Age'].mean())
    # 添加一个新的类别"Missing"来填充"Cabin"
    df_train['Cabin'] = df_train['Cabin'].fillna('Missing')
    df_test['Cabin'] = df_test['Cabin'].fillna('Missing')
    # 用出现频率最多的类别填充训练集中的"Embarked"属性
    df_train['Embarked'] = df_train['Embarked'].fillna(
        df_train['Embarked'].mode()[0])
    # 用出现频率最多的类别填充测试集中的"Fare"属性
    df_test['Fare'] = df_test['Fare'].fillna(
        df_test['Fare'].mode()[0])

    return df_train, df_test

# 得到填充后的数据集 df_train， df_test
df_train, df_test = fillna_data(df_train, df_test)
# sns.barplot(x='Pclass', y='Survived', data=df_train,
#             palette="Set1",
#             errwidth=1.2,
#             errcolor="0.1",
#             capsize=0.05,
#             alpha=0.6)
# plt.show()

id_test = df_test.loc[:, 'PassengerId']

# 第一次处理
# 去掉了以下特征
# 即对 Pclass Sex Age SibSp Parch Embarked 分析
# df_train = df_train.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin', 'Fare'])
# df_test = df_test.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin', 'Fare'])
# # 第二次处理
# # 在第一次的基础上，添加了归一化处理的特征 Fare
# # 即对 Pclass Sex Age SibSp Parch Fare Embarked 分析
# df_train = df_train.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'])
# df_test = df_test.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'])
# 第三次处理
# 在第二次的基础上，去掉了特征 SibSp Parch
df_train = df_train.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp', 'Parch'])
df_test = df_test.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp', 'Parch'])


# 对数据集中的字符串数据进行编码处理
def preprocess_data(train, test):
    # 使用one-hot编码将登船港口"Embarked"进行转换
    # 训练集
    Embarked = pd.get_dummies(train['Embarked'], prefix='Embarked')
    tmp_train = pd.concat([train, Embarked], axis=1)
    tmp_train.drop(columns=['Embarked'], inplace=True)
    # 测试集
    Embarked = pd.get_dummies(test['Embarked'], prefix='Embarked')
    tmp_test = pd.concat([test, Embarked], axis=1)
    tmp_test.drop(columns=['Embarked'], inplace=True)
    # 将年龄归一化
    tmp_train['Age'] = (tmp_train['Age'] - tmp_train['Age'].min()) / (tmp_train['Age'].max() - tmp_train['Age'].min())
    tmp_test['Age'] = (tmp_test['Age'] - tmp_test['Age'].min()) / (tmp_test['Age'].max() - tmp_test['Age'].min())
    # 将船票价格归一化
    if 'Fare' in tmp_train.columns:
        tmp_train['Fare'] = (tmp_train['Fare'] - tmp_train['Fare'].min()) / (
                    tmp_train['Fare'].max() - tmp_train['Fare'].min())
    if 'Fare' in tmp_test.columns:
        tmp_test['Fare'] = (tmp_test['Fare'] - tmp_test['Fare'].min()) / (
                    tmp_test['Fare'].max() - tmp_test['Fare'].min())
    # 将性别"Sex"这一特征从字符串映射至数值
    # 0代表female，1代表male
    gender_class = {'female': 0, 'male': 1}
    tmp_train['Sex'] = tmp_train['Sex'].map(gender_class)
    tmp_test['Sex'] = tmp_test['Sex'].map(gender_class)

    return tmp_train, tmp_test

data_train, data_test = preprocess_data(df_train, df_test)

label_train = data_train.loc[:, 'Survived']
data_train = data_train.drop(columns=['Survived'])
data_test = data_test.drop(columns=['Survived'])

from sklearn.model_selection import train_test_split
'''
从原始数据集（source）中拆分出训练数据集（用于模型训练train），测试数据集（用于模型评估test）
train_test_split是交叉验证中常用的函数，功能是从样本中随机的按比例选取train data和test data
train_data：所要划分的样本特征集
train_target：所要划分的样本结果
test_size：样本占比，如果是整数的话就是样本的数量
'''

# 建立模型用的训练数据集和测试数据集
train_X, test_X, train_y, test_y = train_test_split(data_train,
                                                    label_train,
                                                    train_size=.8)

def SVM():
    from sklearn import svm
    '''
     SVM函数参数解析：
     C:float, default=1.0
         正则化参数。正则化的强度与C成反比，必须是严格的正数。惩罚是一个平方的l2惩罚。
     gamma:{‘scale’, ‘auto’} or float, default=’scale’
         rbf'、'poly'和'sigmoid'的内核系数。
         如果gamma='scale'（默认）被执行，那么它使用1/（n_features * X.var()）作为gamma的值。
         如果是'auto'，则使用1/n_features。
     decision_function_shape:{‘ovo’, ‘ovr’}, default=’ovr’
         多分类问题选择'ovo'
    '''
    clf_SVM = svm.SVC(C=2, gamma=0.4, kernel='rbf')
    # 训练SVM模型
    clf_SVM.fit(train_X, train_y)
    from sklearn.metrics import confusion_matrix, classification_report

    pred_SVM = clf_SVM.predict(test_X)

    # 混淆矩阵
    print(confusion_matrix(test_y, pred_SVM))
    '''
     classification_report函数用于显示主要分类指标的文本报告
     显示每个类的精确度，召回率，F1值等信息
     混淆矩阵 TP FP
              FN TN
    '''
    print(classification_report(test_y, pred_SVM))
    from sklearn.model_selection import cross_val_score

    # 在训练集和测试集上的准确性
    train_acc_SVM = cross_val_score(clf_SVM, train_X, train_y, cv=10, scoring='accuracy')
    test_acc_SVM = cross_val_score(clf_SVM, test_X, test_y, cv=10, scoring='accuracy')

    print('SVM Model on Train Data Accuracy: %f' %(train_acc_SVM.mean()))
    print('SVM Model on Test Data Accuracy: %f' %(test_acc_SVM.mean()))

    pred = clf_SVM.predict(data_test)
    output_SVM = pd.DataFrame({'PassengerId': id_test,'Survived': pred})
    output_SVM.to_csv('./output/submission_SVM.csv',index = False)
    print('submission_SVM.csv生成完毕！')


def RandomForest():
    from sklearn.ensemble import RandomForestClassifier

    clf_RFC = RandomForestClassifier()  # 未填参数，需调优
    # 训练随机森林分类器模型
    clf_RFC.fit(train_X,train_y)

    from sklearn.metrics import confusion_matrix,classification_report

    pred_RFC = clf_RFC.predict(test_X)

    # 混淆矩阵
    print(confusion_matrix(test_y,pred_RFC))
    # 分类报告
    print(classification_report(test_y, pred_RFC))
    from sklearn.model_selection import cross_val_score

    # 在训练集和测试集上的准确性
    train_acc_RFC = cross_val_score(clf_RFC,train_X,train_y,cv = 10,scoring = 'accuracy')
    test_acc_RFC = cross_val_score(clf_RFC,test_X,test_y,cv = 10,scoring = 'accuracy')

    print('Random Forest Classifier Model on Train Data Accuracy: %f' % (train_acc_RFC.mean()))
    print('Random Forest Classifier Model on Test Data Accuracy: %f' % (test_acc_RFC.mean()))

    pred = clf_RFC.predict(data_test)
    output_RFC = pd.DataFrame({'PassengerId': id_test,'Survived': pred})
    output_RFC.to_csv('./output/submission_RFC.csv',index = False)
    print('submission_RFC.csv生成完毕！')


def BPNetwork():
    from sklearn.neural_network import MLPClassifier

    # 两个隐藏层，第一层为64个神经元，第二层为32个神经元
    mlp = MLPClassifier(hidden_layer_sizes = (64,32),activation = 'relu',
                        solver = 'adam',
                        max_iter = 800)
    # 训练神经网络
    mlp.fit(train_X,train_y)
    from sklearn.metrics import confusion_matrix, classification_report

    pred_BP = mlp.predict(test_X)

    # 混淆矩阵
    print(confusion_matrix(test_y, pred_BP))
    # 分类报告
    print(classification_report(test_y,pred_BP))
    train_acc_BP = mlp.score(train_X,train_y)
    test_acc_BP = mlp.score(test_X,test_y)

    print('MLP Classifier Model on Train Data Accuracy: %f' % (train_acc_BP))
    print('MLP Classifier Model on Test Data Accuracy: %f' % (test_acc_BP))

    pred = mlp.predict(data_test)
    output_BP = pd.DataFrame({'PassengerId': id_test,'Survived': pred})
    output_BP.to_csv('./output/submission_BP.csv',index = False)
    print('submission_BP.csv生成完毕！')


# SVM()
# RandomForest()
BPNetwork()



