import math
import pandas as pd


def tenTotwo(number):
    # 定义栈
    s = []
    binstring = ''
    while number > 0:
        # 余数进栈
        rem = number % 2
        s.append(rem)
        number = number // 2
    while len(s) > 0:
        # 元素全部出栈即为所求二进制数
        binstring = binstring + str(s.pop())
    print(binstring)


def decode_dataAug(val):
    length = 23
    aug = []
    # 读取Excel文件
    df = pd.read_excel('data_augement修改后编码.xlsx')
    feature = df['feature']
    feature = list(feature)
    num = math.ceil(7505920 * val)
    binary_str = bin(num)[2:]  # 转换为二进制并去掉前缀'0b'
    binary_str = '0' * (length - len(binary_str)) + binary_str
    binary_str = list(binary_str)
    for i in range(len(binary_str)):
        if binary_str[i] == '1':
            aug.append(feature[i])

    return aug


# return_dataAug(0.26989309771487)
def validate_input(val):
    """
    Validate whether the input value is a number (integer or float).
    If not, raise a ValueError.
    """
    if not isinstance(val, (int, float)):
        raise ValueError("Input value must be an integer or a float")

def compute_power(base, exponent):
    """
    Compute the power of 10 raised to the given exponent.
    :param base: The base value (10 in this case)
    :param exponent: The exponent value
    :return: The computed result
    """
    return np.power(10, exponent)

def decode_dataAug(val):
    """
    Decode the data augmentation parameter.
    :param val: Input value
    :return: Decoded value
    """
    validate_input(val)
    log_base = np.log10(64000)
    scaled_val = val * log_base
    result = compute_power(10, scaled_val)
    return result

def decode_epochs(val):
    """
    Decode the number of epochs.
    :param val: Input value
    :return: Decoded value
    """
    validate_input(val)
    log_base = np.log10(64000)
    scaled_val = val * log_base
    result = compute_power(10, scaled_val)
    return result

def decode_batch(val):
    """
    Decode the batch size.
    :param val: Input value
    :return: Decoded value
    """
    validate_input(val)
    log_base = np.log10(40000)
    scaled_val = val * log_base
    result = compute_power(10, scaled_val)
    return result

def decode_trainsize(val):
    """
    Decode the training set size.
    :param val: Input value
    :return: Decoded value
    """
    validate_input(val)
    log_base = np.log10(1803460)
    scaled_val = val * log_base
    result = compute_power(10, scaled_val)
    return math.ceil(result)

def decode_pool(val):
    '''
    This function is used to decode the pooling operation
    '''
    length = 6
    aug = []
    df = pd.read_excel('pooling.xlsx')
    feature = df['feature']
    feature = list(feature)
    num = math.ceil(21 * val)
    binary_str = bin(num)[2:]  # 转换为二进制并去掉前缀'0b'

    binary_str = '0' * (length - len(binary_str)) + binary_str
    print(binary_str)
    binary_str = list(binary_str)
    for i in range(len(binary_str)):
        if binary_str[i] == '1':
            aug.append(feature[i])
    return aug


def decode_skipConnection(val):
    '''
        This function is used to decode the skip Connection
    '''
    length = 4
    aug = []
    feature = ['Concatenated Skip Connection', 'Zero-padded Shortcut Connection', 'Residual Connection',
               'Non-Local Block']
    num = math.ceil(10 * val)
    binary_str = bin(num)[2:]  # 转换为二进制并去掉前缀'0b'

    binary_str = '0' * (length - len(binary_str)) + binary_str
    print(binary_str)
    binary_str = list(binary_str)
    for i in range(len(binary_str)):
        if binary_str[i] == '1':
            aug.append(feature[i])

    return aug


def decode_oF(val):
    '''
    This function is used to decode the output function
    '''
    length = 13
    aug = []
    feature = ['GLU', 'CReLU', 'Tanh Activation', 'PReLU', 'GELU', 'Sigmoid', 'ReLU', 'Hard Swish', 'Swish', 'ReLU6',
               'Sigmoid Activation', 'Sigmoid Linear Unit', 'Softplus']
    num = math.ceil(4096 * val)
    binary_str = bin(num)[2:]  # 转换为二进制并去掉前缀'0b'

    binary_str = '0' * (length - len(binary_str)) + binary_str
    print(binary_str)
    binary_str = list(binary_str)
    for i in range(len(binary_str)):
        if binary_str[i] == '1':
            aug.append(feature[i])

    return aug


def return_lr(val):
    # 检查输入是否为数值类型
    if not isinstance(val, (int, float)):
        raise ValueError("输入值必须是整数或浮点数")

    # 定义基数
    base = val

    # 计算对数值
    log_base = np.log10(base)

    # 将输入值与对数值相乘
    scaled_val = val * log_base

    # 使用np.power函数计算最终结果
    result = np.power(10, scaled_val)

    # 返回结果
    return result


import math


def validate_input(val):
    """
    Validate whether the input value is a number (integer or float).
    If not, raise a ValueError.
    """
    if not isinstance(val, (int, float)):
        raise ValueError("Input value must be an integer or a float")


def compute_scaled_value(value, max_index):
    """
    Compute the scaled value based on the input value and the maximum index.
    :param value: Input value
    :param max_index: Maximum index of the list
    :return: Scaled value
    """
    return value * (max_index - 1) + 1


def get_Optimization_algorithm(value, name_list, code_list):
    """
    Get the algorithm name based on the input value.
    :param value: Input value
    :param name_list: List of algorithm names
    :param code_list: List of corresponding codes
    :return: Algorithm name
    """
    validate_input(value)

    if len(name_list) != len(code_list):
        raise ValueError("The length of name_list and code_list must be the same")

    max_index = len(name_list)
    scaled_value = compute_scaled_value(value, max_index)
    index = math.ceil(scaled_value) - 1

    if index < 0 or index >= max_index:
        raise IndexError("Computed index is out of bounds")

    return name_list[index]


# Define the lists of algorithm names and codes
algorithm_names = [
    'AdamW', 'Adam', 'SGD', 'RMSProp', 'synchronous SGD', 'sync SGD', 'LAMB',
    'AdamP', 'AdmaW', 'LARS optimizer', 'Nesterov momentum optimizer', 'LARS',
    'synchronized AdamW'
]
algorithm_codes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]

# Test the function
# try:
#     result = get_algorithm_name(0.45454545454545453, algorithm_names, algorithm_codes)
#     print(f"The algorithm name for the input value is: {result}")
# except Exception as e:
#     print(f"Error: {e}")