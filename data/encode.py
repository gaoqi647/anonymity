import numpy as np
import logging

# 设置日志记录
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


def validate_input(feature):
    """
    Validate whether the input is a list or a numpy array.
    If not, raise a ValueError.
    """
    if not isinstance(feature, (list, np.ndarray)):
        raise ValueError("Input feature must be a list or a numpy array")


def generate_unique_values(feature):
    """
    Generate a list of unique values from the input feature.
    :param feature: Input feature
    :return: List of unique values
    """
    return np.unique(feature)


def create_encoding(unique_values):
    """
    Create a dictionary to encode unique values with random numbers.
    :param unique_values: List of unique values
    :return: Encoding dictionary
    """
    encoding = {}
    for value in unique_values:
        encoding[value] = np.random.rand()
    return encoding


def encode_feature(feature, encoding):
    """
    Encode the input feature using the provided encoding dictionary.
    :param feature: Input feature
    :param encoding: Encoding dictionary
    :return: Encoded feature
    """
    return [encoding[value] for value in feature]


def random_encode(feature):
    """
    Randomly encode the input feature.
    :param feature: Input feature (list or numpy array)
    :return: Encoded feature
    """
    validate_input(feature)

    logging.debug("Generating unique values...")
    unique_values = generate_unique_values(feature)
    logging.debug(f"Unique values: {unique_values}")

    logging.debug("Creating encoding dictionary...")
    encoding = create_encoding(unique_values)
    logging.debug(f"Encoding dictionary: {encoding}")

    logging.debug("Encoding the feature...")
    encoded_feature = encode_feature(feature, encoding)
    logging.debug(f"Encoded feature: {encoded_feature}")

    return encoded_feature


def process_data(data, features):
    """
    处理数据集，生成特征编码。

    :param data: 包含原始数据的DataFrame
    :param features: 需要检查的特征列表
    :return: 包含二进制编码值的DataFrame
    """
    # 初始化编码字典
    encode = {f'feature{i}': [] for i in range(len(features))}

    # 遍历数据集中的每一行
    for _, row in data.iterrows():
        # 检查每个特征是否存在于当前行的数据中
        for j, feat in enumerate(features):
            label = any(feat[k] in str(row['Data Augmentation']).replace(' ', '').lower() for k in range(len(feat)))
            encode[f'feature{j}'].append(int(label))

    # 将编码字典转换为DataFrame
    encode_df = pd.DataFrame(encode)

    # 生成二进制编码值
    re = [int('0b' + ''.join(str(val) for val in row), 2) for _, row in encode_df.iterrows()]

    return pd.DataFrame({'value': re})


def read_and_prepare_data(file_path):
    """
    读取Excel文件并准备数据。

    :param file_path: Excel文件路径
    :return: 处理后的DataFrame
    """
    try:
        tmp = pd.read_excel(file_path)
        # 将前6列的数据转换为字符串
        for j in range(6):
            tmp.iloc[:, j] = tmp.iloc[:, j].astype(str).str.strip()
        return tmp
    except Exception as e:
        logging.error(f"Failed to read and prepare data from {file_path}: {e}")
        raise


def create_empty_dataframe(columns, shape):
    """
    创建一个空的DataFrame。

    :param columns: 列名列表
    :param shape: DataFrame的形状 (行数, 列数)
    :return: 空的DataFrame
    """
    arr = np.zeros(shape)
    return pd.DataFrame(arr, columns=columns)


def match_and_fill(n_data, tmp, columns):
    """
    匹配并填充新的DataFrame。

    :param n_data: 空的DataFrame
    :param tmp: 处理后的原始DataFrame
    :param columns: 列名列表
    :return: 填充后的DataFrame
    """
    try:
        for i in range(tmp.shape[0]):
            for j in range(len(columns)):
                if any(columns[j] == tmp.iloc[i, k] for k in range(6)):
                    n_data.iloc[i, j] = 1
        return n_data
    except Exception as e:
        logging.error(f"Failed to match and fill data: {e}")
        raise


def save_data(n_data, output_file):
    """
    保存处理后的数据到Excel文件。

    :param n_data: 处理后的DataFrame
    :param output_file: 输出文件路径
    """
    try:
        n_data.to_excel(output_file, index=False)
        logging.info(f"Data saved to {output_file}")
    except Exception as e:
        logging.error(f"Failed to save data to {output_file}: {e}")
        raise
#
# # test
# if __name__ == "__main__":
#     try:
#         feature = [1, 2, 3, 2, 1, 4, 5, 3, 2, 1]
#         encoded_feature = random_encode(feature)
#         print(f"Encoded feature: {encoded_feature}")
#     except Exception as e:
#         logging.error(f"An error occurred: {e}")