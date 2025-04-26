import pandas as pd
import logging

# 设置日志记录
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


def validate_input(data, column_name):
    """
    Validate whether the input is a DataFrame and the specified column exists.
    If not, raise a ValueError.
    """
    if not isinstance(data, pd.DataFrame):
        raise ValueError("Input data must be a pandas DataFrame")
    if column_name not in data.columns:
        raise ValueError(f"Column '{column_name}' does not exist in the DataFrame")


def split_and_normalize(text, delimiters):
    """
    Split the text by the given delimiters and normalize the resulting substrings.
    :param text: Input text
    :param delimiters: List of delimiters
    :return: List of normalized substrings
    """
    import re
    pattern = '|'.join(map(re.escape, delimiters))
    parts = re.split(pattern, text)
    return [part.strip().lower().replace(' ', '') for part in parts if part]


def process_data_augmentation1(data, column_name):
    """
    Process the data augmentation column in the DataFrame.
    :param data: Input DataFrame
    :param column_name: Name of the column to process
    :return: List of processed values
    """
    validate_input(data, column_name)

    logging.debug("Splitting and normalizing the data...")
    delimiters = [',', ' and', 'and ', ' and ']
    tmp = data[column_name].str.split('|'.join(map(re.escape, delimiters)), expand=True)

    li = []
    for i in range(tmp.shape[0]):
        for j in range(tmp.shape[1]):
            if pd.notna(tmp.iloc[i][j]):
                normalized_value = split_and_normalize(tmp.iloc[i][j], delimiters)[0]
                li.append(normalized_value)

    logging.debug(f"Processed values: {li}")
    return li


def process_data_augmentation2(data, column_name):
    """
    Process the data augmentation column in the DataFrame.
    :param data: Input DataFrame
    :param column_name: Name of the column to process
    :return: DataFrame with counts of each unique value
    """
    validate_input(data, column_name)

    logging.debug("Splitting and normalizing the data...")
    delimiters = [',', ' and', 'and ', ' and ']
    tmp = data[column_name].str.split('|'.join(map(re.escape, delimiters)), expand=True)

    li = []
    for i in range(tmp.shape[0]):
        for j in range(tmp.shape[1]):
            if pd.notna(tmp.iloc[i][j]) and isinstance(tmp.iloc[i][j], str):
                normalized_value = split_and_normalize(tmp.iloc[i][j], delimiters)[0]
                li.append(normalized_value)

    logging.debug("Counting occurrences of each unique value...")
    counter = Counter(li)
    sorted_x = sorted(counter.items(), key=lambda x: x[1], reverse=True)

    dic = {'name': [], 'num': []}
    for item in sorted_x:
        dic['name'].append(item[0])
        dic['num'].append(item[1])

    df = pd.DataFrame(dic)

    logging.debug("Creating a set of unique values...")
    unique_values = set()
    for i in range(tmp.shape[0]):
        for j in range(tmp.shape[1]):
            if pd.notna(tmp.iloc[i][j]) and isinstance(tmp.iloc[i][j], str):
                unique_values.add(tmp.iloc[i][j].strip())

    logging.debug(f"Unique values: {unique_values}")

    return df, unique_values


def validate_input2(file_path):
    """
    Validate whether the input file path is a string and the file exists.
    If not, raise a ValueError.
    """
    if not isinstance(file_path, str):
        raise ValueError("Input file path must be a string")
    if not pd.io.common.file_exists(file_path):
        raise ValueError(f"File '{file_path}' does not exist")


def normalize_text(text):
    """
    Normalize the text by removing spaces, converting to lowercase, and splitting by '、'.
    :param text: Input text
    :return: List of normalized substrings
    """
    if not isinstance(text, str):
        return []
    text = text.replace(' ', '').lower()
    return text.split('、')


def process_features(data):
    """
    Process the features in the DataFrame.
    :param data: Input DataFrame
    :return: List of lists containing processed features
    """
    feature = []
    for i in range(data.shape[0]):
        tmp = data.iloc[i][0]
        normalized_values = normalize_text(tmp)
        feature.append(normalized_values)

    logging.debug(f"Processed features: {feature}")
    return feature


def read_and_process_excel(file_path):
    """
    Read an Excel file and process the features.
    :param file_path: Path to the Excel file
    :return: List of lists containing processed features
    """
    validate_input2(file_path)

    logging.debug(f"Reading Excel file: {file_path}")
    data = pd.read_excel(file_path)

    logging.debug("Processing features...")
    feature = process_features(data)

    return feature


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


def main():
    try:
        # 假设data是从某个CSV文件加载的DataFrame
        data = pd.read_csv('your_data.csv')  # 请替换为实际的文件路径

        # 特征列表，每个元素都是一个包含关键词的列表
        features = [
            ['aug1'],  # 替换为实际的特征关键词
            ['aug2'],  # ...
            # 添加更多的特征...
        ]

        # 处理数据
        result = process_data(data, features)

        # 保存结果到Excel文件
        result.to_excel('value.xlsx', index=False)

        # 记录完成信息
        logging.info("Data processing and saving completed.")

        # 打印结果
        print(result)

    except Exception as e:
        logging.error(f"An error occurred: {e}")


def min_max_normalization(value, feature_range=(0, 1)):
    """
    归一化，也称离差标准化
    公式：(原始值-最小值)/(最大值-最小值) * (max_range - min_range) + min_range
    :param value: 待归一化的数值或数组
    :param feature_range: 目标范围，默认为 (0, 1)
    :return: 归一化后的值，值域为指定的 feature_range
    """
    min_val = value.min()
    max_val = value.max()
    if max_val == min_val:
        raise ValueError("All values are the same; cannot perform min-max normalization.")
    new_value = (value - min_val) / (max_val - min_val) * (feature_range[1] - feature_range[0]) + feature_range[0]
    return new_value


def log_transfer(value, base=10):
    """
    log转换
    公式：log_base(x) / log_base(max)
    :param value: 待转换的数值或数组
    :param base: 对数的底数，默认为 10
    :return: 转换后的值，值域 [0, 1]
    """
    if (value <= 0).any():
        raise ValueError("All values must be greater than 0 for log transformation.")
    max_val = value.max()
    if max_val <= 0:
        raise ValueError("Maximum value must be greater than 0 for log transformation.")
    new_value = np.log(value) / np.log(max_val) if base == np.e else np.log10(value) / np.log10(max_val)
    return new_value


def normalization(value):
    """
    标准化
    公式：(原始值-均值)/标准差
    :param value: 待标准化的数值或数组
    :return: 标准化后的值，均值为 0，标准差为 1
    """
    mean_val = value.mean()
    std_val = value.std()
    if std_val == 0:
        raise ValueError("Standard deviation is zero; cannot perform normalization.")
    new_value = (value - mean_val) / std_val
    return new_value


def proportional_normalization(value):
    """
    比例归一
    公式：值 / 总和
    :param value: 待归一化的数值或数组
    :return: 归一化后的值，值域 [0, 1]
    """
    sum_val = value.sum()
    if sum_val == 0:
        raise ValueError("Sum of values is zero; cannot perform proportional normalization.")
    new_value = value / sum_val
    return new_value


def arctan_normalization(value, scale_factor=1.0):
    """
    反正切归一化
    公式：反正切值 * (2 / pi) * scale_factor
    :param value: 待归一化的数值或数组
    :param scale_factor: 缩放因子，默认为 1.0
    :return: 归一化后的值，值域 [-scale_factor, scale_factor]
    """
    new_value = np.arctan(value) * (2 / np.pi) * scale_factor
    return new_value


import pandas as pd
import numpy as np
import logging

def min_max_normalization2(value):
    """
    Min-Max normalization, also known as range scaling.
    Formula: (original value - minimum value) / (maximum value - minimum value)
    :param value: Values or array to be normalized
    :return: Normalized values, with range [0, 1]
    """
    min_val = value.min()
    max_val = value.max()
    if max_val == min_val:
        raise ValueError("All values are the same; cannot perform min-max normalization.")
    new_value = (value - min_val) / (max_val - min_val)
    return new_value


def load_data(file_path):
    """
    Load data from an Excel file.

    :param file_path: Path to the Excel file
    :return: DataFrame containing the loaded data
    """
    try:
        data = pd.read_excel(file_path)
        logging.info(f"Data loaded from {file_path}")
        return data
    except Exception as e:
        logging.error(f"Failed to load data from {file_path}: {e}")
        raise


def normalize_and_transform_data(data, normalization_columns, log_columns):
    """
    Normalize and log transform specified columns in the DataFrame.

    :param data: DataFrame to be processed
    :param normalization_columns: List of columns to be normalized using min-max normalization
    :param log_columns: List of columns to be transformed using logarithmic transformation
    :return: Processed DataFrame
    """
    try:
        for col in normalization_columns:
            data[col] = min_max_normalization(data[col])
            logging.info(f"Min-Max normalization applied to column: {col}")

        for col in log_columns:
            data[col] = log_transfer(data[col])
            logging.info(f"Log transformation applied to column: {col}")

        return data
    except Exception as e:
        logging.error(f"Failed to normalize and transform data: {e}")
        raise


def save_data(data, output_file):
    """
    Save the processed data to a CSV file.

    :param data: Processed DataFrame
    :param output_file: Path to the output CSV file
    """
    try:
        data.to_csv(output_file, index=False)
        logging.info(f"Processed data saved to {output_file}")
    except Exception as e:
        logging.error(f"Failed to save data to {output_file}: {e}")
        raise


def main2(input_file, output_file, normalization_columns, log_columns):
    """
    Main function to execute the data processing workflow.

    :param input_file: Path to the input Excel file
    :param output_file: Path to the output CSV file
    :param normalization_columns: List of columns to be normalized using min-max normalization
    :param log_columns: List of columns to be transformed using logarithmic transformation
    """
    try:
        # Load data
        data = load_data(input_file)

        # Normalize and log transform data
        data = normalize_and_transform_data(data, normalization_columns, log_columns)

        # Save processed data
        save_data(data, output_file)

        # Print the result
        print(data)
    except Exception as e:
        logging.error(f"An error occurred: {e}")


if __name__ == "__main__":
    # Define input and output file paths
    input_file = 'tmp.xlsx'
    output_file = '归一化.csv'

    # Define columns to be normalized and log transformed
    normalization_columns = [
        'Normalization', 'Initialization', 'Convolutions', 'Position Embeddings',
        'Pooling Operations', 'Regularization', 'Data Augmentagion',
        'Feedforward Networks', 'Attention Mechanisms', 'Attention Modules',
        'Skip Connections', 'Activation Functions', 'Learning Rate Schedules',
        'Training Algorithm', 'Output Functions', 'Hardware'
    ]

    log_columns = [
        'Dataset', 'train_size', 'test_size', 'number_label', 'cos', 'JS', 'L2',
        'Test_Input', 'Input_Size', 'Framwork', 'paraeters', 'batch size',
        'learning rate', 'epochs', 'Number'
    ]

    # Run the main function
    main2(input_file, output_file, normalization_columns, log_columns)