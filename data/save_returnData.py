import pandas as pd
import numpy as np
import logging
import configparser

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def read_config(config_file):
    """
    Read configuration file.

    :param config_file: Path to the configuration file
    :return: Configuration object
    """
    config = configparser.ConfigParser()
    config.read(config_file)
    return config


def min_max_normalization(value):
    """
    Min-Max normalization, also known as range scaling.
    Formula: (original value - minimum value) / (maximum value - minimum value)
    :param value: Values or array to be normalized
    :return: Normalized values and normalization parameters
    """
    min_val = value.min()
    max_val = value.max()
    if max_val == min_val:
        raise ValueError("All values are the same; cannot perform min-max normalization.")
    new_value = (value - min_val) / (max_val - min_val)
    return new_value, {'max': max_val, 'min': min_val}


def log_transfer(value):
    """
    Logarithmic transformation.
    Formula: log10(x) / log10(max)
    :param value: Values or array to be transformed
    :return: Transformed values and transformation parameters
    """
    if (value <= 0).any():
        raise ValueError("All values must be greater than 0 for log transformation.")
    max_val = value.max()
    if max_val <= 0:
        raise ValueError("Maximum value must be greater than 0 for log transformation.")
    new_value = np.log10(value) / np.log10(max_val)
    return new_value, {'logMax': max_val}


def normalize_data(data, config):
    """
    Normalize and log transform the data, and store the normalization parameters.

    :param data: DataFrame to be processed
    :param config: Configuration object
    :return: Processed DataFrame and normalization parameters
    """
    tmp_data = {}
    normalization_columns = config.get('Columns', 'normalization_columns').split(',')
    log_columns = config.get('Columns', 'log_columns').split(',')

    for col in normalization_columns:
        new_value, params = min_max_normalization(data[col])
        tmp_data[col] = params
        data[col] = new_value

    for col in log_columns:
        new_value, params = log_transfer(data[col])
        tmp_data[col] = params
        data[col] = new_value

    return data, tmp_data


def save_normalized_data(data, tmp_data, output_file, params_file):
    """
    Save the processed data and normalization parameters.

    :param data: Processed DataFrame
    :param tmp_data: Normalization parameters
    :param output_file: Path to the output file
    :param params_file: Path to the parameters file
    """
    try:
        data.to_csv(output_file, index=False)
        logging.info(f"Normalized data saved to {output_file}")

        params_df = pd.DataFrame(tmp_data)
        params_df.to_csv(params_file, index=False)
        logging.info(f"Normalization parameters saved to {params_file}")
    except Exception as e:
        logging.error(f"Failed to save data: {e}")
        raise


def main(config_file):
    """
    Main function to execute the data processing workflow.

    :param config_file: Path to the configuration file
    """
    try:
        # Read configuration file
        config = read_config(config_file)

        # Load input file
        input_file = config.get('Files', 'input_file')
        data = pd.read_excel(input_file)
        logging.info(f"Data loaded from {input_file}")

        # Normalize and log transform
        normalized_data, tmp_data = normalize_data(data, config)

        # Save the processed data and normalization parameters
        output_file = config.get('Files', 'output_file')
        params_file = config.get('Files', 'params_file')
        save_normalized_data(normalized_data, tmp_data, output_file, params_file)

        # Print the result
        print(normalized_data)
    except Exception as e:
        logging.error(f"An error occurred: {e}")


if __name__ == "__main__":
    # Define the path to the configuration file
    config_file = 'config.ini'

    # Run the main function
    main(config_file)