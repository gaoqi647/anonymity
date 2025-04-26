import pickle
import numpy as np
import sklearn.gaussian_process as gp
import random
import warnings
from scipy.stats import norm
from scipy.optimize import minimize
import pandas as pd

warnings.filterwarnings("ignore")


def sample_loss(params):
    """
    Loads a pre-trained model from a pickle file and predicts the output using the provided parameters.

    Args:
    params (list): A list of 26 input features to be used for prediction.

    Returns:
    float: The predicted top-1 accuracy score from the loaded model.
    """
    with open('result.pickle', 'rb') as f:
        model = pickle.load(f)  # Load the model from the pickle file
    top_1 = model.predict(np.array(params).reshape(1, -1))  # Predict using the loaded model
    return top_1[0]


def gama_expected_improvement(x, gaussian_process, evaluated_loss, greater_is_better=False, n_params=1, a=1, b=1):
    """
    This method is used to balance exploration and exploitation in optimization by considering not just the expected improvement
    but also incorporating the variance and probability of improvement with scaling factors `a` and `b`.

    :param x: Input parameter(s) to be evaluated, assumed to be a single point or set of points.
    :param gaussian_process: A trained Gaussian Process model for predicting mean and variance at the given point.
    :param evaluated_loss: The losses observed so far (the objective function values already evaluated).
    :param greater_is_better: If True, the goal is to maximize the objective function, else minimize it.
    :param n_params: Number of parameters (dimensions) of the input `x`.
    :param a: Scaling factor for the expected improvement.
    :param b: Scaling factor for the variance term (exploration).

    :return: A combined measure of expected improvement, probability of improvement, and variance.
    """

    # Ensure that `x` is reshaped as a column vector or set of column vectors to match the input format of the Gaussian Process
    x_to_predict = x.reshape(-1, n_params)

    # Predict the mean (`mu`) and standard deviation (`sigma`) of the Gaussian process at the input `x`
    try:
        mu, sigma = gaussian_process.predict(x_to_predict, return_std=True)
    except Exception as e:
        raise ValueError(f"Error in Gaussian Process prediction: {e}")

    # Determine the optimal value of the evaluated loss (either max or min based on `greater_is_better`)
    if greater_is_better:
        loss_optimum = np.max(evaluated_loss)  # Maximize the objective function
    else:
        loss_optimum = np.min(evaluated_loss)  # Minimize the objective function

    # Determine the scaling factor for improvement calculations (inversion for minimization problems)
    scaling_factor = (-1) ** (not greater_is_better)

    # Safeguard against cases where sigma equals zero (avoid division by zero)
    with np.errstate(divide='ignore'):
        Z = scaling_factor * (mu - loss_optimum) / sigma  # Standardized improvement

        # Calculate the classical Expected Improvement (EI)
        expected = scaling_factor * (mu - loss_optimum) * norm.cdf(Z) + sigma * norm.pdf(Z)

        # Handle cases where sigma is zero (no uncertainty about the prediction)
        expected[sigma == 0.0] = 0.0

    # Recompute the scaling factor for further calculations (probability of improvement)
    scaling_factor = (-1) ** (not greater_is_better)

    # Recalculate Z for probability estimation
    with np.errstate(divide='ignore'):
        Z = scaling_factor * (mu - loss_optimum) / sigma

    # Calculate the probability of improvement based on the current model predictions
    probability = norm.cdf(Z)

    # Variance (sigma^2) is used to enhance exploration (exploit variance for uncertain regions)
    variance = sigma ** 2

    # Compute the modified Expected Improvement formula with scaling terms `a` and `b` for exploration-exploitation trade-off
    # a * Expected Improvement + Probability of Improvement + b * Variance
    modified_ei = a * expected + probability + b * variance

    return modified_ei


def sample_next_hyperparameter(acquisition_func, gaussian_process, evaluated_loss, bounds, greater_is_better=False,
                               n_restarts=25, patience=5, tol=1e-6, diversification=True):
    """
    Finds the next hyperparameter to sample using the acquisition function with added complexity and diversification.

    Args:
    acquisition_func (function): Acquisition function to optimize.
    gaussian_process (GaussianProcessRegressor): Trained GP model.
    evaluated_loss (np.array): The current loss values from evaluated samples.
    bounds (np.array): Bounds for the hyperparameters.
    greater_is_better (bool): Whether we are maximizing or minimizing the loss.
    n_restarts (int): Number of restarts for the optimization process.
    patience (int): The number of iterations to tolerate without significant improvement.
    tol (float): The tolerance for significant improvement.
    diversification (bool): Whether to apply diversification strategies.

    Returns:
    np.array: The next sample point for the hyperparameters.
    """
    best_x = None
    best_acquisition_value = float('inf') if not greater_is_better else -float('inf')
    n_params = bounds.shape[1]

    # Tracking for patience and stagnation
    no_improvement_counter = 0

    # Randomly choose starting points for the optimization
    indexs = [random.randint(0, bounds.shape[0] - 1) for _ in range(n_restarts)]

    for restart_idx, i in enumerate(indexs):
        starting_point = bounds[i]

        print(f"Restart {restart_idx + 1}/{n_restarts}, starting point: {starting_point}")

        try:
            # Attempt to minimize the acquisition function
            res = minimize(fun=acquisition_func,
                           x0=starting_point.reshape(1, -1),
                           bounds=bounds,
                           method='L-BFGS-B',
                           args=(gaussian_process, evaluated_loss, greater_is_better, n_params))

            if res.success:
                print(f"Optimization successful: {res.fun} at {res.x}")
                # Check if the result improves significantly
                improvement = (best_acquisition_value - res.fun) if not greater_is_better else (
                            res.fun - best_acquisition_value)

                if abs(improvement) > tol:
                    best_acquisition_value = res.fun
                    best_x = res.x
                    no_improvement_counter = 0  # Reset the counter when improvement is made
                else:
                    no_improvement_counter += 1
                    print(f"No significant improvement. Counter: {no_improvement_counter}")
            else:
                print(f"Optimization failed at restart {restart_idx + 1}")

        except Exception as e:
            print(f"Error during optimization at restart {restart_idx + 1}: {str(e)}")

        # Early stopping criteria if no improvement after several iterations
        if no_improvement_counter >= patience:
            print(f"Early stopping after {restart_idx + 1} restarts due to lack of improvement.")
            break

    # Diversification strategy: if no significant progress, try a random sample
    if diversification and best_x is None:
        random_index = random.randint(0, bounds.shape[0] - 1)
        best_x = bounds[random_index]
        print(f"Diversification: Sampling randomly at index {random_index}, with point {best_x}")

    return best_x if best_x is not None else starting_point  # Fallback to the last starting point


def bayesian_optimisation(n_iters, sample_loss, bounds, x0=None, y0=None, n_pre_samples=50,
                          gp_params=None, random_search=False, alpha=1e-5, epsilon=1e-7, a=1, b=1):
    """
    Implements Bayesian Optimization using a Gaussian Process (GP) to find the optimal hyperparameters.

    Args:
    n_iters (int): Number of optimization iterations.
    sample_loss (function): Function to sample the loss (objective function).
    bounds (np.array): Bounds for the hyperparameters.
    x0 (np.array): Initial sampled points for the hyperparameters.
    y0 (np.array): Initial loss values corresponding to x0.
    gp_params (dict): Parameters for the Gaussian Process Regressor.
    random_search (bool): Whether to perform random search.
    alpha (float): GP alpha value for noise level.
    epsilon (float): Minimum distance between points to avoid duplication.
    a, b (float): Coefficients to control the influence of EI and variance.

    Returns:
    tuple: The list of sampled hyperparameters and their corresponding loss values.
    """
    x_list = []
    y_list = []

    if x0 is None:
        # Perform random sampling for the first n_pre_samples points
        initial_indices = [random.randint(0, bounds.shape[0] - 1) for _ in range(n_pre_samples)]
        for params in initial_indices:
            x_list.append(bounds[params])
            y_list.append(sample_loss(bounds[params]))
    else:
        x_list = list(x0)
        y_list = list(y0)

    xp = np.array(x_list)
    yp = np.array(y_list)

    # Create and train the Gaussian Process Regressor
    if gp_params is not None:
        model = gp.GaussianProcessRegressor(**gp_params)
    else:
        kernel = gp.kernels.Matern()
        model = gp.GaussianProcessRegressor(kernel=kernel, alpha=alpha, n_restarts_optimizer=10, normalize_y=True)

    for n in range(n_iters):
        model.fit(xp, yp)

        # Select the next hyperparameter sample point
        if random_search:
            if random.random() < 0.1:  # Introduce randomness in exploration
                next_sample = bounds[random.randint(0, bounds.shape[0] - 1)]
            else:
                ei = [gama_expected_improvement(bounds[i], model, yp, greater_is_better=True, n_params=26, a=a, b=b)
                      for i in range(bounds.shape[0])]
                next_sample = bounds[np.argmax(ei)]
        else:
            next_sample = sample_next_hyperparameter(gama_expected_improvement, model, yp, greater_is_better=True,
                                                     bounds=bounds, n_restarts=100)

        # Ensure that the new point isn't too close to existing points
        if np.any(np.linalg.norm(next_sample - xp, axis=1) <= epsilon):
            continue

        # Sample the loss for the next sample point
        cv_score = sample_loss(next_sample)

        # Update the list of samples and losses
        x_list.append(next_sample)
        y_list.append(cv_score)
        xp = np.array(x_list)
        yp = np.array(y_list)

    return np.array(x_list), np.array(y_list)


# Data preprocessing function
def preprocess_data(file_path):
    # Load the dataset and handle encoding issues
    data = pd.read_csv(file_path, encoding='GB2312')

    # Extract labels and remove unnecessary columns
    labels = data['Top-1'].values
    data.drop(['Architecture', 'Top-1', 'Initialization', 'Feedforward Networks', 'Attention Mechanisms', 'test_size',
               'number_label', 'cos'], axis=1, inplace=True)

    # Handle missing values (you can add more sophisticated handling if needed)
    data.fillna(data.mean(), inplace=True)

    # Extract features and return the processed data
    features = data.values
    return features, labels


# Function to create parameter grid from given bounds
def create_param_grid(param_dict):
    param_grid = np.array([[Architecture, Normalization, Convolutions, Skip_Connections, Position_Embeddings,
                            Activation_Functions, Pooling_Operations, Regularization, Data_Augmentagion,
                            Attention_Modules, Learning_Rate_Schedules, Training_Algorithm, Output_Functions,
                            Dataset, train_size, JS, L2, Test_Input, Input_Size, Framwork, paraeters, batch_size,
                            learning_rate, epochs, Hardware, Number]
                           for Architecture in param_dict['Architecture'] for Normalization in
                           param_dict['Normalization']
                           for Convolutions in param_dict['Convolutions']
                           for Skip_Connections in param_dict['Skip_Connections']
                           for Position_Embeddings in param_dict['Position_Embeddings']
                           for Activation_Functions in param_dict['Activation_Functions']
                           for Pooling_Operations in param_dict['Pooling_Operations']
                           for Regularization in param_dict['Regularization']
                           for Data_Augmentagion in param_dict['Data_Augmentagion']
                           for Attention_Modules in param_dict['Attention_Modules']
                           for Learning_Rate_Schedules in param_dict['Learning_Rate_Schedules']
                           for Training_Algorithm in param_dict['Training_Algorithm']
                           for Output_Functions in param_dict['Output_Functions']
                           for Dataset in param_dict['Dataset']
                           for train_size in param_dict['train_size']
                           for JS in param_dict['JS']
                           for L2 in param_dict['L2']
                           for Test_Input in param_dict['Test_Input']
                           for Input_Size in param_dict['Input_Size']
                           for Framwork in param_dict['Framwork']
                           for paraeters in param_dict['paraeters']
                           for batch_size in param_dict['batch_size']
                           for learning_rate in param_dict['learning_rate']
                           for epochs in param_dict['epochs']
                           for Hardware in param_dict['Hardware']
                           for Number in param_dict['Number']])

    np.random.shuffle(param_grid)  # Shuffle the grid for randomness
    return param_grid


# Complex Bayesian search function integrating data preprocessing and optimization
def search_best(a, b, data_file, param_dict, n_iters=500):
    """
    Performs Bayesian Optimization to find the best hyperparameter configuration.

    Args:
    a, b (float): Coefficients to control the influence of Expected Improvement (EI) and variance.
    data_file (str): Path to the CSV file containing the data.
    param_dict (dict): Dictionary containing hyperparameter configurations.
    n_iters (int): Number of iterations for Bayesian optimization.

    Returns:
    float: The best performance score.
    """

    # Step 1: Data preprocessing
    features, labels = preprocess_data(data_file)

    # Step 2: Create parameter grid from provided dictionary
    param_grid = create_param_grid(param_dict)

    # Step 3: Perform Bayesian Optimization
    xp, yp = bayesian_optimisation(n_iters=n_iters,
                                   sample_loss=sample_loss,
                                   bounds=param_grid,
                                   x0=features,
                                   y0=labels,
                                   random_search=True,
                                   n_pre_samples=50,
                                   a=a,
                                   b=b)

    # Step 4: Save results to a CSV file
    result_df = pd.DataFrame({'Params': list(xp), 'Performance': yp})
    result_df.to_csv(f'result/{round(a, 2)}_{round(b, 2)}.csv', index=False)

    # Step 5: Return the best score and remaining iterations
    best_score = max(yp)
    remaining_iters = n_iters - len(yp)
    return best_score + remaining_iters

# 构造epochs搜索空间
def log_transfer(value):
    """log转换，需要原始数据都大于1
    公式：log10(x)/log10(max)
    :return 值域[0,1]
    """
    new_value = np.log10(value) / np.log10(64000)
    return new_value
# resnet
tmp_epochs = []
for i in range(10, 500, 5):
    tmp_epochs.append(log_transfer(i))
for i in range(500,1000,50):
    tmp_epochs.append(log_transfer(i))
for i in range(1000,65000,5000):
    tmp_epochs.append(log_transfer(i))


# You can test the code, with the following code
# # swin
# param = {'Architecture': [0.31696108]
#     , 'Normalization': [0.08]
#     , "Convolutions": [0]
#     , "Skip_Connections": [0.2]
#     , "Activation_Functions": [0]
#     , "Pooling_Operations": [0]
#     , "Regularization": [0.01003861]
#     , "Data_Augmentagion": [0.0,
#                             0.0000001332281718963160000,
#                             0.0000002664563437926330000,
#                             0.0000021316507503410600000,
#                             0.0000023981070941336900000,
#                             0.0000026645634379263300000,
#                             0.0000341064120054570000000,
#                             0.0000511596180081855000000,
#                             0.00218281036834925,
#                             0.00219133697135061,
#                             0.0043656207366985,
#                             0.00655695770804911,
#                             0.0130968622100954,
#                             0.0218281036834924,
#                             0.0240194406548431,
#                             0.0327506821282401,
#                             0.117905866302864,
#                             0.14406548431105,
#                             0.144065617539222,
#                             0.144099590723055,
#                             0.146248294679399,
#                             0.146793997271487,
#                             0.161527967257844,
#                             0.164802182810368,
#                             0.209549795361527,
#                             0.269892564802182,
#                             0.26989309771487,
#                             0.269893364171214,
#                             0.269893630627558,
#                             0.269893897083901,
#                             0.269894696452933,
#                             0.269894829681105,
#                             0.269894962909276,
#                             0.269895495821964,
#                             0.269896028734652,
#                             0.269896561647339,
#                             0.269899226210777,
#                             0.269909618008185,
#                             0.269913881309686,
#                             0.27862593792633,
#                             0.698499317871759,
#                             0.698500383697135,
#                             0.700716234652114,
#                             0.700827080491132,
#                             0.702864938608458,
#                             0.702865205064802,
#                             0.702867070259208,
#                             0.702899045020463,
#                             0.705047882204979,
#                             0.70519270122783,
#                             0.715961800818553,
#                             0.715964198925648,
#                             0.720327421555252,
#                             0.720329553206002,
#                             0.722510231923601,
#                             0.722518758526603,
#                             0.722783083219645,
#                             0.768349249658935,
#                             0.790177353342428,
#                             0.828692285555934,
#                             0.828692552012278,
#                             0.828694150750341,
#                             0.828694417206684,
#                             0.828694950119372,
#                             0.828696548857435,
#                             1.0
#                             ]
#     , "Feedforward_Networks": [0.446428571]
#     , "Learning_Rate_Schedules": [0.222222222222222]
#     , "Training_Algorithm": [0]
#     , "Output_Functions": [1]
#     , "train_size": [0.976263068688302]
#          , "number_label": [0.809359989090148]
#          , "cos": [3.5256721970926]
#     , "Input_Size": [0.802993877748649]
#     , "Framwork": [0.792481250360578]
#     , "paraeters": [0.559221991]
#     , "batch_size": [0.654120103]
#     , "learning_rate": [-2.35436764]
#     , "epochs": np.array(tmp_epochs)
#     , "Hardware": [0.333333333, 0.222222222, 0.185185185]
#     , "Number": [0.333333333]
#          }

# mobilenet
# param = {'Architecture': [0.616542512],
#                   'Normalization': [0.64]
#         , "Convolutions": [0.090163934]
#         , "Skip_Connections": [0]
#         , "Position_Embeddings": [0]
#         , "Activation_Functions": [0.026367188]
#     , "Pooling_Operations": [0.0,
#                              0.047619047619047616,
#                              0.19047619047619047,
#                              0.23809523809523808,
#                              0.38095238095238093,
#                              0.7619047619047619,
#                              0.8095238095238095,
#                              0.8571428571428571,
#                              0.9523809523809523,
#                              1.0]
#     , "Regularization": [0.003088803]
#     , "Data_Augmentagion": [0.000159008]
# , "Feedforward_Networks": [0.285714285714285]
#     , "Attention_Modules": [0]
#     , "Learning_Rate_Schedules": [0]
#     , "Training_Algorithm": [0.272727273]
#     , "Output_Functions": [0]
#     , "train_size": np.array(train_size)
#     , "number_label": [0.809359989090148]
#     , "cos": [3.5256721970926]
#     , "Input_Size": [0.846341827]
#     , "Framwork": [0.79248125]
#     , "paraeters": [0.210632217]
#     , "batch_size": np.array(batch_size)
#     , "learning_rate": [-0.76577573]
#     , "epochs": np.array(tmp_epochs)
#     , "Hardware": [0.222222222]
#     , "Number": [0.111111111]
#          }

# # Call the function with parameters
# best_result = search_best(a=0.5, b=3, data_file='归一化删除样例.csv', param_dict=create_param_grid(param))
# print(f"The best result from optimization is: {best_result}")
