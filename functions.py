
# --------------------------------------------------------------------------------------------------------- #
# -- project: Name of the Kaggle Competition                                                             -- #
# -- File: visualizations.py | Data visualization functions                                              -- #
# -- author: Name of the Team or the Author                                                              -- #
# -- license: GPL-3.0 License                                                                            -- #
# -- repository/notebook: Private repository URL and/or public notebook in kaggle                        -- #
# -- --------------------------------------------------------------------------------------------------- -- #

import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None
from multiprocessing import cpu_count
import multiprocessing as mp

# fijar semilla de aleatorios
np.random.seed(123)

# ------------------------------------------------------------------------------------- One-Hot Encoding -- # 
def variable_onehot(p_data):
    """
    Function to perform One-Hot encoding to transform a variable from categorical to numerical

    Parameters
    ----------
    p_data: pd.DataFrame
        DataFrame that contains the columns with categorical data to be transformed to numerical

    References
    ----------

    Returns
    -------

    """

    types = list(p_data.unique())
    types_dict = {types[i]:i for i in range(0, len(list(p_data.unique())))}
    encoded_result = p_data.map(types_dict).astype(float)

    return encoded_result



# ------------------------------------------------------------------------- One-Vs-Rest Based Data Split -- # 
def data_ovr(p_df, p_target):
    """
    p_df = data_train
    p_target = 'type'

    """  

    # count number of occurrences for each class
    labels = p_df[p_target].unique()
    # feature names
    x_cols = list(p_df.columns)
    x_cols.remove('type')
    # target name
    y_col = 'type'

    # dictionary to store dynamically splitted data using One-vs-Rest heuristic
    ovr_data = {'data_' + str(int(label)): {} for label in labels}

    # data separation
    for label in labels:
        # label = labels[0]
        df_data = p_df.copy()
        
        indx = list(p_df[p_df[p_target] == label].index)
        df_data['type'] = 0
        df_data['type'].iloc[indx] = 1
        x_data = np.array(df_data[x_cols])
        ovr_data['data_' + str(int(label))]['x_train'] = x_data
        
        y_data = np.array(df_data[p_target]).reshape(len(x_data), 1)
        ovr_data['data_' + str(int(label))]['y_train'] = y_data
    
    return ovr_data


# ---------------------------------------------------------------------------------- Logistic Regression -- #
def LogisticRegression(p_data_x, p_data_y, p_lambda, p_alpha, p_epochs, p_tol):

    def activation(x, w):
        
        z_i = np.dot(x, w.T)
        sigma = 1 / (1 + np.exp(-z_i))
        
        return sigma

    def cost(x, y, w, lmbd):
        
        m = len(y)
        j_w_1 = (1 - y)*np.log(1 - activation(x, w))
        j_w_2 = y*np.log(activation(x, w))
        j_w_3 = j_w_2 + j_w_1
        j_w = -1*np.average(j_w_3)
        j_reg = (lmbd/(2*m))*np.dot(w, w.T).squeeze()
        fin = j_w + j_reg
        
        return fin

    def gradient(x, y, w, lmbd):
        
        m = len(y)
        d_j_w_1 = 1/m
        d_j_w_2 = np.dot(x.T, (activation(x, w) - y))
        d_j_w = (d_j_w_1*d_j_w_2)
        d_j_w_reg = np.sum((lmbd*w)/m)
        fin = d_j_w + d_j_w_reg
        
        return fin

    def gradient_descent(x, y, w, alpha, lmbd, epochs):
        
        J = []      # loss function with training set
        J_v = []    # loss function with val set
        
        # big initial cost (since is a minimization problem)
        i_cost = float('inf')
        # epoch counter
        epoch = 0

        # random choice 20% of values for validation set
        split_size = int(x.shape[0]*0.20)
        s1 = np.random.choice(range(x.shape[0]), size=(split_size,), replace=False)
        s2 = list(set(range(x.shape[0])) - set(s1))
        x_v = x[s1]
        x = x[s2]
        y_v = y[s1]
        y = y[s2]

        # continue while cost is greater than tolerance
        while i_cost > p_tol:
            epoch += 1                            # register epoch
            w_grad = gradient(x, y, w, lmbd).T    # compute the gradinet
            w = w - alpha*w_grad                  # update weights
            i_cost = cost(x, y, w, lmbd)          # compute and store current cost (training set)
            v_cost = cost(x_v, y_v, w, lmbd)      # compute and store current cost (validation set)
            J.append(i_cost)                      # append cost to history
            J_v.append(v_cost)                    # append cost to history
            
            # end process if max epochs reached
            if p_epochs == epoch:
                break

        return w, J, J_v
    
    w = np.random.normal(loc=1e-3, scale=1e-6, size=(1, p_data_x.shape[1]))
    w, J, J_v = gradient_descent(x=p_data_x, y=p_data_y, w=w, alpha=p_alpha, lmbd=p_lambda, epochs=p_epochs)
    
    result_dict = {'weights': w, 'params': {'lambda': p_lambda, 'alpha': p_alpha},
                   'train': {'cost' :J[-1], 'history': J}, 'val':{'cost': J_v[-1], 'history': J_v}}

    return result_dict

# -- ------------------------------------------------------------------------- One-vs-Rest Model trainin -- #

def ovr_learning(p_data_ovr):
    """
    p_data_ovr=train_data_ovr

    """
    
    # -- parallel processing module - incomplete
    # workers = cpu_count() - 1
    # pool = mp.Pool(workers)

    models = list(p_data_ovr.keys())
    model_data = {'model_' + model[-1]: {} for model in models}

    # regularization component
    # lambda_s = [0.25, 0.5, 0.75, 0.9, 1.1, 1.25, 1.5]
    lambda_s = [1.1]
    
    # learning rate
    # alpha_s = [1e-1, 1.1e-1, 1.5e-1, 1e-2, 1e-3, 1e-4]
    alpha_s = [1e-2]
    
    # iterations
    epochs = 10000
    
    # tolerance for minimium cost value to early stopping
    tolerance = 1e-4
    
    # define metrics for model fit = 
    for model in models:
        # model = models[2]

        # set theoretical global cost to inf+
        global_cost = 1e10

        # perform gridsearch
        for lambda_i in lambda_s:
            # lambda_i = lambda_s[1]
            for alpha_i in alpha_s:
                # alpha_i = alpha_s[1]

                """
                pool.starmap(LogisticRegression, [(p_data_x=p_data_ovr[model]['x_train'].copy(),
                                                   p_data_y=p_data_ovr[model]['y_train'].copy(),
                                                   p_lambda=lambda_i[0]) for exp in exec_exp])
                """

                # get model performance
                model_learning = LogisticRegression(p_data_x=p_data_ovr[model]['x_train'].copy(),
                                                    p_data_y=p_data_ovr[model]['y_train'].copy(),
                                                    p_lambda=lambda_i, p_alpha=alpha_i, p_epochs=epochs,
                                                    p_tol=tolerance)

                # save local cost (inv-weighted-mean)
                local_cost = abs(model_learning['train']['cost']*0.2) + abs(model_learning['val']['cost']*0.8)
                
                # update local cost metric and models if improvement is detected
                if local_cost < global_cost:
                    print('mejora encontrada para: ', model)
                    global_cost = local_cost
                    model_learning['fitted_cost'] = global_cost
                    fittest_model = model_learning

        # save fittest individual for model
        model_data['model_' + model[-1]] = fittest_model

    return model_data

# -- ------------------------------------------------------------------------- One-vs-Rest Model trainin -- #

def ovr_predict(p_data_ovr, p_models_ovr, p_vote_w):
    """
    p_data_ovr=test_data_ovr
    p_models_ovr=models_ovr
    p_vote_w=[1,1,1]
    """

    def activation(x, w):
        z_i = np.dot(x, w.T)
        sigma = 1 / (1 + np.exp(-z_i))
        return sigma
    
    def predict(x_data, w):
        p = activation(x=x_data, w=w)
        return p.astype(float)

    predictions = {model: 0 for model in list(p_models_ovr.keys())}

    for model in list(p_models_ovr.keys()):  
        # model = list(p_models_ovr.keys())[0]
        predictions[model] = predict(x_data=p_data_ovr, w=p_models_ovr[model]['weights'])
    
    # voting
    voting = pd.DataFrame(np.concatenate((
                          predictions['model_0']*p_vote_w[0],
                          predictions['model_1']*p_vote_w[1],
                          predictions['model_2']*p_vote_w[2]), axis=1))

    voting['decision'] = voting.idxmax(axis=1)

    return voting
