
# -- EXPERIMENTS -- #

#%%
# -- [v22]
# scalling = standarizacion
# dummies = color
# bias = si
# epochs = 10000
# sample split = 0.20
# tolerance = 1e-4
# hyper_m0 = {'lambda': 1.1, 'alpha': 0.01}
# hyper_m1 = {'lambda': 1.1, 'alpha': 0.01}
# hyper_m2 = {'lambda': 1.1, 'alpha': 0.01}
# funcion para eleccion de hyper = train*0.8 + val*0.20

# peso para voto = [1, 1, 1]
# 0    223
# 2    193
# 1    113
# acc_test = 0.72778
# archivo = submission_v22.csv

#%%
# -- [v221]
# scalling = standarizacion
# dummies = color
# bias = si
# epochs = 10000
# sample split = 0.20
# tolerance = 1e-4
# hyper_m0 = {'lambda': 1.1, 'alpha': 0.01}
# hyper_m1 = {'lambda': 1.1, 'alpha': 0.01}
# hyper_m2 = {'lambda': 1.1, 'alpha': 0.01}
# funcion para eleccion de hyper = train*0.8 + val*0.20

# peso para voto = [0.3477, 0.3369, 0.3154]
# 0    225
# 2    188
# 1    116
# acc_test = 0.72778
# archivo = submission_v221.csv

#%%
# -- [v222]
# 221 - quitar bias

# scalling = standarizacion
# dummies = color
# bias = no
# epochs = 10000
# sample split = 0.20
# tolerance = 1e-4
# hyper_m0 = {'lambda': 1.1, 'alpha': 0.01}
# hyper_m1 = {'lambda': 1.1, 'alpha': 0.01}
# hyper_m2 = {'lambda': 1.1, 'alpha': 0.01}
# funcion para eleccion de hyper = train*0.8 + val*0.20

# peso para voto = [0.3477, 0.3369, 0.3154]
# 0    218
# 2    203
# 1    108
# acc_test = .72400
# archivo = submission_v222.csv

#%%
# -- [v223]
# 221 - inv-weighted hypercost

# scalling = standarizacion
# dummies = color
# bias = si
# epochs = 10000
# sample split = 0.20
# tolerance = 1e-4
# hyper_m0 = {'lambda': 1.1, 'alpha': 0.01}
# hyper_m1 = {'lambda': 1.1, 'alpha': 0.01}
# hyper_m2 = {'lambda': 1.1, 'alpha': 0.01}
# funcion para eleccion de hyper = train*0.2 + val*0.8

# peso para voto = [0.3477, 0.3369, 0.3154]
# 0    219
# 2    199
# 1    111
# acc_test = 0.7710
# archivo = submission_v223.csv

#%%
# -- [v224]
# 221 - weighted hypercost

# scalling = standarizacion
# dummies = color
# bias = si
# epochs = 10000
# sample split = 0.10
# tolerance = 1e-4
# hyper_m0 = {'lambda': 1.1, 'alpha': 0.01}
# hyper_m1 = {'lambda': 1.1, 'alpha': 0.01}
# hyper_m2 = {'lambda': 1.1, 'alpha': 0.01}
# funcion para eleccion de hyper = train*0.9 + val*0.1

# peso para voto = [0.3477, 0.3369, 0.3154]
# 0    229
# 2    190
# 1    110
# acc_test = .72400
# archivo = submission_v224.csv

#%%
# -- [v23]
# 223 - modified hyperparams search space
# lambda_s = [0.25, 0.5, 0.75, 0.9, 1.1, 1.25, 1.5]
# alpha_s = [1e-1, 1.1e-1, 1.5e-1, 1e-2, 1e-3, 1e-4]

# scalling = standarizacion
# dummies = color
# bias = si
# epochs = 8000
# sample split = 0.20
# tolerance = 1e-4
# hyper_m0 = {'lambda': 1.1, 'alpha': 0.01}
# hyper_m1 = {'lambda': 1.1, 'alpha': 0.01}
# hyper_m2 = {'lambda': 1.1, 'alpha': 0.01}
# funcion para eleccion de hyper = train*0.2 + val*0.8

# peso para voto = [0.3477, 0.3369, 0.3154]
# 0    226
# 2    185
# 1    118
# acc_test = 0.74102
# archivo = submission_v23.csv

#%%
# -- [v3]
# 223 - modified hyperparams search space - no split
# lambda_s = [0.25, 0.5, 0.75, 0.9, 1.1, 1.25, 1.5]
# alpha_s = [1e-1, 1.1e-1, 1.5e-1, 1e-2, 1e-3, 1e-4]

# scalling = standarizacion
# dummies = color
# bias = si
# epochs = 8000
# sample split = 0
# tolerance = 1e-4
# hyper_m0 = {'lambda': 1.1, 'alpha': 0.01}
# hyper_m1 = {'lambda': 1.1, 'alpha': 0.01}
# hyper_m2 = {'lambda': 1.1, 'alpha': 0.01}
# funcion para eleccion de hyper = train

# peso para voto = [0.3477, 0.3369, 0.3154]

# acc_test = 0.72589
# archivo = submission_v3.csv

#%%
# -- [vfinal]
# inv-weighted hypercost

# scalling = standarizacion
# dummies = color
# bias = si
# epochs = 10000
# sample split = 0.20
# tolerance = 1e-4
# hyper_m0 = {'lambda': 1.1, 'alpha': 0.01}
# hyper_m1 = {'lambda': 1.1, 'alpha': 0.01}
# hyper_m2 = {'lambda': 1.1, 'alpha': 0.01}
# funcion para eleccion de hyper = train*0.2 + val*0.8

# peso para voto = [0.3477, 0.3369, 0.3154]
# 0    219
# 2    199
# 1    111
# acc_test = 0.7710
# archivo = submission_vfinal.csv