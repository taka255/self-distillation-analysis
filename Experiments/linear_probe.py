# linear probe experiment (optimize hyperparameters in 0-SD and 1-SD)
# Before running this code, please run extract_feature_vec.py.


import torch
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from skopt import gp_minimize
from skopt.space import Real
from scipy.special import expit
import warnings


warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=FutureWarning)




def load_and_preprocess_data(model_name, label_name, M=None, theta=0.0, seed=42):
    train_feats = torch.load(f"./data/cifar10_all_feats_train_{model_name}.pt", map_location="cpu")  
    train_labels = torch.load(f"./data/cifar10_all_labels_train_{model_name}.pt", map_location="cpu")
    test_feats = torch.load(f"./data/cifar10_all_feats_test_{model_name}.pt", map_location="cpu")
    test_labels = torch.load(f"./data/cifar10_all_labels_test_{model_name}.pt", map_location="cpu")
    N = train_feats.shape[1]
    
    # label filter
    mask_train = (train_labels == label_name[0]) | (train_labels == label_name[1])  
    mask_test = (test_labels == label_name[0]) | (test_labels == label_name[1])

    train_feats = train_feats[mask_train]
    train_labels = train_labels[mask_train]
    test_feats = test_feats[mask_test]
    test_labels = test_labels[mask_test]

    # label conversion
    train_labels = (train_labels == label_name[1]).long()
    test_labels = (test_labels == label_name[1]).long()

    # convert to NumPy array
    X_train_full = train_feats.numpy()
    y_train_full = train_labels.numpy()
    X_test = test_feats.numpy()
    y_test = test_labels.numpy()
    
    # if M is specified, sampling
    if M is not None:
        np.random.seed(seed)
        # get M/2 samples from each class
        M_per_class = M // 2
        indices_class0 = np.random.choice(np.where(y_train_full == 0)[0], M_per_class, replace=False)
        indices_class1 = np.random.choice(np.where(y_train_full == 1)[0], M_per_class, replace=False)
        
        # flip only the target samples
        n_flip_per_class = int(M_per_class * theta)
        flip0 = np.random.choice(indices_class0, n_flip_per_class, replace=False)
        flip1 = np.random.choice(indices_class1, n_flip_per_class, replace=False)
        y_train_full[flip0] = 1
        y_train_full[flip1] = 0

        # shuffle after that
        indices = np.concatenate([indices_class0, indices_class1])
        np.random.shuffle(indices)
        X_train = X_train_full[indices]
        y_train = y_train_full[indices]
            
    else:
        X_train = X_train_full
        y_train = y_train_full
    
    return X_train, y_train, X_test, y_test, N



def optimize_linear_probe(X_train, y_train, X_test, y_test, n_calls=40, n_random_starts=10, seed=42):

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # objective function
    def objective(params):
        lambda_val = 10 ** params[0]
        C = 1.0 / lambda_val
        clf = LogisticRegression(
            C=C,
            max_iter=10000,
            random_state=seed,
            penalty='l2',
            fit_intercept=True,
            solver='lbfgs'  
        )
        clf.fit(X_train_scaled, y_train)
        y_pred = clf.predict(X_test_scaled)
        return -accuracy_score(y_test, y_pred)
    
    # define search space
    space = [Real(-9, 6, name='log_lambda')]
    
    # execute optimization
    result = gp_minimize(
        objective,
        space,
        n_calls=n_calls,
        n_random_starts=n_random_starts,
        noise=0.1,
        random_state=seed
    )
    
    # print optimal parameters
    optimal_lambda = 10 ** result.x[0]
    print(f"optimal_lambda: {optimal_lambda}")
    
    best_accuracy = -result.fun
    test_error = 1 - best_accuracy
    
    return test_error, optimal_lambda



def optimize_teacher_student(X_train, y_train, X_test, y_test, n_calls=80, n_random_starts=20, seed=42):

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    def objective(params):
        log_l1, log_l2, beta = params
        # λ1, λ2
        lambda1 = 10 ** log_l1
        lambda2 = 10 ** log_l2

        # === first stage: teacher learning ===
        C1 = 1.0 / lambda1
        C2 = 1.0 / lambda2
        teacher = LogisticRegression(
            C=C1,
            max_iter=10000,
            random_state=seed,
            penalty='l2',
            fit_intercept=True,
            solver='lbfgs'  
        )
        teacher.fit(X_train_scaled, y_train)
        z = teacher.decision_function(X_train_scaled)
        p_soft = expit(beta * z)

        # === second stage: student learning ===
        X_dup = np.vstack([X_train_scaled, X_train_scaled])
        y_dup = np.concatenate([np.ones_like(p_soft), np.zeros_like(p_soft)])
        w_dup = np.concatenate([p_soft, 1.0 - p_soft])
        student = LogisticRegression(
            C=C2,
            max_iter=10000,
            random_state=seed,
            penalty='l2',
            fit_intercept=True,
            solver='lbfgs'  
        )
        student.fit(X_dup, y_dup, sample_weight=w_dup)
        p_test = student.predict(X_test_scaled)
        y_pred = (p_test >= 0.5).astype(int)

        return -accuracy_score(y_test, y_pred)

    # search space (log10 scale: -9 ~ 6)
    space = [
        Real(-9, 6, name='log_l1'),
        Real(-9, 6, name='log_l2'),
        Real( 0.1, 30.0, name='beta'),
    ]

    # execute Bayesian optimization
    result = gp_minimize(
        objective,
        space,
        n_calls=n_calls,
        n_random_starts=n_random_starts,
        noise=0.1,
        random_state=seed
    )
    
    # print optimal parameters
    optimal_lambda0 = 10 ** result.x[0]
    optimal_lambda1 = 10 ** result.x[1]
    optimal_beta    = result.x[2]
    
    print("--------------------------------")
    print("Teacher-Student")
    print(f"Optimal parameters: log_l1={result.x[0]:.4f}, log_l2={result.x[1]:.4f}, beta={result.x[2]:.4f}")
    
    best_acc  = -result.fun
    test_err  = 1 - best_acc
    return test_err, optimal_lambda0, optimal_lambda1, optimal_beta


def optimize_teacher_student_hardlabel(
    X_train, y_train, X_test, y_test,
    n_calls=40, n_random_starts=10,
    seed=42
):
    # standardize data
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_train)
    X_te_s = scaler.transform(X_test)

    # objective function (β fixed)
    def objective(log_params):
        log_l1, log_l2 = log_params
        λ1 = 10 ** log_l1
        λ2 = 10 ** log_l2

        # teacher
        C1 = 1.0 / λ1
        teacher = LogisticRegression(
            C=C1, max_iter=10000, solver='lbfgs',
            penalty='l2', fit_intercept=True,
            random_state=seed
        )
        teacher.fit(X_tr_s, y_train)
        z = teacher.decision_function(X_tr_s)
        p_hard = teacher.predict(X_tr_s)

        # student
        X_dup = np.vstack([X_tr_s, X_tr_s])
        y_dup = np.concatenate([np.ones_like(p_hard), np.zeros_like(p_hard)])
        w_dup = np.concatenate([p_hard, 1.0 - p_hard])

        C2 = 1.0 / λ2
        student = LogisticRegression(
            C=C2, max_iter=10000, solver='lbfgs',
            penalty='l2', fit_intercept=True,
            random_state=seed
        )
        student.fit(X_dup, y_dup, sample_weight=w_dup)
        y_pred = student.predict(X_te_s)
        return -accuracy_score(y_test, y_pred)

    # search space: log₁₀λ₁, log₁₀λ₂
    space = [
        Real(-9, 6, name='log_l1'),
        Real(-9, 6, name='log_l2'),
    ]

    res = gp_minimize(
        objective, space,
        n_calls=n_calls, n_random_starts=n_random_starts,
        noise=0.1, random_state=seed
    )

    # optimal parameters
    optimal_lambda1 = 10 ** res.x[0]
    optimal_lambda2 = 10 ** res.x[1]
    best_acc = -res.fun
    test_err = 1 - best_acc

    print("--------------------------------")
    print("Hardlabel")
    print(f"optimal log_l1={res.x[0]:.4f}, log_l2={res.x[1]:.4f}")
    print(f"λ₁={optimal_lambda1:.3e}, λ₂={optimal_lambda2:.3e}, Test error={test_err:.3f}")

    return test_err, optimal_lambda1, optimal_lambda2


########################################################
#! Problem Parameters
########################################################
model_name = "resnet18" # resnet18, resnet50
label_name = [3, 5] # 3=dog, 5=cat
M_values = [100, 200, 500, 1000, 2000, 5000, 10000]


theta = 0.0  # probability of label noise
num_trials = 10
seed_start = 54321
alphas          = np.empty((num_trials, len(M_values)), dtype=float)
test_errors_0SD = np.empty((num_trials, len(M_values)), dtype=float)
test_errors_1SD = np.empty((num_trials, len(M_values)), dtype=float)
test_errors_hard = np.empty((num_trials, len(M_values)), dtype=float)
SD0_optimal_lambda0_list = np.empty((num_trials, len(M_values)), dtype=float)
SD1_optimal_lambda0_list = np.empty((num_trials, len(M_values)), dtype=float)
SD1_optimal_lambda1_list = np.empty((num_trials, len(M_values)), dtype=float)
SD1_optimal_beta_list = np.empty((num_trials, len(M_values)), dtype=float)
SD1_optimal_lambda1_hard_list = np.empty((num_trials, len(M_values)), dtype=float)
SD1_optimal_lambda2_hard_list = np.empty((num_trials, len(M_values)), dtype=float)





def process(trial, j, M):
    seed = seed_start + trial
    X_train, y_train, X_test, y_test, N = load_and_preprocess_data(
        model_name, label_name, M, theta, seed=seed
    )
    alpha = M / N

    err0, SD0_optimal_lambda0 = optimize_linear_probe(X_train, y_train, X_test, y_test, seed=seed)
    err1, SD1_optimal_lambda0, SD1_optimal_lambda1, SD1_optimal_beta = optimize_teacher_student(X_train, y_train, X_test, y_test, seed=seed)
    errhard, SD1_optimal_lambda1_hard, SD1_optimal_lambda2_hard = optimize_teacher_student_hardlabel(X_train, y_train, X_test, y_test, seed=seed)
    
    print(f"trial: {trial}, M: {M}, alpha: {alpha}, err0: {err0}, err1: {err1}, SD0_optimal_lambda0: {SD0_optimal_lambda0}, SD1_optimal_lambda0: {SD1_optimal_lambda0}, SD1_optimal_lambda1: {SD1_optimal_lambda1}, SD1_optimal_beta: {SD1_optimal_beta}")
    
    return trial, j, alpha, err0, err1, errhard, SD0_optimal_lambda0, SD1_optimal_lambda0, SD1_optimal_lambda1, SD1_optimal_beta, SD1_optimal_lambda1_hard, SD1_optimal_lambda2_hard




results = []
for trial in range(num_trials):
    for j, M in enumerate(M_values):
        results.append(process(trial, j, M))

for trial, j, alpha, err0, err1, errhard, SD0_optimal_lambda0, SD1_optimal_lambda0, SD1_optimal_lambda1, SD1_optimal_beta, SD1_optimal_lambda1_hard, SD1_optimal_lambda2_hard in results:
    alphas[trial, j]          = alpha
    test_errors_0SD[trial, j] = err0
    test_errors_1SD[trial, j] = err1
    test_errors_hard[trial, j] = errhard
    SD0_optimal_lambda0_list[trial, j] = SD0_optimal_lambda0
    SD1_optimal_lambda0_list[trial, j] = SD1_optimal_lambda0
    SD1_optimal_lambda1_list[trial, j] = SD1_optimal_lambda1
    SD1_optimal_beta_list[trial, j] = SD1_optimal_beta
    SD1_optimal_lambda1_hard_list[trial, j] = SD1_optimal_lambda1_hard
    SD1_optimal_lambda2_hard_list[trial, j] = SD1_optimal_lambda2_hard



# mean and standard deviation
mean_err0 = test_errors_0SD.mean(axis=0)
std_err0  = test_errors_0SD.std(axis=0)
mean_err1 = test_errors_1SD.mean(axis=0)
std_err1  = test_errors_1SD.std(axis=0)
mean_errhard = test_errors_hard.mean(axis=0)
std_errhard  = test_errors_hard.std(axis=0) 

# standard error (SEM)
sem_err0 = std_err0 / np.sqrt(num_trials)
sem_err1 = std_err1 / np.sqrt(num_trials)
sem_errhard = std_errhard / np.sqrt(num_trials)

# α (M/N) is the same for all trials, so use the first row for the x-axis
x = alphas[0]

# プロット
plt.figure(figsize=(8,6))
plt.errorbar(x, mean_err0, yerr=sem_err0, fmt='o-', capsize=4, label='0SD (SEM)')
plt.errorbar(x, mean_err1, yerr=sem_err1, fmt='s-', capsize=4, label='1SD (SEM)')
plt.errorbar(x, mean_errhard, yerr=sem_errhard, fmt='^-', capsize=4, label='hard (SEM)')
plt.xscale('log')
plt.xlabel(r'$\alpha = M/N$')
plt.ylabel('Test Error')
plt.title(f'{model_name} (θ={theta}) — Mean ± SEM over {num_trials} trials')
plt.grid(True, which='both', ls='--', lw=0.5)
plt.legend()
plt.tight_layout()
plt.savefig('test_error_sem_plot.png', dpi=300)
plt.show()