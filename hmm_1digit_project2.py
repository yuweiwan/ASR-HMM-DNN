#!/usr/bin/env python3

# Copyright 2018 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import argparse
import logging
import numpy as np

# neural network related
from sklearn.neural_network import MLPClassifier


def get_data_dict(data):
    data_dict = {}
    for line in data:
        if "[" in line:
            key = line.split()[0]
            mat = []
        elif "]" in line:
            line = line.split(']')[0]
            mat.append([float(x) for x in line.split()])
            data_dict[key] = np.array(mat)
        else:
            mat.append([float(x) for x in line.split()])
    return data_dict


# you can add more functions here if needed
def log_gaussian(o, mu, r):
    compute = (- 0.5 * np.log(r) - np.divide(
        np.square(o - mu), 2 * r) - 0.5 * np.log(2 * np.pi)).sum()
    return compute


def lse(x):
    # log sum exp
    m = np.max(x)
    x -= m
    return m + np.log(np.sum(np.exp(x)))


def good_log(x):
    x_log = np.log(x, where=(x != 0))
    x_log[np.where(x == 0)] = -10000000
    return x_log


def forward(pi, a, o, mu, r):
    # pi is initial probability over states, a is transition matrix
    T = o.shape[0]
    J = mu.shape[0]
    log_alpha = np.zeros((T, J))

    for j in range(J):
        log_alpha[0][j] = np.log(pi)[j] + log_gaussian(o[0], mu[j], r[j])

    for t in range(1, T):
        for j in range(J):
            log_alpha[t, j] = log_gaussian(o[t], mu[j], r[j]) + lse(good_log(a[:, j].T) + log_alpha[t - 1])

    return log_alpha


def com_log_prior(s):
    states, counts = np.unique(s, return_counts=True)
    p = np.zeros(len(states))
    for sts, cts in zip(states, counts):
        p[sts] = cts
    p_dis = p / np.sum(p)
    return good_log(p_dis)


def context_expand(data):
    T = data.shape[0]
    data_1 = np.copy(data[0])
    data_T = np.copy(data[-1])
    for i in range(3):
        data = np.insert(data, 0, data_1, axis=0)
        data = np.insert(data, -1, data_T, axis=0)
    expand_data = np.zeros((T, 7 * data.shape[1]))
    for t in range(3, T + 3):
        np.concatenate((data[t - 3], data[t - 2], data[t - 1], data[t], data[t + 1], data[t + 2], data[t + 3]),
                       out=expand_data[t - 3])
    return expand_data


class SingleGauss():
    def __init__(self):
        # Basic class variable initialized, feel free to add more (Use from project1)
        self.dim = None
        self.mu = None
        self.r = None

    def train(self, data):
        data = np.vstack(data)
        self.mu = np.mean(data, axis=0)
        self.r = np.mean(np.square(np.subtract(data, self.mu)), axis=0)

    def loglike(self, data_mat):
        ll = 0
        for each_line in data_mat:
            ll += log_gaussian(each_line, self.mu, self.r)
        return ll


class HMM():
    def __init__(self, sg_model, nstate):
        # Basic class variable initialized, feel free to add more
        self.pi = np.full(nstate, 1 / nstate)
        self.mu = np.tile(sg_model.mu, (nstate, 1))
        self.r = np.tile(sg_model.r, (nstate, 1))
        self.nstate = nstate

    def initStates(self, data):
        # states: s elements, each T length
        self.states = []
        for data_s in data:
            T = data_s.shape[0]
            state_seq = np.array([self.nstate * t / T for t in range(T)], dtype=int)
            self.states.append(state_seq)

    def get_state_seq(self, data):
        T = data.shape[0]
        J = self.nstate
        s_hat = np.zeros(T, dtype=int)
        log_a = good_log(self.a)
        log_delta = np.zeros((T, J))
        log_delta[0] = np.log(self.pi)
        psi = np.zeros((T, J))

        # initialize
        for j in range(J):
            log_delta[0, j] += log_gaussian(data[0], self.mu[j], self.r[j])

        for t in range(1, T):
            for j in range(J):
                temp = np.zeros(J)
                for i in range(J):
                    temp[i] = log_delta[t - 1, i] + log_a[i, j] + log_gaussian(data[t], self.mu[j], self.r[j])
                log_delta[t, j] = np.max(temp)
                psi[t, j] = np.argmax(log_delta[t - 1] + log_a[:, j])

        s_hat[T - 1] = np.argmax(log_delta[T - 1])

        for t in reversed(range(T - 1)):
            s_hat[t] = psi[t + 1, s_hat[t + 1]]

        return s_hat

    def viterbi(self, data):
        for u, data_u in enumerate(data):
            s_hat = self.get_state_seq(data_u)
            self.states[u] = s_hat

    def m_step(self, data):
        self.a = np.zeros((self.nstate, self.nstate))
        gamma_0 = np.zeros(self.nstate)
        gamma_1 = np.zeros((self.nstate, data[0].shape[1]))
        gamma_2 = np.zeros((self.nstate, data[0].shape[1]))

        for s in range(len(data)):
            T = data[s].shape[0]

            # state_seq is a list of states with length t
            state_seq = self.states[s]
            # gamma is emission_matrix
            gamma = np.zeros((T, self.nstate))

            # calculate frequency for a and gamma according to current states
            for t, j in enumerate(state_seq[:-1]):
                self.a[j, state_seq[t + 1]] += 1
            for t, j in enumerate(state_seq):
                gamma[t, j] = 1

            # gamma^0_j = \sum^T_{t=1} gamma_t(j)
            gamma_0 += np.sum(gamma, axis=0)
            # gamma^1_j = \sum^T_{t=1}gamma_t(j)o_t
            # gamma^2_j = \sum^\sum^T_{t=1}gamma_t(j)o_t**2
            for t, j in enumerate(state_seq):
                gamma_1[j] += data[s][t]
                gamma_2[j] += np.square(data[s][t])

        for j in range(self.nstate):
            self.a[j] /= np.sum(self.a[j])
            self.mu[j] = gamma_1[j] / gamma_0[j]
            self.r[j] = (gamma_2[j] - np.multiply(gamma_0[j], self.mu[j] ** 2)) / gamma_0[j]

    def train(self, data, iter):
        if iter == 0:
            self.initStates(data)
        self.m_step(data)
        # renew states
        self.viterbi(data)

    def loglike(self, data):
        log_alpha_t = forward(self.pi, self.a, data, self.mu, self.r)[-1]
        ll = lse(log_alpha_t)

        return ll


class HMMMLP():
    def __init__(self, mlp, hmm_model, S, uniq_state_dict):
        self.mlp = mlp
        self.hmm = hmm_model
        self.log_prior = com_log_prior(S)
        self.uniq_state_dict = uniq_state_dict

    def forward_dnn(self, data, digit):
        T = data.shape[0]
        J = self.hmm.nstate
        o_expand = context_expand(data)
        mlp_ll = self.mlp.predict_log_proba(o_expand)
        log_alpha = np.zeros((T, J))
        log_alpha[0] = good_log(self.hmm.pi)
        for j in range(J):
            log_alpha[0] += np.array(mlp_ll[0][self.uniq_state_dict[(digit, j)]] + self.log_prior[
                self.uniq_state_dict[(digit, j)]])
        for t in range(1, T):
            for j in range(J):
                tmp = mlp_ll[t][self.uniq_state_dict[(digit, j)]] + self.log_prior[self.uniq_state_dict[(digit, j)]]
                log_alpha[t, j] = tmp + lse(good_log(self.hmm.a[:, j].T) + log_alpha[t - 1])

        return log_alpha

    def loglike(self, data, digit):
        log_alpha_t = self.forward_dnn(data, digit)[-1]
        ll = lse(log_alpha_t)

        return ll


def sg_train(digits, train_data):
    model = {}
    for digit in digits:
        model[digit] = SingleGauss()

    for digit in digits:
        data = [train_data[id] for id in train_data.keys() if digit in id.split('_')[1]]
        logging.info("process %d data for digit %s", len(data), digit)
        model[digit].train(data)

    return model


def hmm_train(digits, train_data, sg_model, nstate, niter):
    logging.info("hidden Markov model training, %d states, %d iterations", nstate, niter)

    hmm_model = {}
    for digit in digits:
        hmm_model[digit] = HMM(sg_model[digit], nstate=nstate)

    i = 0
    while i < niter:
        logging.info("iteration: %d", i)
        total_log_like = 0.0
        for digit in digits:
            data = [train_data[id] for id in train_data.keys() if digit in id.split('_')[1]]
            logging.info("process %d data for digit %s", len(data), digit)

            hmm_model[digit].train(data, i)

            for data_u in data:
                total_log_like += hmm_model[digit].loglike(data_u)

        logging.info("log likelihood: %f", total_log_like)
        i += 1

    return hmm_model


def mlp_train(digits, train_data, hmm_model, uniq_state_dict, nunits=(256, 256), bsize=128, nepoch=10, lr=0.01):
    # Complete the function to train MLP and create HMMMLP object for each digit
    # Get unique output IDs for MLP, perform alignment to get labels and perform context expansion
    data_dict = {}
    seq_dict = {}
    for digit in digits:
        uniq = lambda t: uniq_state_dict[(digit, t)]
        vfunc = np.vectorize(uniq)

        sequences = []
        data = [train_data[id] for id in train_data.keys() if digit in id.split('_')[1]]
        data_dict[digit] = data

        for data_u in data:
            seq = hmm_model[digit].get_state_seq(data_u)
            sequences.append(vfunc(seq))

        seq_dict[digit] = sequences
    # A simple scikit-learn MLPClassifier call is given below, check other arguments and play with it
    # OPTIONAL: Try pytorch instead of scikit-learn MLPClassifier
    O = []
    S = []
    for digit in digits:
        data = data_dict[digit]
        sequences = seq_dict[digit]
        for data_u, seq in zip(data, sequences):
            data_u_expanded = context_expand(data_u)
            O.append(data_u_expanded)
            S.append(seq)
    O = np.vstack(O)
    S = np.concatenate(S, axis=0)

    mlp = MLPClassifier(hidden_layer_sizes=nunits, random_state=1, early_stopping=True, verbose=True,
                        validation_fraction=0.1)
    mlp.fit(O, S)

    mlp_model = {}
    for digit in digits:
        # variables to initialize HMMMLP are incomplete below, pass additional variables that are required
        mlp_model[digit] = HMMMLP(mlp, hmm_model[digit], S, uniq_state_dict)

    return mlp_model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('train', type=str, help='training data')
    parser.add_argument('test', type=str, help='test data')
    parser.add_argument('--niter', type=int, default=10)
    parser.add_argument('--nstate', type=int, default=5)
    parser.add_argument('--nepoch', type=int, default=10)
    parser.add_argument('--lr', type=int, default=0.01)
    parser.add_argument('--mode', type=str, default='mlp',
                        choices=['hmm', 'mlp'],
                        help='Type of models')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    # set seed
    np.random.seed(777)

    # logging info
    log_format = "%(asctime)s (%(module)s:%(lineno)d) %(levelname)s:%(message)s"
    logging.basicConfig(level=logging.INFO, format=log_format)

    digits = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "z", "o"]
    uniq_state_dict = {}
    i = 0
    for digit in digits:
        for state in range(args.nstate):
            uniq_state_dict[(digit, state)] = i
            i += 1

    # read training data
    with open(args.train) as f:
        train_data = get_data_dict(f.readlines())
    # for debug
    if args.debug:
        train_data = {key: train_data[key] for key in list(train_data.keys())[:200]}

    # read test data
    with open(args.test) as f:
        test_data = get_data_dict(f.readlines())
    # for debug
    if args.debug:
        test_data = {key: test_data[key] for key in list(test_data.keys())[:200]}

    # Single Gaussian
    sg_model = sg_train(digits, train_data)

    if args.mode == 'hmm':
        model = hmm_train(digits, train_data, sg_model, args.nstate, args.niter)
    elif args.mode == 'mlp':
        hmm_model = hmm_train(digits, train_data, sg_model, args.nstate, args.niter)
        # Modify MLP training function call with appropriate arguments here
        model = mlp_train(digits, train_data, hmm_model, uniq_state_dict, nepoch=args.nepoch, lr=args.lr, nunits=(256, 256))

    # test
    total_count = 0
    correct = 0
    for key in test_data.keys():
        lls = []
        for digit in digits:
            ll = model[digit].loglike(test_data[key], digit)
            lls.append(ll)
        predict = digits[np.argmax(np.array(lls))]
        log_like = np.max(np.array(lls))

        logging.info("predict %s for utt %s (log like = %f)", predict, key, log_like)
        if predict in key.split('_')[1]:
            correct += 1
        total_count += 1

    logging.info("accuracy: %f", float(correct / total_count * 100))
