import numpy as np
class SyntheticDataset:

    def __init__(self, p_steps=10, num_covariates=100, gamma=0.1):
        self.p = p_steps
        self.num_covariates = num_covariates
        self.gamma = gamma

    def generate_xt_zt(self, previous_xt, previous_zt, previous_treatment):
        xt = [0] * self.num_covariates
        zt = [0] * self.num_covariates

        for j in range(self.num_covariates):
            alpha = np.random.normal(1 - (j / self.p), (1 / self.p) ** 2)
            beta = np.random.normal(0, 0.02 ** 2)
            mu = np.random.normal(1 - (j / self.p), (1 / self.p) ** 2)
            v = np.random.normal(0, 0.02 ** 2)

            xt[j] = (1 / self.p) * sum(
                [alpha * previous_xt[t - j] + beta * previous_treatment[t] for t in range(self.p)]) + np.random.normal(
                0, 0.01 ** 2)
            zt[j] = (1 / self.p) * sum(
                [mu * previous_zt[t - j] + v * previous_treatment[t] for t in range(self.p)]) + np.random.normal(0,
                                                                                                                 0.01 ** 2)

        return xt, zt

    def generate_st_yt(self, current_xt, static_features_c, current_zt):
        st = self.gamma * (1 / len(current_zt)) * sum(current_zt) + (1 - self.gamma) * self.mapping_function(current_xt,
                                                                                                             static_features_c)
        wt = np.random.uniform(-1, 1)
        b = np.random.normal(0, 0.1)
        yt = wt * st + b
        return st, yt

    def mapping_function(self, xt, c):
        return np.concatenate((xt, c))

    def generate_samples(self, num_treated_samples=1000, num_control_samples=3000):
        treated_samples = []
        for _ in range(num_treated_samples):
            treatment_start_point = np.random.choice(self.p)
            previous_xt = [0] * self.p
            previous_zt = [0] * self.p
            previous_treatment = [1 if t >= treatment_start_point else 0 for t in range(self.p)]

            xt, zt = self.generate_xt_zt(previous_xt, previous_zt, previous_treatment)
            st, yt = self.generate_st_yt(xt, [], zt)

            treated_samples.append((xt, zt, st, yt))

        control_samples = []
        for _ in range(num_control_samples):
            previous_xt = [0] * self.p
            previous_zt = [0] * self.p
            previous_treatment = [0] * self.p

            xt, zt = self.generate_xt_zt(previous_xt, previous_zt, previous_treatment)
            st, yt = self.generate_st_yt(xt, [], zt)

            control_samples.append((xt, zt, st, yt))

        return treated_samples, control_samples
