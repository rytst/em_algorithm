import numpy as np
import matplotlib.pyplot as plt


def multivariate_normal(x, mu, cov):
    det = np.linalg.det(cov)
    inv = np.linalg.inv(cov)

    d = len(x)

    z = 1 / np.sqrt((2 * np.pi)**d * det)

    y = z * np.exp((x - mu).T @ inv @ (x - mu) / -2.0)

    return y


def gmm(x, phis, mus, covs):
    K = len(phis)

    y = 0
    for k in range(K):
        phi, mu, cov = phis[k], mus[k], covs[k]
        y += phi * multivariate_normal(x, mu, cov)

    return y


def likelihood(xs, phis, mus, covs):

    # avoid log(0)
    eps = 1e-8

    L = 0
    N = len(xs)
    for x in xs:
        y = gmm(x, phis, mus, covs)
        L += np.log(y + eps)
    return L / N




def main():
    xs = np.loadtxt("old_faithful.txt")

    phis = np.array([0.5, 0.5])

    mus = np.array([[0.0, 50.0],
                    [0.0, 100.0]])

    covs = np.array([np.eye(2), np.eye(2)])

    K = len(phis)
    N = len(xs)

    MAX_ITER = 100
    THESHOLD = 1e-4

    current_likelihood = likelihood(xs, phis, mus, covs)

    for iter in range(MAX_ITER):
        # E-step
        qs = np.zeros((N, K))
        for n in range(N):
            x = xs[n]
            for k in range(K):
                phi, mu, cov = phis[k], mus[k],covs[k]
                qs[n, k] = phi * multivariate_normal(x, mu, cov)
            qs[n] /= gmm(x, phis, mus, covs)


        # M-step
        qs_sum = qs.sum(axis=0)

        for k in range(K):
            # 1. phis
            phis[k] = qs_sum[k] / N

            # 2. mus
            c = 0
            for n in range(N):
                c += qs[n, k] * xs[n]
            mus[k] = c / qs_sum[k]

            # 3. covs
            c = 0
            for n in range(N):
                z = xs[n] - mus[k]
                z = z[:, np.newaxis] # convert to column vector
                c += qs[n, k] * z @ z.T
            covs[k] = c / qs_sum[k]


        # control termination
        print(f'{current_likelihood:.3f}')

        next_likelihood =  likelihood(xs, phis, mus, covs)
        diff = np.abs(next_likelihood - current_likelihood)

        if diff < THESHOLD:
            break

        current_likelihood = next_likelihood


if __name__ == "__main__":
    main()


