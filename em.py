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



    # ======== visualization ===========
    trained_phis = phis
    trained_mus = mus
    trained_covs = covs


    x = np.arange(1, 6, 0.1)
    y = np.arange(40, 100, 0.1)
    X, Y = np.meshgrid(x, y)

    Z = np.zeros_like(X)

    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            point = np.array([X[i, j], Y[i, j]])
            Z[i, j] = gmm(point, trained_phis, trained_mus, trained_covs)



    # plotting
    plt.scatter(xs[:, 0], xs[:, 1], alpha=0.8)
    plt.contour(X, Y, Z)
    plt.xlabel('Eruptions(Min)')
    plt.ylabel('Waiting(Min)')
    plt.savefig('trained_gmm.png')
    plt.clf()
    # ======== visualization ===========



    # ========= generate data based on trained GMM ===========
    N = 500
    new_xs = np.zeros((N, K))
    for n in range(N):
        k = np.random.choice(K, p=trained_phis)
        mu, cov = mus[k], covs[k]
        new_xs[n] = np.random.multivariate_normal(mu, cov)


    plt.scatter(xs[:, 0], xs[:, 1], alpha=0.7)
    plt.scatter(new_xs[:, 0], new_xs[:, 1], alpha=0.7)
    plt.xlabel('Eruptions(Min)')
    plt.ylabel('Waiting(Min)')
    plt.savefig('generate.png')
    # ========= generate data based on trained GMM ===========




if __name__ == "__main__":
    main()


