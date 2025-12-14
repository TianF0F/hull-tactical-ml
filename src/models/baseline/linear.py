from sklearn.linear_model import Ridge


def make_model(alpha=1.0):
    return Ridge(alpha=alpha)
