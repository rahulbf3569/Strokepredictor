from stroke_risk.features import build_preprocessor, split_xy
from stroke_risk.config import ALL_FEATURES


def test_preprocessor_fit_transform(sample_df):
    X, y = split_xy(sample_df)
    prep = build_preprocessor()
    Xt = prep.fit_transform(X)
    assert Xt.shape[0] == len(X)
    assert Xt.shape[1] > len(ALL_FEATURES)  # one-hot expands