from TFs.normalizer import Normalizer
from TFs.outlier_cleaner import OutlierCleaner
from TFs.mv_imputer import NumMVImputer, NumCatMVImputer, CatMVImputer
from TFs.discretizer import Discretizer
from TFs.identity import Identity

space = [
    {
        "name": "missing_value_imputation",
        "num_tf_options": [NumMVImputer("mean"),
                           NumMVImputer("median"),
                           NumMVImputer("DT"),
                           NumMVImputer("MICE"),
                           NumCatMVImputer("mode")],
        "cat_tf_options": [
            NumCatMVImputer("mode"),
            CatMVImputer("dummy"),
        ],
        "default": [NumMVImputer("mean"), NumCatMVImputer("mode")],
        "init": [(NumMVImputer("mean"), 0.5), (NumCatMVImputer("mode"), 0.5)]
    },
    {
        "name": "normalization",
        "tf_options": [Normalizer("ZS"),
                       Normalizer("MM"),
                       Normalizer("MA"),
                       Normalizer("RS")],
        "default": Normalizer("ZS"),
        "init": (Normalizer("ZS"), 0.5)
    },
    {
        "name": "cleaning_outliers",
        "tf_options": [OutlierCleaner("ZS_4"),
                       OutlierCleaner("ZS_3"),
                       OutlierCleaner("ZS_2"),
                       OutlierCleaner("MAD_3"),
                       OutlierCleaner("MAD_2.5"),
                       OutlierCleaner("MAD_2"),
                       OutlierCleaner("IQR_2"),
                       OutlierCleaner("IQR_1.5"),
                       OutlierCleaner("IQR_1"),
                       Identity()],
        "default": Identity(),
        "init": (Identity(), 0.5)
    },
    {
      "name": "discretization",
      "tf_options": [Discretizer(n_bins=5, strategy="uniform"),
                     Discretizer(n_bins=10, strategy="uniform"),
                     Discretizer(n_bins=20, strategy="uniform"),
                     Discretizer(n_bins=5, strategy="quantile"),
                     Discretizer(n_bins=10, strategy="quantile"),
                     Discretizer(n_bins=20, strategy="quantile"),
                     Identity()],
      "default": Identity(),
      "init": (Identity(), 0.5)
    }
]
