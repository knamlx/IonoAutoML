import unittest

import numpy as np
import pandas as pd

from run_ml_baselines import impute_frames


class ImputeFrameTests(unittest.TestCase):
    """Проверяет заполнение пропусков в ML-признаках."""
    def test_impute_frames_preserves_empty_training_features(self):
        """Выполняет вспомогательный шаг test_impute_frames_preserves_empty_training_features в текущем сценарии."""
        features = ["observed", "empty_in_train"]
        train = pd.DataFrame(
            {
                "observed": [1.0, np.nan, 3.0],
                "empty_in_train": [np.nan, np.nan, np.nan],
            },
            index=[10, 11, 12],
        )
        test = pd.DataFrame(
            {
                "observed": [np.nan, 5.0],
                "empty_in_train": [7.0, np.nan],
            },
            index=[20, 21],
        )

        train_imputed, validation_imputed, test_imputed = impute_frames(train, test, features)

        self.assertIsNone(validation_imputed)
        self.assertEqual(list(train_imputed.columns), features)
        self.assertEqual(list(test_imputed.columns), features)
        self.assertEqual(train_imputed.shape, (3, 2))
        self.assertEqual(test_imputed.shape, (2, 2))
        self.assertTrue((train_imputed["empty_in_train"] == 0.0).all())
        self.assertEqual(test_imputed.loc[20, "empty_in_train"], 7.0)


if __name__ == "__main__":
    unittest.main()
