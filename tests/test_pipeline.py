import unittest
from src.pipeline.build_pipeline import create_pipeline
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer

class TestCreatePipeline(unittest.TestCase):
    """Test suite for create_pipeline function"""

    def test_pipeline_structure(self):
        """Test that the pipeline has the correct structure"""
        pipe = create_pipeline(n_trees=10)

        # Check that it returns a Pipeline
        self.assertIsInstance(pipe, Pipeline)

        # Check that it has two steps: preprocessor and classifier
        self.assertEqual(len(pipe.steps), 2)
        self.assertEqual(pipe.steps[0][0], "preprocessor")
        self.assertEqual(pipe.steps[1][0], "classifier")


    def test_numeric_transformer(self):
        """Test numeric transformer pipeline"""
        pipe = create_pipeline(n_trees=10)
        preprocessor = pipe.named_steps["preprocessor"]
        numeric_transformer = preprocessor.transformers[0][1]

        # Check it's a Pipeline with 2 steps
        self.assertIsInstance(numeric_transformer, Pipeline)
        self.assertEqual(len(numeric_transformer.steps), 2)

        # Check imputer
        self.assertEqual(numeric_transformer.steps[0][0], "imputer")
        self.assertIsInstance(numeric_transformer.steps[0][1], SimpleImputer)
        self.assertEqual(numeric_transformer.steps[0][1].strategy, "median")

        # Check scaler
        self.assertEqual(numeric_transformer.steps[1][0], "scaler")
        self.assertIsInstance(numeric_transformer.steps[1][1], MinMaxScaler)

    def test_classifier_custom_parameters(self):
        """Test classifier with custom parameters"""
        pipe = create_pipeline(
            n_trees=100,
            max_depth=10,
            max_features="log2"
        )
        classifier = pipe.named_steps["classifier"]

        self.assertEqual(classifier.n_estimators, 100)
        self.assertEqual(classifier.max_depth, 10)
        self.assertEqual(classifier.max_features, "log2")

