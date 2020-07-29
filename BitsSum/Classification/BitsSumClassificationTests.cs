using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;
using MlNetTest.Helpers;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using MlNetTest.BitsSum;
using MlNetTest.BitsSum.Classification;

namespace MlNetTest.BitsSum.Classification
{
    [TestClass]
    public class BitsSumClassificationTests
    {
        [TestMethod]
        public void BitsSumTest()
        {
            var mlContext = new MLContext(seed: 0);
            IDataView data = LoadData(mlContext);

            IDataView trainData = data;
            IDataView testData = data;

            var estimator = GetTrainer(mlContext);

            var trainedModel = Train(trainData, estimator);

            PrintResult(mlContext, trainData, trainedModel, testData);
        }

        private void PrintResult(MLContext mlContext, IDataView trainData, 
            TransformerChain<MulticlassPredictionTransformer<LinearMulticlassModelParameters>> trainedModel, 
            IDataView testData)
        {
            var transformedTestData = trainedModel.Transform(testData);

            var predictions = mlContext.Data
                .CreateEnumerable<BitSumResult>(transformedTestData,
                reuseRowObject: false).ToList();

            int success = 0;
            Console.WriteLine($"Тестирование завершено:");
            foreach (var p in predictions)
            {
                Console.WriteLine($"Ожидание: {p.Label}, Результат: {p.PredictedLabel}");
                if (p.Label == p.PredictedLabel)
                {
                    success++;
                }
            }

            Console.WriteLine($"\nИтог: успешно {success}/{trainData.GetRowCount()}\n");

            var metrics = mlContext.MulticlassClassification
                .Evaluate(transformedTestData);

            Console.WriteLine($"Micro Accuracy: {metrics.MicroAccuracy:F2}");
            Console.WriteLine($"Macro Accuracy: {metrics.MacroAccuracy:F2}");
            Console.WriteLine($"Log Loss: {metrics.LogLoss:F2}");
            Console.WriteLine(
                $"Log Loss Reduction: {metrics.LogLossReduction:F2}\n");

            Console.WriteLine(metrics.ConfusionMatrix.GetFormattedConfusionTable());
        }

        private TransformerChain<MulticlassPredictionTransformer<LinearMulticlassModelParameters>> Train(IDataView trainData, EstimatorChain<MulticlassPredictionTransformer<LinearMulticlassModelParameters>> estimator)
        {
        // обучение
        var trainedModel = estimator.Fit(trainData);
            Console.WriteLine($"Обучение завершено.");
            return trainedModel;
        }

        private EstimatorChain<MulticlassPredictionTransformer<LinearMulticlassModelParameters>> GetTrainer(MLContext mlContext)
        {
            var sdcaTrnOptions = new SdcaNonCalibratedMulticlassTrainer.Options()
            {
                ExampleWeightColumnName = "Weight",
                MaximumNumberOfIterations = 100,

            };

            var estimator =
                // Convert the string labels into key types.
                mlContext.Transforms.Conversion
                .MapValueToKey("Label")
                // Apply SdcaNonCalibrated multiclass trainer.
                .Append(mlContext.MulticlassClassification.Trainers
                .SdcaNonCalibrated(sdcaTrnOptions));

            return estimator;
        }

        private IDataView LoadData(MLContext mlContext)
        {
            var data = new List<BitSumData>()
            {
                new BitSumData(0,0,0,0),

                new BitSumData(0,0,0,1),
                new BitSumData(0,0,1,0),
                new BitSumData(0,1,0,0),
                new BitSumData(1,0,0,0),

                new BitSumData(0,0,1,1),
                new BitSumData(0,1,0,1),
                new BitSumData(0,1,1,0),
                new BitSumData(1,1,0,0),
                new BitSumData(1,0,0,1),
                new BitSumData(1,0,1,0),

                new BitSumData(0,1,1,1),
                new BitSumData(1,1,1,0),
                new BitSumData(1,0,1,1),
                new BitSumData(1,1,0,1),

                new BitSumData(1,1,1,1),
            };

            var trainingData = mlContext.Data.LoadFromEnumerable<BitSumData>(data);
            trainingData = mlContext.Data.Cache(trainingData);
            return trainingData;
        }
    }
}
