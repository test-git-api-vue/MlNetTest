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

namespace MlNetTest.Xor
{
    [TestClass]
    public class BitsSumTests
    {
        [TestMethod]
        public void BitsSumTest()
        {
            var mlContext = new MLContext(seed: 0);
            IDataView data = LoadData(mlContext);

            IDataView trainData = data;
            IDataView testData = data;

            KMeansTrainer estimator = GetTrainer(mlContext);

            var trainedModel = Train(trainData, estimator);

            PrintResult(mlContext, trainData, trainedModel, testData);
        }

        private static void PrintResult(MLContext mlContext, IDataView trainData, 
            ClusteringPredictionTransformer<KMeansModelParameters> trainedModel,
            IDataView testData)
        {
            var transformedTestData = trainedModel.Transform(testData);

            var predictions = mlContext.Data
                .CreateEnumerable<BitSumResult>(transformedTestData,
                reuseRowObject: false).ToList();

            int success = 0;
            Console.WriteLine($"Тестрование завершено:");
            foreach (var p in predictions)
            {
                Console.WriteLine($"Ожидание: {p.Label}, Результат: {p.PredictedLabel}");
                if (p.Label == p.PredictedLabel)
                {
                    success++;
                }
            }

            Console.WriteLine($"\nИтог: успешно {success}/{trainData.GetRowCount()}\n");

            var trainedModelMetrics = mlContext.Clustering.Evaluate(transformedTestData);
            ConsolePrintHelper.PrintMetrics(trainedModelMetrics);

            VBuffer<float>[] centroids = default;
            var modelParams = trainedModel.Model;

            PredictionEngine<BitSumData, BitSumResult> predictionEngine
                = mlContext.Model.CreatePredictionEngine<BitSumData, BitSumResult>(trainedModel);

            modelParams.GetClusterCentroids(ref centroids, out int k);
            Console.WriteLine($"\nкластер 1: ({string.Join(", ", centroids[0].GetValues().ToArray())})");
            Console.WriteLine($"кластер 2: ({string.Join(", ", centroids[1].GetValues().ToArray())})");
            Console.WriteLine($"кластер 3: ({string.Join(", ", centroids[2].GetValues().ToArray())})");
            Console.WriteLine($"кластер 4: ({string.Join(", ", centroids[3].GetValues().ToArray())})");
            Console.WriteLine($"кластер 5: ({string.Join(", ", centroids[4].GetValues().ToArray())})");
        }

        private ClusteringPredictionTransformer<KMeansModelParameters> Train(IDataView trainData, KMeansTrainer estimator)
        {
            // обучение
            var trainedModel = estimator.Fit(trainData);
            Console.WriteLine($"Обучение завершено.");
            return trainedModel;
        }

        private KMeansTrainer GetTrainer(MLContext mlContext)
        {
            var ldsvmTrnOptions = new KMeansTrainer.Options()
            {
                InitializationAlgorithm = KMeansTrainer.InitializationAlgorithm.KMeansPlusPlus,
                MaximumNumberOfIterations = 100000,
                NumberOfClusters = 5,
                //NumberOfThreads = 4,
                //AccelerationMemoryBudgetMb = 1024 * 4,
                ExampleWeightColumnName = "Weight"
            };

            // установка алгоритма обучения
            var estimator = mlContext.Clustering.Trainers.KMeans(ldsvmTrnOptions);
            return estimator;
        }

        private IDataView LoadData(MLContext mlContext)
        {
            //4\15

            /*
            var data = new List<BitSumData>()
            {
                new BitSumData(0,0,0,0),

                new BitSumData(0,0,0,1),
                new BitSumData(0,0,1,0),
                new BitSumData(0,1,0,0),
                new BitSumData(1,0,0,0),

                new BitSumData(0,0,1,1),
                new BitSumData(0,1,1,0),
                new BitSumData(1,1,0,0),
                new BitSumData(1,0,0,1),
                new BitSumData(0,1,0,1),

                new BitSumData(0,1,1,1),
                new BitSumData(1,0,1,1),
                new BitSumData(1,1,1,0),
                new BitSumData(1,1,0,1),

                new BitSumData(1,1,1,1),
            };*/


            //6\15
            /*
            var data = new List<BitSumData>()
            {
                new BitSumData(0,0,0,0),

                
                new BitSumData(0,0,1,0),
                
                new BitSumData(1,1,0,0),
                new BitSumData(0,1,0,0),
                new BitSumData(1,0,0,0),
                new BitSumData(0,1,1,0),
                new BitSumData(0,0,1,1),
                new BitSumData(1,1,1,0),
                new BitSumData(1,0,0,1),
                new BitSumData(0,1,0,1),

                new BitSumData(0,1,1,1),
                new BitSumData(1,0,1,1),
                new BitSumData(0,0,0,1),
                new BitSumData(1,1,0,1),

                new BitSumData(1,1,1,1),
            };*/

            //6\15
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

                new BitSumData(0,1,1,1),
                new BitSumData(1,1,1,0),
                new BitSumData(1,0,1,1),
                new BitSumData(1,1,0,1),

                new BitSumData(1,1,1,1),
            };

            /*
            for (int i = 0; i < 50; i++)
            {
                data.AddRange(data);
            }*/

            return mlContext.Data.LoadFromEnumerable<BitSumData>(data);
        }
    }
}
