using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;
using MlNetTest.Helpers;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace MlNetTest.Xor
{
    [TestClass]
    public class XorTests
    {
        [TestMethod]
        public void XorTest()
        {
            var mlContext = new MLContext(seed: 0);
            IDataView data = LoadData(mlContext);

            IDataView trainData = data;
            IDataView testData = data;

            LdSvmTrainer estimator = GetTrainer(mlContext);

            var trainedModel = Train(trainData, estimator);

            //--------------тестирование---------
            //определяем качество обучения через метрики.
            //трансформации должны быть те же самые, то есть в данном случае никаких

            var transformedTestData = trainedModel.Transform(testData);

            var predictions = mlContext.Data
                .CreateEnumerable<XorResult>(transformedTestData,
                reuseRowObject: false).ToList();

            Console.WriteLine($"Тестрование завершено:");
            foreach (var p in predictions)
            {
                Console.WriteLine($"Ожидание: {p.Label}, Результат: {p.PredictedLabel}");
            }

            var trainedModelMetrics = mlContext.BinaryClassification.EvaluateNonCalibrated(transformedTestData);
            ConsolePrintHelper.PrintMetrics(trainedModelMetrics);

            PredictionEngine<XorData, XorResult> predictionEngine
                = mlContext.Model.CreatePredictionEngine<XorData, XorResult>(trainedModel);

            int success = 0;
            int fail = 0;
            var rnd = new Random();
            for (int i = 0; i < 1000000; i++)
            {
                XorData inputData = new XorData { Features = new float[] { ((float)Math.Round(rnd.NextDouble())), ((float)Math.Round(rnd.NextDouble())) } };
                inputData.Label = Convert.ToBoolean(inputData.Features[0]) ^ Convert.ToBoolean(inputData.Features[1]);

                // Get Prediction
                var prediction = predictionEngine.Predict(inputData);
                var res = prediction.PredictedLabel;

                if (res == prediction.Label)
                {
                    success++;
                }
                else
                {
                    fail++;
                }
            }

            Console.WriteLine($"success = {success}, fail = {fail}");
        }

        private static BinaryPredictionTransformer<LdSvmModelParameters> Train(IDataView trainData, LdSvmTrainer estimator)
        {
            // обучение
            var trainedModel = estimator.Fit(trainData);
            Console.WriteLine($"Обучение завершено.");
            return trainedModel;
        }

        private static LdSvmTrainer GetTrainer(MLContext mlContext)
        {
            //никаких трансформаций не задаем т.к. входны данные уже вектор, от 0 до 1
            var ldsvmTrnOptions = new LdSvmTrainer.Options()
            {
                NumberOfIterations = 100,
                UseBias = false,
                TreeDepth = 3,
                Sigma = 0.32f,
            };

            // установка алгоритма обучения
            var estimator = mlContext.BinaryClassification.Trainers.LdSvm(ldsvmTrnOptions);
            return estimator;
        }

        private static IDataView LoadData(MLContext mlContext)
        {
            return mlContext.Data.LoadFromTextFile<XorData>("xor/xor_data.txt", separatorChar: ',', hasHeader: false);
        }
    }
}
