using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;
using System;
using System.Linq;

namespace MlNetTest
{
    class Program
    {
        public class XorData
        {
            [LoadColumn(0,1)]
            [VectorType(2)]
            public float[] Features { get; set; }
            
            [LoadColumn(2)]
            public bool Label { get; set; }
        }

        public class XorResult
        {
            public bool Label { get; set; }

            public bool PredictedLabel { get; set; }
        }



        public static void Main()
        {
            var mlContext = new MLContext(seed: 0);

            IDataView data = mlContext.Data.LoadFromTextFile<XorData>("xor_data.txt", separatorChar: ',', hasHeader: true);

            /*
            DataOperationsCatalog.TrainTestData dataSplit = mlContext.Data.TrainTestSplit(data, testFraction: 0.25);
            IDataView trainData = dataSplit.TrainSet;
            IDataView testData = dataSplit.TestSet;
            */
            IDataView trainData = data;
            IDataView testData = data;

            //никаких трансформаций не задаем т.к. входны данные уже вектор, от 0 до 1

            var ldsvmTrnOptions = new LdSvmTrainer.Options()
            {
                NumberOfIterations = 10,
                UseBias = false,
                TreeDepth = 1,
                Sigma = 0.32f,
            };

            // установка алгоритма обучения
            var estimator = mlContext.BinaryClassification.Trainers.LdSvm(ldsvmTrnOptions);

            // обучение
            var trainedModel = estimator.Fit(trainData);
            Console.WriteLine($"Обучение завершено.");

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
            PrintMetrics(trainedModelMetrics);

            PredictionEngine<XorData, XorResult> predictionEngine 
                = mlContext.Model.CreatePredictionEngine<XorData, XorResult>(trainedModel);

            int success = 0;
            int fail = 0;
            var rnd = new Random();
            for(int i = 0; i<1000000; i++)
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

                //Console.WriteLine($"{inputData.Input[0]}^{inputData.Input[1]}={Math.Round(prediction.ScoreOut)} exp: {inputData.Out}");
            }

            Console.WriteLine($"success = {success}, fail = {fail}");
        }

        // Pretty-print BinaryClassificationMetrics objects.
        private static void PrintMetrics(BinaryClassificationMetrics metrics)
        {
            /*
            Console.WriteLine($"Accuracy: {metrics.Accuracy:F2}");
            Console.WriteLine($"AUC: {metrics.AreaUnderRocCurve:F2}");
            Console.WriteLine($"F1 Score: {metrics.F1Score:F2}");
            Console.WriteLine($"Negative Precision: " +
                $"{metrics.NegativePrecision:F2}");

            Console.WriteLine($"Negative Recall: {metrics.NegativeRecall:F2}");
            Console.WriteLine($"Positive Precision: " +
                $"{metrics.PositivePrecision:F2}");
            

            Console.WriteLine($"Positive Recall: {metrics.PositiveRecall:F2}\n");*/
            Console.WriteLine(metrics.ConfusionMatrix.GetFormattedConfusionTable());
        }
    }
}

