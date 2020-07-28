using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;

namespace MlNetTest
{
    class Program
    {
        public class XorData
        {
            [LoadColumn(0,1)]
            [VectorType(2)]
            public float[] Input { get; set; }
            
            [LoadColumn(2)]
            [ColumnName("Label")]
            public float Out { get; set; }

            [LoadColumn(2)]
            [ColumnName("Score")]
            public float ScoreOut { get; set; }
        }

        public class XorResult
        {
            [ColumnName("Score")]
            public float ScoreOut { get; set; }

            [ColumnName("Result")]
            public float ResultOut { get; set; }

            [ColumnName("Rank")]
            public float RankOut { get; set; }
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

            // установка алгоритма обучения
            var sdcaEstimator = mlContext.Regression.Trainers.Sdca(featureColumnName: "Input");

            // обучение
            var trainedModel = sdcaEstimator.Fit(trainData);
            Console.WriteLine($"Обучение завершено. Веса: {(string.Join(",", trainedModel.Model.Weights.ToArray()))}");

            //--------------тестирование---------
            //определяем качество обучения через метрики.
            //трансформации должны быть те же самые, то есть в данном случае никаких

            RegressionMetrics trainedModelMetrics = mlContext.Regression.Evaluate(testData);
            double rSquared = trainedModelMetrics.RSquared;

            Console.WriteLine($"Тестрование завершено. Вероятность ошибки RSquared={rSquared}");

            PredictionEngine<XorData, XorResult> predictionEngine 
                = mlContext.Model.CreatePredictionEngine<XorData, XorResult>(trainedModel);

            int success = 0;
            int fail = 0;
            var rnd = new Random();
            for(int i = 0; i<10000; i++)
            {
                XorData inputData = new XorData { Input = new float[] { ((float)Math.Round(rnd.NextDouble())), ((float)Math.Round(rnd.NextDouble())) } };
                inputData.Out = (int)Math.Round(inputData.Input[0]) ^ (int)Math.Round(inputData.Input[1]);

                // Get Prediction
                var prediction = predictionEngine.Predict(inputData);
                var res = Math.Round(prediction.ScoreOut);

                if (res == inputData.Out)
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
    }
}

