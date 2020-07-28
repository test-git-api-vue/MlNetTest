using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Text;

namespace MlNetTest.Helpers
{
    public static class ConsolePrintHelper
    {
        public static void PrintMetrics(BinaryClassificationMetrics metrics)
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

        public static void PrintMetrics(ClusteringMetrics metrics)
        {
            Console.WriteLine($"\nNormalized Mutual Information: " +
                $"{metrics.NormalizedMutualInformation:F2}");

            Console.WriteLine($"Average Distance: " +
                $"{metrics.AverageDistance:F2}");

            Console.WriteLine($"Davies Bouldin Index: " +
                $"{metrics.DaviesBouldinIndex:F2}");
        }
    }
}
