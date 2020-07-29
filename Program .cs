using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;
using MlNetTest.BitsSum.Classification;
using System;
using System.Linq;

namespace MlNetTest
{
    public class Program
    {
        public static void Main()
        {
            var sumClassification = new BitsSumClassificationTests();
            sumClassification.BitsSumTest();


        }
    }
}

