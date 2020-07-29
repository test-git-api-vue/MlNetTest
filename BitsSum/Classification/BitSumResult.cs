using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Numerics;
using System.Text;

namespace MlNetTest.BitsSum.Classification
{
    public class BitSumResult
    {
        [VectorType()]
        public float[] Score { get; set; }

        public uint Label { get; set; }

        public uint PredictedLabel { get; set; }
    }
}
