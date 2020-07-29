using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Text;

namespace MlNetTest.BitsSum.Classification
{
    public class BitSumData
    {
        [VectorType(4)]
        [ColumnName("Features")]
        public float[] Input { get; set; }

        [ColumnName("Label")]
        public uint Label { get; set; }

        public float Weight { get; set; }

        public BitSumData(int x1, int x2, int x3, int x4)
        {
            Input = new float[] { x1, x2, x3, x4 };
            Label = (uint)(x1 + x2 + x3 + x4);
            Weight = (float)new Random().NextDouble();
        }
    }
}
