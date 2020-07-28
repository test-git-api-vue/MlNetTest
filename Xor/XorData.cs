using Microsoft.ML.Data;

namespace MlNetTest
{
    public class XorData
    {
        [LoadColumn(0, 1)]
        [VectorType(2)]
        public float[] Features { get; set; }

        [LoadColumn(2)]
        public bool Label { get; set; }
    }
}
