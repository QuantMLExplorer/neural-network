using MathNet.Numerics.LinearAlgebra;
using NeuralNetwork.Common;
using NeuralNetwork.Common.Layers;
using System;

namespace NeuralNetwork.Layers
{
    internal class DropoutLayer : ILayer, IComponentWithMode
    {
        private readonly double keepProbability;       // p
        private readonly Random random = new Random();
        private int batchSize;
        private Matrix<double> onesRow;                // 1 x B
        private Matrix<double> mask;                   // n x B : μ recopié sur le batch
        private Matrix<double> alpha;
        private Matrix<double> weightedError;

        public Mode Mode { get; set; }
        public int InputSize { get; }
        public int LayerSize { get; }
        internal double KeepProbability => keepProbability;
        public Matrix<double> Activation => alpha;
        public Matrix<double> WeightedError => weightedError;

        public int BatchSize
        {
            get => batchSize;
            set
            {
                batchSize = value;
                onesRow = Matrix<double>.Build.Dense(1, value, 1.0);
            }
        }

        public DropoutLayer(int layerSize, double keepProbability, int batchSize)
        {
            if (keepProbability <= 0 || keepProbability > 1)
                throw new ArgumentOutOfRangeException(nameof(keepProbability), "p doit être dans ]0, 1]");
            InputSize = layerSize;
            LayerSize = layerSize;
            this.keepProbability = keepProbability;
            BatchSize = batchSize;
        }

        public void Propagate(Matrix<double> input)
        {
            if (Mode == Mode.Training)
            {
                var mu = Vector<double>.Build.Dense(LayerSize, i => random.NextDouble() < keepProbability ? 1.0 / keepProbability : 0.0);
                mask = Matrix<double>.Build.DenseOfColumnVectors(mu) * onesRow;   // (n×1)(1×B)
                alpha = input.PointwiseMultiply(mask);
            }
            else
            {
                alpha = input;
            }
        }

        public void BackPropagate(Matrix<double> upstreamWeightedErrors)
        {
            weightedError = Mode == Mode.Training ? upstreamWeightedErrors.PointwiseMultiply(mask) : upstreamWeightedErrors;
        }

        public void UpdateParameters() { }

        public bool Equals(ILayer other) =>
            other is DropoutLayer d && d.LayerSize == LayerSize && d.keepProbability == keepProbability;
    }
}