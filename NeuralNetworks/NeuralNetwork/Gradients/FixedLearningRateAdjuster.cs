using MathNet.Numerics.LinearAlgebra;
using NeuralNetwork.Common.GradientAdjustmentParameters;
using System;

namespace NeuralNetwork.Gradient
{
    internal class FixedLearningRateAdjuster : IGradientAdjuster
    {
        private readonly FixedLearningRateParameters parameters;
        public FixedLearningRateAdjuster(FixedLearningRateParameters parameters)
        {
            this.parameters = parameters ?? throw new ArgumentNullException(nameof(parameters));
        }
        public IGradientAdjustmentParameters Parameters => parameters;

        public Matrix<double> AdjustWeight(Matrix<double> gradient) => parameters.LearningRate * gradient;
        public Matrix<double> AdjustBias(Matrix<double> gradient) => parameters.LearningRate * gradient;
    }
}