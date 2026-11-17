using MathNet.Numerics.LinearAlgebra;
using NeuralNetwork.Common.GradientAdjustmentParameters;
using System;

namespace NeuralNetwork.Gradient
{
    internal class MomentumAdjuster : IGradientAdjuster
    {
        private readonly MomentumParameters parameters;
        private Matrix<double> velocityWeight;   // v pour Ω  (n x m)
        private Matrix<double> velocityBias;     // v pour β  (m x 1)

        public MomentumAdjuster(MomentumParameters parameters)
        {
            this.parameters = parameters ?? throw new ArgumentNullException(nameof(parameters));
        }

        public IGradientAdjustmentParameters Parameters => parameters;

        public Matrix<double> AdjustWeight(Matrix<double> gradient)
        {
            velocityWeight = UpdateVelocity(velocityWeight, gradient);
            return -velocityWeight;     // l'interface renvoie ce qu'on SOUSTRAIT, donc −v
        }

        public Matrix<double> AdjustBias(Matrix<double> gradient)
        {
            velocityBias = UpdateVelocity(velocityBias, gradient);
            return -velocityBias;
        }

        // v ← δ·v − η·g
        private Matrix<double> UpdateVelocity(Matrix<double> velocity, Matrix<double> gradient)
        {
            if (velocity == null)
                velocity = Matrix<double>.Build.Dense(gradient.RowCount, gradient.ColumnCount);   // v = 0 au départ
            return parameters.Momentum * velocity - parameters.LearningRate * gradient;
        }
    }
}