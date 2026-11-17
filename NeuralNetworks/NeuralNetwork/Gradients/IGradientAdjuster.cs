using MathNet.Numerics.LinearAlgebra;
using NeuralNetwork.Common.GradientAdjustmentParameters;

namespace NeuralNetwork.Gradient
{
    internal interface IGradientAdjuster
    {
        IGradientAdjustmentParameters Parameters { get; }
        // Reçoivent le gradient moyenné sur le batch, renvoient la quantité à soustraire du paramètre
        Matrix<double> AdjustWeight(Matrix<double> gradient);   // n x m -> n x m
        Matrix<double> AdjustBias(Matrix<double> gradient);     // m x 1 -> m x 1
    }
}