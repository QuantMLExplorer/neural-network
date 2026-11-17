using MathNet.Numerics.LinearAlgebra;
using NeuralNetwork.Common.Activators;
using NeuralNetwork.Common.Layers;
using NeuralNetwork.Gradient;
using System;

namespace NeuralNetwork.Layers
{
    internal class StandardLayer : ILayer
    {
        private Matrix<double> omega;          // Ω^j   (n x m)
        private Matrix<double> beta;           // β^j   (m x B)
        private Matrix<double> input;          // α^{j-1} (n x B)
        private Matrix<double> zeta;           // ζ^j   (m x B)
        private Matrix<double> alpha;          // α^j   (m x B)
        private Matrix<double> Bj;             // B^j   (m x B)
        private Matrix<double> weightedError;  // [P^j]^T (n x B)
        private Matrix<double> onesRow; 
        private readonly IGradientAdjuster gradientAdjuster;


        public int InputSize { get; }
        public int LayerSize { get; }
        public IActivator Activator { get; }
        private int batchSize;

        public int BatchSize
        {
            get => batchSize;
            set
            {
                batchSize = value;
                onesRow = Matrix<double>.Build.Dense(1, value, 1.0);
                // un seul biais (la 1re colonne) recopié sur les B colonnes
                beta = beta.SubMatrix(0, LayerSize, 0, 1) * onesRow;
            }
        }

        public Matrix<double> Activation => alpha;

        public Matrix<double> WeightedError => weightedError;

        public StandardLayer(Matrix<double> initialWeights, Matrix<double> initialBias, int batchSize, IActivator activator, IGradientAdjuster adjuster)
        {
            LayerSize = initialWeights.ColumnCount;
            InputSize = initialWeights.RowCount;
            Activator = activator ?? throw new ArgumentNullException(nameof(activator));
            gradientAdjuster = adjuster ?? throw new ArgumentNullException(nameof(adjuster));

            omega = initialWeights;  
            beta = initialBias; // (m × 1) à ce stade
            BatchSize = batchSize;     // en dernier : le setter crée onesRow et étend beta en m × B 
        }

        public void Propagate(Matrix<double> input)
        {
            this.input = input;
            // ζ = Ωᵀ·α + β     (m×n)(n×B) + (m×B)     (alpha couche précédente = input)
            zeta = omega.TransposeThisAndMultiply(input) + beta;
            // α = Φ(ζ)
            alpha = zeta.Map(Activator.Apply);
            
        }

        public void BackPropagate(Matrix<double> upstreamWeightedErrors)
        {
            // B^j = Φ'(ζ^j) ⊙ [P^{j+1}]^T
            Bj = upstreamWeightedErrors.PointwiseMultiply(zeta.Map(Activator.ApplyDerivative));
            // [P^j]^T = Ω^j · B^j                    (n×m)(m×B)
            weightedError = omega * Bj;

        }

        public void UpdateParameters()
        {
            // Gradient(Ω^j) = α^{j-1} · [B^j]^T / M     (n×m)
            Matrix<double> gradOmega = input.TransposeAndMultiply(Bj).Divide(BatchSize);
            // Gradient(β^j) = B^j sommé sur le batch / M  (m×1)
            Matrix<double> gradBeta = Bj.TransposeAndMultiply(onesRow).Divide(BatchSize);

            // update Ω, β  depending on the methods
            omega = omega - gradientAdjuster.AdjustWeight(gradOmega) ;
            beta = beta - gradientAdjuster.AdjustBias(gradBeta) * onesRow;   // (m×1)(1×B) = m×B
        }


        public bool Equals(ILayer other)
        {
            throw new NotImplementedException();
        }
    }
}