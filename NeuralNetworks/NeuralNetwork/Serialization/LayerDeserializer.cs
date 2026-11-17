using MathNet.Numerics.LinearAlgebra;
using NeuralNetwork.Common.Activators;
using NeuralNetwork.Common.Layers;
using NeuralNetwork.Common.Serialization;
using NeuralNetwork.Layers;
using NeuralNetwork.Common.GradientAdjustmentParameters;
using NeuralNetwork.Gradient;
using System;

namespace NeuralNetwork.Serialization
{
    internal static class LayerDeserializer
    {
        public static ILayer Deserialize(ISerializedLayer serializedLayer, int batchSize)
        {
            switch (serializedLayer.Type)
            {
                case LayerType.Standard:
                    var standardSerialized = serializedLayer as SerializedStandardLayer;
                    return DeserializeStandardLayer(standardSerialized, batchSize);
                //TODO: complete depending on what layers are implemented
                case LayerType.Dropout:
                    var dropoutSerialized = serializedLayer as SerializedDropoutLayer;
                    return new DropoutLayer(dropoutSerialized.LayerSize, dropoutSerialized.KeepProbability, batchSize);
                default:
                    throw new InvalidOperationException("Unknown layer type to deserialize");
            }
        }
        // This method can be modified if necessary
        private static ILayer DeserializeStandardLayer(SerializedStandardLayer standardSerialized, int batchSize)
        {
            var weights = Matrix<double>.Build.DenseOfRowArrays(standardSerialized.Weights);
            var bias = Matrix<double>.Build.DenseOfColumnArrays(new double[][] { standardSerialized.Bias });
            var activator = ActivatorFactory.Build(standardSerialized.ActivatorType);
            var gradientAdjuster = BuildAdjuster(standardSerialized.GradientAdjustmentParameters);
            return new StandardLayer(weights, bias, batchSize, activator, gradientAdjuster);
        }
        private static IGradientAdjuster BuildAdjuster(IGradientAdjustmentParameters parameters)
        {
            if (parameters == null)
                throw new InvalidOperationException("Missing gradient adjustment parameters in the serialized layer");

            switch (parameters.Type)
            {
                case GradientAdjustmentType.FixedLearningRate:
                    return new FixedLearningRateAdjuster((FixedLearningRateParameters)parameters);
                case GradientAdjustmentType.Momentum:
                    return new MomentumAdjuster((MomentumParameters)parameters);
                // Adam 
                default:
                    throw new NotSupportedException("Gradient adjustment not implemented: " + parameters.Type);
            }
        }
    }
}