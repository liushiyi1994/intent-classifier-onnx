// src/classifier.js
const ort = require('onnxruntime-node');
const fs = require('fs');
const path = require('path');

class IntentClassifier {
    constructor() {
        this.sessions = {};
        this.labelMapping = JSON.parse(
            fs.readFileSync(path.join(__dirname, '../models/label_mapping.json'))
        );
    }

    async loadModels() {
        try {
            // Load all three models
            this.sessions.logistic_regression = await ort.InferenceSession.create(
                path.join(__dirname, '../models/intent_classifier_logistic_regression.onnx')
            );
            this.sessions.svm = await ort.InferenceSession.create(
                path.join(__dirname, '../models/intent_classifier_svm.onnx')
            );
            this.sessions.knn = await ort.InferenceSession.create(
                path.join(__dirname, '../models/intent_classifier_knn.onnx')
            );
        } catch (error) {
            console.error("Error loading models:", error);
            throw error;
        }
    }

    async predict(embedding, threshold = 0.5) {
        const results = {};
        const allPredictions = [];
        const allConfidences = {};

        // Ensure embedding is a Float32Array
        const inputData = Float32Array.from(embedding);
        const inputTensor = new ort.Tensor('float32', inputData, [1, inputData.length]);

        // Get predictions from each model
        for (const [name, session] of Object.entries(this.sessions)) {
            try {
                const feeds = { float_input: inputTensor };
                const outputData = await session.run(feeds);
                
                // Get prediction (should be first output)
                const prediction = Array.from(outputData.output_label?.data || outputData.label?.data)[0];
                
                // Get probabilities (should be second output if available)
                const probabilities = outputData.output_probability?.data || 
                                    outputData.probability?.data;
                
                const confidence = probabilities ? 
                    Math.max(...Array.from(probabilities)) : 1.0;
                
                const predictedLabel = this.labelMapping[prediction.toString()];
                
                results[name] = {
                    prediction: predictedLabel,
                    confidence: confidence
                };
                
                allPredictions.push(predictedLabel);
                allConfidences[name] = confidence;
                
            } catch (error) {
                console.error(`Error in ${name} prediction:`, error);
                throw error;
            }
        }

        // Ensemble logic
        const predictionCounts = allPredictions.reduce((acc, pred) => {
            acc[pred] = (acc[pred] || 0) + 1;
            return acc;
        }, {});

        const finalPrediction = Object.entries(predictionCounts)
            .reduce((a, b) => b[1] > a[1] ? b : a)[0];

        const finalConfidence = Object.values(allConfidences)
            .reduce((a, b) => a + b, 0) / Object.values(allConfidences).length;

        results.ensemble = {
            prediction: finalConfidence >= threshold ? finalPrediction : 'others',
            confidence: finalConfidence,
            votes: predictionCounts
        };

        return results;
    }
}

module.exports = IntentClassifier;