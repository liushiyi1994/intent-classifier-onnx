// src/index.js
const { getEmbeddings } = require('./embeddings');
const IntentClassifier = require('./classifier');

async function classifyText(text) {
    try {
        // Get embeddings
        const embedding = await getEmbeddings(text);
        
        // Initialize and load classifier
        const classifier = new IntentClassifier();
        await classifier.loadModels();
        
        // Get prediction
        const results = await classifier.predict(embedding);
        
        return results;
    } catch (error) {
        console.error("Error in classification:", error);
        throw error;
    }
}

// Test the pipeline
async function runTest() {
    const testQueries = [
        "Hello, how are you doing today?",
        "I'm looking for a new dress for summer",
        "What's the weather forecast for tomorrow?"
    ];

    console.log("Testing intent classification...\n");
    
    for (const query of testQueries) {
        console.log(`Query: "${query}"`);
        try {
            const results = await classifyText(query);
            console.log("Results:", JSON.stringify(results, null, 2));
            console.log("-".repeat(50), "\n");
        } catch (error) {
            console.error(`Error processing query "${query}":`, error);
        }
    }
}

runTest();