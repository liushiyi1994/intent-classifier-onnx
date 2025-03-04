// src/embeddings.js
const { BedrockRuntimeClient, InvokeModelCommand } = require("@aws-sdk/client-bedrock-runtime");
require('dotenv').config();

const client = new BedrockRuntimeClient({ region: process.env.AWS_REGION });

async function getEmbeddings(text) {
    const payload = {
        inputText: text
    };

    const command = new InvokeModelCommand({
        modelId: "amazon.titan-embed-text-v2:0",
        contentType: "application/json",
        accept: "application/json",
        body: JSON.stringify(payload)
    });

    try {
        const response = await client.send(command);
        const responseBody = JSON.parse(new TextDecoder().decode(response.body));
        return responseBody.embedding;
    } catch (error) {
        console.error("Error getting embeddings:", error);
        throw error;
    }
}

module.exports = { getEmbeddings };