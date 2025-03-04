# Intent Classification with ONNX Runtime

This project demonstrates how to export scikit-learn models to ONNX format and run them using ONNX Runtime in both Python and Node.js environments.

## ONNX Implementation Overview

The project converts trained scikit-learn classifiers (Logistic Regression, SVM, and KNN) to ONNX format for cross-platform inference.

### Python Export

```python
# Export trained sklearn models to ONNX format
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

def export_classifiers_to_onnx(classifiers, X):
    initial_type = [('float_input', FloatTensorType([None, X.shape[1]]))]
    options = {id(clf): {'nocl': True, 'zipmap': False}}
    
    for name, clf in classifiers.items():
        onx = convert_sklearn(
            clf, 
            initial_types=initial_type,
            options=options,
            target_opset=12
        )
        
        onnx_path = f"models/intent_classifier_{name}.onnx"
        with open(onnx_path, "wb") as f:
            f.write(onx.SerializeToString())
```

### Required Files
- ONNX model files (exported from Python)
- Label mapping JSON file
- AWS credentials for Bedrock embeddings

### Node.js Setup

1. Install dependencies:
```bash
npm install onnxruntime-node @aws-sdk/client-bedrock-runtime dotenv
```

2. Project structure:
```
intent-classifier/
  ├── models/
  │   ├── intent_classifier_logistic_regression.onnx
  │   ├── intent_classifier_svm.onnx
  │   ├── intent_classifier_knn.onnx
  │   └── label_mapping.json
  └── src/
      ├── embeddings.js    # Bedrock embedding logic
      ├── classifier.js    # ONNX inference logic
      └── index.js        # Main entry point
```

### Usage

1. Export models from Python:
```python
# In your Python notebook
classifiers = {
    'logistic_regression': lr_titan,
    'svm': svm_titan, 
    'knn': knn_titan
}
export_classifiers_to_onnx(classifiers, X_train)
```

2. Run inference in Node.js:
```javascript
const classifier = new IntentClassifier();
await classifier.loadModels();
const results = await classifier.predict(embedding);
```

### Key Benefits
- Platform independence - run models without Python
- Potential performance improvements
- Smaller deployment footprint
- Integration with Node.js applications

### Dependencies
- Python:
  - scikit-learn
  - skl2onnx
  - onnxruntime
- Node.js:
  - onnxruntime-node
  - @aws-sdk/client-bedrock-runtime
  - dotenv

### Notes
- Ensure AWS credentials are properly configured
- Bedrock API is still required for text embeddings
- Models should produce identical results to Python implementation
- Current implementation supports ensemble prediction with confidence scores

### Limitations
- Not all scikit-learn models can be converted to ONNX
- Some complex preprocessing steps might need to be reimplemented
- Requires managing both embedding and inference steps

For more details on implementation and usage, refer to the source code and comments.
```
