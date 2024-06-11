using UnityEngine;
using Unity.Barracuda;

public class BarracudaInference : MonoBehaviour
{
    public NNModel onnxModel;
    private IWorker worker;

    void Start()
    {
        var model = ModelLoader.Load(onnxModel);
        worker = WorkerFactory.CreateWorker(WorkerFactory.Type.ComputePrecompiled, model);
    }

    void Update()
    {
        // Prepare input tensor
        Tensor input = new Tensor(1, 1, 28, 28); // Example dimensions
        // Fill tensor with input data (replace with your data)
        input[0] = 0.5f;

        // Execute the model
        worker.Execute(input);

        // Get the output
        Tensor output = worker.PeekOutput();

        // Process output data (example)
        Debug.Log("Model output: " + output[0]);

        // Dispose tensors
        input.Dispose();
        output.Dispose();
    }

    void OnDestroy()
    {
        worker.Dispose();
    }
}
