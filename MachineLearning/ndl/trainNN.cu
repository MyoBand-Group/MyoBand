/*  Author: Nikola D. Lilov, 14 Sep 2026
 *
 *  Desc: Trains a simple NxM neural network with an Adam optimizer, decreasing LR,
 *  a squared loss function and the x/1+abs(x) activation function. Instead of
 *  propagating one training example at a time, the GPU is fed `parallel_samples` at
 *  once. Specifically, it uses a cuBLAS-based matmul for which all the samples are
 *  caluclated at the same time. (The kernels should work with reasonable parameters,
 *  but I would just give them 512 threads & the repsective amuont of blocks).
 *  Requires a training dataset file with alternating input and output lines. Saves
 *  (and can load) the trained net to a file for later retraining or inferencing.
 *
 *  Requirements:       CUDA-compatible GPU (NVIDIA) ; CUDA toolkit drivers ; cuBLAS.
 *
 *  Compilation eg.:    nvcc trainNN.cu -o trainNN.exe -lcublas
 *                      nvcc trainNN.cu -o trainNN.exe -lcublas -O3 -arch=sm_XX
 *
 *  Common Usage:       trainNN.exe -d dataset.txt
 *  Expanded Usage:     trainNN.exe -h
 */

/*  The data should be in the format:

    feature1 feature2 feature3 ... featureX
    output1 output2 ... outputY
    feature1 feature2 feature3 ... featureX
    output1 output2 ... outputY
    ...

*/

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <iomanip>
#include <cmath>
#include <random>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>
#include <algorithm>
#include <numeric>

#define SAMPLES 512

std::ofstream CUDA_errorlog;
void report_cuda_error(const char *file, int line, cudaError_t result)
{
    std::time_t timestamp = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    char time_buffer[32];
    std::strftime(time_buffer, sizeof(time_buffer), "%Y-%m-%d %H:%M:%S", std::localtime(&timestamp));

    if (!CUDA_errorlog.is_open())
        CUDA_errorlog.open("CUDA_errorlog.txt", std::ios::out | std::ios::trunc);

    if (CUDA_errorlog.is_open())
        CUDA_errorlog << '[' << time_buffer << "]  |  CUDA Runtime Error: " << file << ':' << line << "  |  Error " << result << " = " << cudaGetErrorString(result) << ".\n";
    else
        std::fprintf(stderr, "CUDA Runtime Error: %s:%d = %s\n", file, line, cudaGetErrorString(result));
}
void report_cuda_error(const char *file, int line, cublasStatus_t result)
{
    std::time_t timestamp = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    char time_buffer[32];
    std::strftime(time_buffer, sizeof(time_buffer), "%Y-%m-%d %H:%M:%S", std::localtime(&timestamp));

    if (!CUDA_errorlog.is_open())
        CUDA_errorlog.open("CUDA_errorlog.txt", std::ios::out | std::ios::trunc);

    if (CUDA_errorlog.is_open())
        CUDA_errorlog << '[' << time_buffer << "]  |  cuBLAS Error:      " << file << ':' << line << "  |  Error " << result << " = " << cublasGetStatusString(result) << ".\n";
    else
        std::fprintf(stderr, "cuBLAS Error: %s:%d = %s\n", file, line, cublasGetStatusString(result));
}
#define CC(expr_to_check)                                         \
    do                                                            \
        if (expr_to_check != 0) /* 0 means success */             \
            report_cuda_error(__FILE__, __LINE__, expr_to_check); \
    while (false)

void help(const std::string &exe)
{
    std::cerr << " Common Usages:\n"
              << "   " << exe << " -d dataset.txt\n"
              << "   " << exe << " -l trained_net.txt -i in.txt -o out.txt\n"
              << "\n"
              << " All Options: (order isn't important)\n"
              << "   -d --data <path>     Training data file containing space-separated input and output lines.\n"
              << "   -l --load <path>     Load a trained model from file to inference* with.\n"
              << "   -s --save <path>     Save the trained model to file. Defaults to trained_net.txt.\n"
              << "   -t --testing <float> Set what proportion of the dataset (-d) should be for testing. Defaults to 0.20.\n"
              << "   -b --batch_size <int>         Set how many samples should be propagated in parallel before evolving (0 <=> BGD). Defaults to BGD/4.\n"
              << "   -lr --learning_rate <float>   Set the initial learning rate for training. Defaults to 0.1/D.\n"
              << "   -p --print <int>     How often to print status updates, in milliseconds. Defaults to 5000.\n"
              << "   -i --inputs <path>   Load inference input from file.\n"
              << "   -o --output <path>   Save inference output to file. Defaults to output.txt.\n"
              << "   -h --help            Show this usage information.\n"
              << "\n"
              << "  *Note: If both --load and --data are specified, the model will be loaded and then further trained on the data as needed.\n"
              << " **Note: This code utilizes the GPU's capability to train on multiple examples at once (SIMT).\n"
              << "         GPU's use \"warps\", which parallelize 32 threads at a time. Thus, it is recommended that N is a multiple of 32\n"
              << "\n"
              << " Example --data file format:\n"
              << "   <input1> <input2> <input3> ... <inputX>\n   <output1> <output2> ... <outputY>\n   <input1> <input2> <input3> ... <inputX>\n   <output1> <output2> ... <outputY>\n   ..."
              << "\n\n\n";

    exit(0);
}

void initialize(int &argc, char **argv, std::ifstream &data_file, std::ifstream &load_file, std::ifstream &inference_file, std::ofstream &output_file, int &whereToSave, std::size_t &batch_size, float &part_testing, float &LR, int &print)
{
    bool show_help = false;
    if (argc == 1)
        show_help = true;
    else
        for (int i = 1; i < argc; i++)
        {
            std::string arg = argv[i];
            if (arg == "--data" || arg == "-d")
            {
                if (++i >= argc)
                {
                    std::cerr << "Missing value for --data\n";
                    exit(-1);
                }
                data_file.open(argv[i]);
                if (!data_file.is_open())
                {
                    std::cerr << "Error: could not open data file: " << argv[i] << "\n";
                    exit(-1);
                }
            }
            else if (arg == "--inputs" || arg == "-i")
            {
                if (++i >= argc)
                {
                    std::cerr << "Missing value for --inputs\n";
                    exit(-1);
                }
                inference_file.open(argv[i]);
                if (!inference_file.is_open())
                {
                    std::cerr << "Error: could not open input file: " << argv[i] << "\n";
                    exit(-1);
                }
            }
            else if (arg == "--output" || arg == "-o")
            {
                if (++i >= argc)
                {
                    std::cerr << "Missing value for --output\n";
                    exit(-1);
                }
                output_file.open(argv[i]);
                if (!output_file.is_open())
                {
                    std::cerr << "Error: could not open output file: " << argv[i] << "\n";
                    exit(-1);
                }
            }
            else if (arg == "--load" || arg == "-l")
            {
                if (++i >= argc)
                {
                    std::cerr << "Missing value for --load\n";
                    exit(-1);
                }
                load_file.open(argv[i]);
                if (!load_file.is_open())
                {
                    std::cerr << "Error: could not open load file: " << argv[i] << "\n";
                    exit(-1);
                }
            }
            else if (arg == "--save" || arg == "-s")
            {
                if (++i >= argc)
                {
                    std::cerr << "Missing value for --save\n";
                    exit(-1);
                }
                whereToSave = i;
            }
            else if (arg == "--learning_rate" || arg == "-lr" || arg == "--lr")
            {
                if (++i >= argc)
                {
                    std::cerr << "Missing value for --lr\n";
                    exit(-1);
                }
                LR = std::stod(argv[i]);
            }
            else if (arg == "--batch_size" || arg == "-b")
            {
                if (++i >= argc)
                {
                    std::cerr << "Missing value for --batch_size\n";
                    exit(-1);
                }
                batch_size = std::stoi(argv[i]);
            }
            else if (arg == "--testing" || arg == "-t")
            {
                if (++i >= argc)
                {
                    std::cerr << "Missing value for --testing\n";
                    exit(-1);
                }
                part_testing = std::stof(argv[i]);
            }
            else if (arg == "--print" || arg == "-p")
            {
                if (++i >= argc)
                {
                    std::cerr << "Missing value for --print\n";
                    exit(-1);
                }
                print = std::stoi(argv[i]);
                if (print <= 0)
                {
                    std::cerr << "Error: --print must be greater than zero seconds\n";
                    exit(-1);
                }
            }
            else if (arg == "--help" || arg == "-h")
            {
                show_help = true;
            }
            else
            {
                std::cerr << "Unknown argument: " << arg << "\n";
                show_help = true;
            }
        }
    if (show_help)
        help(argv[0]);
}

struct layer
{
    bool onDevice = false;
    std::size_t input_size, output_size;
    float *value, *delta, *bias, *grad_b, *weight, *grad_w;
    // Every node stores its value and delta=dL/dvalue. More specifically, for each of those it has samples slots (2D array).
    // Every node has a bias (1D array).
    // Every node has a weight to every next node (2D array).
    // Use the grad_ variables to store the summed results of the deltas from the samples to then update the parameter at once at the end of the batch.
    float *m_b, *v_b, *m_w, *v_w;
    // Variables storing information for the Adam optimizer (m is for the momentum step; v is for the RMSprop step).

    layer() = default;

    layer(const std::size_t &in_size, const std::size_t &batch_size) // Used only as an expected output to compare against => no logic stored inside
    {
        input_size = in_size;
        output_size = 0;
        value = (float *)malloc(input_size * batch_size * sizeof(float));
        delta = bias = grad_b = weight = grad_w = m_b = v_b = m_w = v_w = nullptr;
    }

    layer(const std::size_t &in_size, const std::size_t &out_size, const std::string *load, const std::size_t &batch_size) // Used only for loading a ready-made network. Assumes load_size=in.
    {
        input_size = in_size;
        output_size = out_size;

        value = (float *)malloc(input_size * batch_size * sizeof(float));
        memset(value, 0.0f, input_size * batch_size * sizeof(float));
        delta = (float *)malloc(input_size * batch_size * sizeof(float));
        memset(delta, 0.0f, input_size * batch_size * sizeof(float));

        weight = (float *)malloc(input_size * output_size * sizeof(float));
        bias = (float *)malloc(input_size * sizeof(float));
        std::istringstream in;
        for (int j = 0; j < input_size; j++)
        {
            in.clear();
            in.str(load[j]);
            for (int k = 0; k < output_size; k++)
                in >> weight[j * output_size + k];
            in >> bias[j];
        }

        grad_b = (float *)malloc(input_size * sizeof(float));
        memset(grad_b, 0.0f, input_size * sizeof(float));
        grad_w = (float *)malloc(input_size * output_size * sizeof(float));
        memset(grad_w, 0.0f, input_size * output_size * sizeof(float));

        m_b = (float *)malloc(input_size * sizeof(float));
        memset(m_b, 0.0f, input_size * sizeof(float));
        v_b = (float *)malloc(input_size * sizeof(float));
        memset(v_b, 0.0f, input_size * sizeof(float));
        m_w = (float *)malloc(input_size * output_size * sizeof(float));
        memset(m_w, 0.0f, input_size * output_size * sizeof(float));
        v_w = (float *)malloc(input_size * output_size * sizeof(float));
        memset(v_w, 0.0f, input_size * output_size * sizeof(float));
    }

    layer(const std::size_t &in_size, const std::size_t &out_size, std::uniform_real_distribution<float> &distribution, std::mt19937 &gen, const std::size_t &batch_size) // Normal Constructor
    {
        input_size = in_size;
        output_size = out_size;

        value = (float *)malloc(input_size * batch_size * sizeof(float));
        memset(value, 0.0f, input_size * batch_size * sizeof(float));
        delta = (float *)malloc(input_size * batch_size * sizeof(float));
        memset(delta, 0.0f, input_size * batch_size * sizeof(float));

        weight = (float *)malloc(input_size * output_size * sizeof(float));
        bias = (float *)malloc(input_size * sizeof(float));
        for (int i = 0; i < input_size; i++)
        {
            for (int j = 0; j < output_size; j++)
                weight[i * output_size + j] = distribution(gen);
            bias[i] = distribution(gen);
        }

        grad_b = (float *)malloc(input_size * sizeof(float));
        memset(grad_b, 0.0f, input_size * sizeof(float));
        grad_w = (float *)malloc(input_size * output_size * sizeof(float));
        memset(grad_w, 0.0f, input_size * output_size * sizeof(float));

        m_b = (float *)malloc(input_size * sizeof(float));
        memset(m_b, 0.0f, input_size * sizeof(float));
        v_b = (float *)malloc(input_size * sizeof(float));
        memset(v_b, 0.0f, input_size * sizeof(float));
        m_w = (float *)malloc(input_size * output_size * sizeof(float));
        memset(m_w, 0.0f, input_size * output_size * sizeof(float));
        v_w = (float *)malloc(input_size * output_size * sizeof(float));
        memset(v_w, 0.0f, input_size * output_size * sizeof(float));
    }

    __host__ __device__ std::size_t size() const
    {
        return input_size;
    }

    ~layer()
    {
        if (onDevice)
        {
            CC(cudaFree(value));
            CC(cudaFree(delta));
            CC(cudaFree(bias));
            CC(cudaFree(grad_b));
            CC(cudaFree(weight));
            CC(cudaFree(grad_w));
            CC(cudaFree(m_b));
            CC(cudaFree(v_b));
            CC(cudaFree(m_w));
            CC(cudaFree(v_w));
        }
        else
        {
            free(value);
            free(delta);
            free(bias);
            free(grad_b);
            free(weight);
            free(grad_w);
            free(m_b);
            free(v_b);
            free(m_w);
            free(v_w);
        }
    }
};
using NN = layer *;
struct d_NN
{
    NN head;
    NN shells;
};

// Copying "net" to/from device is a two-step process. Same for datasets.
// Using cudaMemcpy on just "net" would only make device-side copies of the pointers towards the arrays (which are host-side).
void layer_to_device_shell(layer &d, const layer &host_layer, std::size_t value_count)
{
    d.onDevice = true;
    d.input_size = host_layer.input_size;
    d.output_size = host_layer.output_size;

    CC(cudaMalloc(&d.value, value_count * sizeof(float)));
    CC(cudaMemcpy(d.value, host_layer.value, value_count * sizeof(float), cudaMemcpyHostToDevice));

    if (host_layer.bias != nullptr)
    {
        std::size_t wsize = host_layer.size() * host_layer.output_size * sizeof(float);
        std::size_t bsize = host_layer.size() * sizeof(float);

        CC(cudaMalloc(&d.delta, value_count * sizeof(float)));
        CC(cudaMemcpy(d.delta, host_layer.delta, value_count * sizeof(float), cudaMemcpyHostToDevice));
        CC(cudaMalloc(&d.bias, bsize));
        CC(cudaMemcpy(d.bias, host_layer.bias, bsize, cudaMemcpyHostToDevice));
        CC(cudaMalloc(&d.grad_b, bsize));
        CC(cudaMemcpy(d.grad_b, host_layer.grad_b, bsize, cudaMemcpyHostToDevice));
        CC(cudaMalloc(&d.weight, wsize));
        CC(cudaMemcpy(d.weight, host_layer.weight, wsize, cudaMemcpyHostToDevice));
        CC(cudaMalloc(&d.grad_w, wsize));
        CC(cudaMemcpy(d.grad_w, host_layer.grad_w, wsize, cudaMemcpyHostToDevice));

        CC(cudaMalloc(&d.m_b, bsize));
        CC(cudaMemcpy(d.m_b, host_layer.m_b, bsize, cudaMemcpyHostToDevice));
        CC(cudaMalloc(&d.v_b, bsize));
        CC(cudaMemcpy(d.v_b, host_layer.v_b, bsize, cudaMemcpyHostToDevice));
        CC(cudaMalloc(&d.m_w, wsize));
        CC(cudaMemcpy(d.m_w, host_layer.m_w, wsize, cudaMemcpyHostToDevice));
        CC(cudaMalloc(&d.v_w, wsize));
        CC(cudaMemcpy(d.v_w, host_layer.v_w, wsize, cudaMemcpyHostToDevice));
    }
    else
        d.delta = d.bias = d.grad_b = d.weight = d.grad_w = d.m_b = d.v_b = d.m_w = d.v_w = nullptr;
}
void NN_to_device(d_NN &d_net, const NN &host_net, const int &D, const std::size_t &batch_size)
{
    d_net.shells = (NN)malloc((D + 2) * sizeof(layer));
    for (int i = 0; i < D + 2; i++)
    {
        std::size_t value_count = batch_size * host_net[i].size();
        layer_to_device_shell(d_net.shells[i], host_net[i], value_count);
    }

    CC(cudaMalloc(&d_net.head, (D + 2) * sizeof(layer)));
    CC(cudaMemcpy(d_net.head, d_net.shells, (D + 2) * sizeof(layer), cudaMemcpyHostToDevice));
}
void NN_from_device(NN host_net, const d_NN &d_net, const int &D)
{
    for (int i = 0; i < D + 2; i++)
    {
        std::size_t wsize = host_net[i].input_size * host_net[i].output_size * sizeof(float);
        std::size_t bsize = host_net[i].input_size * sizeof(float);
        CC(cudaMemcpy(host_net[i].weight, d_net.shells[i].weight, wsize, cudaMemcpyDeviceToHost));
        CC(cudaMemcpy(host_net[i].bias, d_net.shells[i].bias, bsize, cudaMemcpyDeviceToHost));
    }
}
void free_NN(NN &host_net, const int &D)
{
    for (int i = 0; i <= D + 1; i++)
        host_net[i].~layer(); // onDevice = false;
    cudaFreeHost(host_net);
}
void free_d_NN(d_NN &d_net, const int &D)
{
    for (int i = 0; i <= D + 1; i++)
        d_net.shells[i].~layer(); // onDevice = true;
    free(d_net.shells);
    CC(cudaFree(d_net.head));
    // free(&d_net);
}

void dataset_to_device(const float *host_data_in, const float *host_data_out, float *&dev_data_in, float *&dev_data_out, const std::size_t n, const std::size_t input_size, const std::size_t output_size)
{
    CC(cudaMalloc(&dev_data_in, n * input_size * sizeof(float)));
    CC(cudaMemcpy(dev_data_in, host_data_in, n * input_size * sizeof(float), cudaMemcpyHostToDevice));
    CC(cudaMalloc(&dev_data_out, n * output_size * sizeof(float)));
    CC(cudaMemcpy(dev_data_out, host_data_out, n * output_size * sizeof(float), cudaMemcpyHostToDevice));
}

__device__ __forceinline__ int bitceil(const int &x)
{
    return 1u << (32 - __clz(x - 1));
}
__device__ float reduce(const float val, const bool flag)
{
    const int tid = threadIdx.x;
    __shared__ float new_vec[SAMPLES];
    new_vec[tid] = val;

    for (int stride = bitceil(blockDim.x) / 2; stride > 0; stride >>= 1)
    {
        __syncthreads();
        if (flag)
            if (tid + stride < blockDim.x && tid < stride)
                new_vec[tid] += new_vec[tid + stride];
    }

    __syncthreads();
    return new_vec[0];
}

cublasHandle_t handle;
cudaError_t matmul(float *C, float *A, float *B, const int &first_dim_C, const int &second_dim_C, const int &common_dim_AB, const std::string &context)
{
    float alpha = 1.0f, beta;
    if (context == "forward" || context == "Forward" || context == "fp" || context == "FP")
    {
        beta = 1.0f;
        CC(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, second_dim_C, first_dim_C, common_dim_AB, &alpha, B, second_dim_C, A, first_dim_C, &beta, C, second_dim_C));
    }
    else if (context == "backward" || context == "Backward" || context == "bp" || context == "BP")
    {
        beta = 0.0f;
        CC(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, second_dim_C, first_dim_C, common_dim_AB, &alpha, B, second_dim_C, A, common_dim_AB, &beta, C, second_dim_C));
    }
    else if (context == "gradient" || context == "Gradient" || context == "gd" || context == "GD")
    {
        beta = 0.0f;
        CC(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, second_dim_C, first_dim_C, common_dim_AB, &alpha, B, common_dim_AB, A, common_dim_AB, &beta, C, second_dim_C));
    }
    else
        std::cerr << "Unexpected context given for `void matmul()`. Options: \"forward\", \"backward\", \"gradient\"\n";

    return cudaGetLastError();
}

__global__ void loss(float *ans, NN net, const int D, const float *data_out, const std::size_t data_size, const int *order, const std::size_t order_size, const std::size_t batch_size)
{
    const int node = blockIdx.y, num_samples = batch_size, sample = blockIdx.x * blockDim.x + threadIdx.x;
    const bool flag = (sample < num_samples && sample < order_size && order[sample] < data_size);

    if (flag && node < net[D + 1].size())
    {
        float diff = net[D + 1].value[node * num_samples + sample] - data_out[node * data_size + order[sample]];
        atomicAdd(ans, diff * diff);
    }
}
__device__ __forceinline__ float d_loss(const float &expected, const float &output)
{
    return 2 * (output - expected); // Derivative of the loss function with respect to the output
}
__global__ void activation(float *mat, const std::size_t mat_size)
{
    for (std::size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < mat_size; i += gridDim.x * blockDim.x)
        mat[i] /= (1.0f + fabsf(mat[i])); // For every node:  val  =  val  /  (1 + |val|)
}
__device__ __forceinline__ float d_activation(const float &y)
{
    float t = 1.0f - abs(y);
    return t * t;
}

__global__ void set_input(NN net, const float *training_in, const std::size_t training_size, const int *order, const std::size_t order_size, const std::size_t batch_size)
{
    const int node = blockIdx.y, num_samples = batch_size, sample = blockIdx.x * blockDim.x + threadIdx.x;
    const bool flag = (sample < num_samples && sample < order_size && order[sample] < training_size);

    if (flag && node < net[0].size())
        net[0].value[node * num_samples + sample] = training_in[node * training_size + order[sample]];
}
__global__ void bias(float *value, float *bias, const std::size_t l_size, const std::size_t batch_size)
{
    const int node = blockIdx.y, num_samples = batch_size, sample = blockIdx.x * blockDim.x + threadIdx.x;

    if (sample < num_samples && node < l_size)
        value[node * num_samples + sample] = bias[node];
}
cudaError_t forward_propagate(d_NN &d_net, NN &net, const int &D, const float *training_in, const std::size_t &training_size, const int *order, const std::size_t order_size, const std::size_t &batch_size)
{

    set_input<<<dim3((batch_size + SAMPLES - 1) / SAMPLES, net[0].size()), SAMPLES>>>(d_net.head, training_in, training_size, order, order_size, batch_size);

    for (int l = 1; l <= D + 1; l++)
    {
        bias<<<dim3((batch_size + SAMPLES - 1) / SAMPLES, net[l].size()), SAMPLES>>>(d_net.shells[l].value, d_net.shells[l].bias, net[l].size(), batch_size);
        CC(matmul(d_net.shells[l].value, d_net.shells[l - 1].weight, d_net.shells[l - 1].value, net[l].size(), batch_size, net[l - 1].size(), "forward"));
        activation<<<(net[l].size() + SAMPLES - 1) / SAMPLES, SAMPLES>>>(d_net.shells[l].value, d_net.shells[l].size() * batch_size);
    }

    return cudaGetLastError();
}

__global__ void compare_outputs(NN net, const int D, const float *training_out, const std::size_t training_size, const int *order, const std::size_t order_size, const std::size_t batch_size)
{
    const int node = blockIdx.y, num_samples = batch_size, sample = blockIdx.x * blockDim.x + threadIdx.x;
    const bool flag = (sample < num_samples && sample < order_size && order[sample] < training_size && node < net[D + 1].size());

    const int bid = node * num_samples + sample;
    if (flag)
        net[D + 1].delta[bid] = d_loss(training_out[node * training_size + order[sample]], net[D + 1].value[bid]) * d_activation(net[D + 1].value[bid]);
    __syncthreads();
    float temp = reduce(flag ? net[D + 1].delta[bid] : 0.0f, flag);

    if (flag)
        if (threadIdx.x == 0)
            atomicAdd(&net[D + 1].grad_b[node], temp);
}
__global__ void update_grad_b(NN net, const int l, const std::size_t training_size, const int *order, const std::size_t order_size, const std::size_t batch_size)
{
    const int node = blockIdx.y, num_samples = batch_size, sample = blockIdx.x * blockDim.x + threadIdx.x;
    const bool flag = (sample < num_samples && sample < order_size && order[sample] < training_size);

    if (flag)
        net[l].delta[node * num_samples + sample] *= d_activation(net[l].value[node * num_samples + sample]);

    __syncthreads();
    float temp = reduce(flag ? net[l].delta[node * num_samples + sample] : 0.0f, flag);

    if (flag)
        if (threadIdx.x == 0)
            atomicAdd(&net[l].grad_b[node], temp);
}
cudaError_t backward_propagate(d_NN &d_net, NN &net, const int &D, const float *training_out, const std::size_t &training_size, const int *order, const std::size_t order_size, const std::size_t &batch_size)
{
    compare_outputs<<<dim3((batch_size + SAMPLES - 1) / SAMPLES, net[D + 1].size()), SAMPLES>>>(d_net.head, D, training_out, training_size, order, order_size, batch_size);

    for (int l = D; l >= 1; l--)
    {
        CC(matmul(d_net.shells[l].delta, d_net.shells[l].weight, d_net.shells[l + 1].delta, net[l].size(), batch_size, net[l + 1].size(), "backward"));
        update_grad_b<<<dim3((batch_size + SAMPLES - 1) / SAMPLES, net[l].size()), SAMPLES>>>(d_net.head, l, training_size, order, order_size, batch_size);
    }

    return cudaGetLastError();
}

__global__ void GD_weight(NN net, const int l, const float LR, const float pow_beta1_t, const float pow_beta2_t, const float beta1 = 0.9f, const float beta2 = 0.99f)
{
    const int node = blockIdx.y, dest = blockIdx.x * blockDim.x + threadIdx.x;
    const bool flag = (dest < net[l + 1].size() && node < net[l].size());

    if (flag)
    {
        net[l].m_w[node * net[l + 1].size() + dest] = beta1 * net[l].m_w[node * net[l + 1].size() + dest] + (1 - beta1) * net[l].grad_w[node * net[l + 1].size() + dest];
        net[l].v_w[node * net[l + 1].size() + dest] = beta2 * net[l].v_w[node * net[l + 1].size() + dest] + (1 - beta2) * net[l].grad_w[node * net[l + 1].size() + dest] * net[l].grad_w[node * net[l + 1].size() + dest];

        net[l].weight[node * net[l + 1].size() + dest] -= LR * (net[l].m_w[node * net[l + 1].size() + dest] / (1 - pow_beta1_t)) / (1e-6f + std::sqrt(net[l].v_w[node * net[l + 1].size() + dest] / (1 - pow_beta2_t)));
        net[l].grad_w[node * net[l + 1].size() + dest] = 0.0f;
    }
}
__global__ void GD_bias(NN net, const int l, const float LR, const float pow_beta1_t, const float pow_beta2_t, const float beta1 = 0.9f, const float beta2 = 0.99f)
{
    const int node = blockIdx.x * blockDim.x + threadIdx.x;
    const bool flag = (node < net[l].size());

    if (flag)
    {
        net[l].m_b[node] = beta1 * net[l].m_b[node] + (1 - beta1) * net[l].grad_b[node];
        net[l].v_b[node] = beta2 * net[l].v_b[node] + (1 - beta2) * net[l].grad_b[node] * net[l].grad_b[node];

        net[l].bias[node] -= LR * (net[l].m_b[node] / (1 - pow_beta1_t)) / (1e-6f + std::sqrt(net[l].v_b[node] / (1 - pow_beta2_t)));
        net[l].grad_b[node] = 0.0f;
    }
}
const float beta1 = 0.9f, beta2 = 0.99f;
float pow_beta1_t = 1.0f, pow_beta2_t = 1.0f;
cudaError_t gradient_descent(d_NN &d_net, NN &net, const int &D, const float &LR, const std::size_t &batch_size)
{
    pow_beta1_t *= beta1;
    pow_beta2_t *= beta2;

    for (int l = 0; l <= D; l++)
    {
        CC(matmul(d_net.shells[l].grad_w, d_net.shells[l].value, d_net.shells[l + 1].delta, net[l].size(), net[l + 1].size(), batch_size, "gradient"));
        GD_weight<<<dim3((net[l + 1].size() + SAMPLES - 1) / SAMPLES, net[l].size()), SAMPLES>>>(d_net.head, l, LR, pow_beta1_t, pow_beta2_t, beta1, beta2);
        if (l != 0)
            GD_bias<<<(net[l].size() + SAMPLES - 1) / SAMPLES, SAMPLES>>>(d_net.head, l, LR, pow_beta1_t, pow_beta2_t, beta1, beta2);
    }
    GD_bias<<<(net[D + 1].size() + SAMPLES - 1) / SAMPLES, SAMPLES>>>(d_net.head, D + 1, LR, pow_beta1_t, pow_beta2_t, beta1, beta2);

    return cudaGetLastError();
}

std::pair<int *, int *> range(const std::size_t &n)
{
    int *order, *d_order;
    CC(cudaMallocHost(&order, n * sizeof(int)));
    std::iota(order, order + n, 0);
    CC(cudaMalloc(&d_order, n * sizeof(int)));
    CC(cudaMemcpy(d_order, order, n * sizeof(int), cudaMemcpyHostToDevice));

    return {order, d_order};
}

float get_loss(d_NN &d_net, NN &host_net, const int D, const int N, const float *d_data_in, const float *d_data_out, const std::size_t data_size, std::size_t batch_size)
{
    std::pair<int *, int *> orders = range(data_size);

    float total_loss, *d_total_loss;
    CC(cudaMalloc(&d_total_loss, sizeof(float)));
    CC(cudaMemset(d_total_loss, 0.0f, sizeof(float)));

    int b;
    std::size_t order_size = batch_size;
    for (b = 0; b < data_size; b += batch_size)
    {
        int *order = orders.second + b;

        if (b + batch_size > data_size)
            order_size = data_size - b;

        CC(forward_propagate(d_net, host_net, D, d_data_in, data_size, order, order_size, batch_size));
        loss<<<dim3((batch_size + SAMPLES - 1) / SAMPLES, host_net[D + 1].size()), SAMPLES>>>(d_total_loss, d_net.head, D, d_data_out, data_size, order, order_size, batch_size);
        CC(cudaGetLastError());
    }

    CC(cudaFreeHost(orders.first));
    CC(cudaFree(orders.second));

    CC(cudaMemcpy(&total_loss, d_total_loss, sizeof(float), cudaMemcpyDeviceToHost));
    CC(cudaFree(d_total_loss));

    return total_loss / data_size;
}

void train(d_NN &d_net, NN &net, const int &D, const int &N, const float *training_in, const float *training_out, const std::size_t &training_size, float &LR, std::mt19937 &gen, const std::size_t &batch_size = 32, const int &print = 5000)
{
    float *d_training_in = nullptr, *d_training_out = nullptr;
    dataset_to_device(training_in, training_out, d_training_in, d_training_out, training_size, net[0].size(), net[D + 1].size());
    std::pair<int *, int *> orders = range(training_size);

    float last_loss = get_loss(d_net, net, D, N, d_training_in, d_training_out, training_size, batch_size), curr_loss;
    auto start = std::chrono::high_resolution_clock::now(), last_time = std::chrono::high_resolution_clock::now(), curr_time = std::chrono::high_resolution_clock::now();

    std::cout << "Training neural network on " << training_size << " data points. Initial loss: " << last_loss << "\n\n";

    dim3 propGrid(N, batch_size);

    int epochs;
    while (true)
    {
        std::cout << "How many epochs to train the NN for (0 to stop): ";
        std::cin >> epochs;
        if (epochs <= 0)
            break;

        start = std::chrono::high_resolution_clock::now();
        last_time = std::chrono::high_resolution_clock::now();
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(curr_time - last_time);
        int epochs_at_last_print = epochs;

        while (epochs--)
        {
            if (epochs % 25 == 0)
            {
                curr_loss = get_loss(d_net, net, D, N, d_training_in, d_training_out, training_size, batch_size);
                if (curr_loss > 1.025f * last_loss)
                    LR *= 0.975f; // If the loss increased, the training is unstable. Reduce the learning rate.
                last_loss = curr_loss;
            }

            if (print != 0) // The program should print interim status reports every print milliseconds{
            {
                curr_time = std::chrono::high_resolution_clock::now();
                ms = std::chrono::duration_cast<std::chrono::milliseconds>(curr_time - last_time);
                if (ms.count() >= print)
                {
                    const auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(curr_time - start).count();
                    const auto estimated_total_ms = elapsed_ms + (ms.count() * epochs) / (epochs_at_last_print - epochs);
                    std::cout << "   Elapsed: " << elapsed_ms / 1000 << "s / " << estimated_total_ms / 1000 << "s   |   Epochs remaining: " << epochs << "    |    LR: " << LR << "   |   Loss: " << last_loss << "\n";
                    last_time = curr_time;
                    epochs_at_last_print = epochs;
                }
            }

            std::shuffle(orders.first, orders.first + training_size, gen); // Shuffle the order of the training data for this epoch
            CC(cudaMemcpy(orders.second, orders.first, training_size * sizeof(int), cudaMemcpyHostToDevice));

            std::size_t order_size = batch_size;
            for (int i = 0; i < training_size; i += batch_size)
            {
                int *order = orders.second + i;
                if (i + batch_size > training_size)
                    order_size = training_size - i;

                CC(forward_propagate(d_net, net, D, d_training_in, training_size, order, order_size, batch_size));

                CC(backward_propagate(d_net, net, D, d_training_out, training_size, order, order_size, batch_size));

                CC(gradient_descent(d_net, net, D, LR, batch_size));
            }
        }

        ms = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::high_resolution_clock::now() - start);
        std::cout << "Training complete. Elapsed time: " << ms.count() / 1000 << "s; Loss: " << last_loss << "\n\n";
    }

    CC(cudaFreeHost(orders.first));
    CC(cudaFree(orders.second));

    CC(cudaFree(d_training_in));
    CC(cudaFree(d_training_out));
}

void inference(d_NN &d_net, NN &host_net, const int D, const int N, const float *testing_in, const std::size_t &testing_size, std::size_t batch_size, const std::size_t &input_size, std::ofstream &output_file)
{
    std::cout << "Making predictions on data from input file... Saving to output file...\n\n";

    float *d_in = nullptr, *d_out = nullptr; // d_out should remain nullptr
    dataset_to_device(testing_in, nullptr, d_in, d_out, testing_size, host_net[0].size(), 0);
    std::pair<int *, int *> orders = range(testing_size);

    dim3 propGrid(N, batch_size);

    int b;
    std::size_t order_size = batch_size;
    for (b = 0; b < testing_size; b += batch_size)
    {
        int *order = orders.second + b;
        if (b + batch_size > testing_size)
            order_size = testing_size - b;

        CC(forward_propagate(d_net, host_net, D, d_in, testing_size, order, order_size, batch_size));

        CC(cudaMemcpy(host_net[D + 1].value, d_net.shells[D + 1].value, batch_size * host_net[D + 1].size() * sizeof(float), cudaMemcpyDeviceToHost));

        for (int i = 0; i < batch_size && b + i < testing_size; i++)
        {
            for (int j = 0; j < host_net[D + 1].size(); j++)
                output_file << host_net[D + 1].value[j * batch_size + i] << " ";
            output_file << "\n";
        }
    }

    CC(cudaFreeHost(orders.first));
    CC(cudaFree(orders.second));

    CC(cudaFree(d_in));
}

void build_NN(NN &net, const int &D, const int &N, const std::size_t &input_size, const std::size_t &output_size, std::uniform_real_distribution<float> &distribution, std::mt19937 &gen, const std::size_t &batch_size)
{
    if (D <= 0)
    {
        std::cerr << "Error: can't make such a network. D > 0 required\n";
        exit(-1);
    }

    CC(cudaMallocHost(&net, (D + 2) * sizeof(layer)));

    new (&net[0]) layer(input_size, N, distribution, gen, batch_size); // NN[0]  -  input layer
    for (int i = 1; i < D; i++)
        new (&net[i]) layer(N, N, distribution, gen, batch_size);           // NN[1 ... D-1]  -  hidden layers
    new (&net[D]) layer(N, output_size, distribution, gen, batch_size);     // NN[D]  -  last hidden layer
    new (&net[D + 1]) layer(output_size, 0, distribution, gen, batch_size); // NN[D+1]  -  output layer
}

void analyzeDataFile(std::ifstream &data_file, std::size_t &input_size, std::size_t &output_size, std::size_t &lines)
{
    input_size = 0;
    output_size = 0;

    std::string line1, line2;
    std::getline(data_file, line1);
    std::getline(data_file, line2);
    std::istringstream in(line1), out(line2);

    float temp;
    while (in >> temp)
        input_size++;
    while (out >> temp)
        output_size++;

    lines = 1;
    while (std::getline(data_file, line1) && std::getline(data_file, line2))
        lines++;

    data_file.clear();
    data_file.seekg(0);
}
void analyzeInputFile(std::ifstream &inference_file, std::size_t &inference_size)
{
    std::string line;

    inference_size = 0;
    while (std::getline(inference_file, line))
        inference_size++;

    inference_file.clear();
    inference_file.seekg(0);
}

void read(const std::string &line, const std::size_t &line_size, float *v, const int &sample, const std::size_t &batch_size)
{
    std::istringstream in(line);
    for (int i = 0; i < line_size && in >> v[i * batch_size + sample]; i++);
}
void get_training_data(std::ifstream &data_file, const int &data_size, float *&data_in, float *&data_out, const std::size_t &input_size, const std::size_t &output_size)
{
    std::string line1, line2;
    data_in = (float *)malloc(input_size * data_size * sizeof(float));
    data_out = (float *)malloc(output_size * data_size * sizeof(float));
    for (int sample = 0; sample < data_size && std::getline(data_file, line1) && std::getline(data_file, line2); sample++)
    {
        read(line1, input_size, data_in, sample, data_size);
        read(line2, output_size, data_out, sample, data_size);
    }
    data_file.clear();
    data_file.seekg(0);
}
void get_inference_data(std::ifstream &inference_file, const int &inference_size, float *&inference_in, const std::size_t &input_size)
{
    std::string line;
    inference_in = (float *)malloc(inference_size * input_size * sizeof(float));
    for (int sample = 0; sample < inference_size && std::getline(inference_file, line); sample++)
        read(line, input_size, inference_in, sample, inference_size);
    inference_file.clear();
    inference_file.seekg(0);
}

void split_data(const float *data_in, const float *data_out, const std::size_t &total_size, float *&training_in, float *&training_out, std::size_t &training_size, float *&testing_in, float *&testing_out, std::size_t &testing_size, const std::size_t &batch_size, const std::size_t &input_size, const std::size_t &output_size, const float &part_testing = 0.2f)
{
    testing_size = total_size * part_testing;
    while ((total_size - testing_size) % batch_size != 0)
        testing_size++; // Make sure the training data can be split into batches evenly
    training_size = total_size - testing_size;

    testing_in = (float *)malloc(testing_size * input_size * sizeof(float));
    testing_out = (float *)malloc(testing_size * output_size * sizeof(float));
    training_in = (float *)malloc(training_size * input_size * sizeof(float));
    training_out = (float *)malloc(training_size * output_size * sizeof(float));

    for (std::size_t f = 0; f < input_size; f++)
    {
        memcpy(testing_in + f * testing_size, data_in + f * total_size, testing_size * sizeof(float));
        memcpy(training_in + f * training_size, data_in + f * total_size + testing_size, training_size * sizeof(float));
    }
    for (std::size_t f = 0; f < output_size; f++)
    {
        memcpy(testing_out + f * testing_size, data_out + f * total_size, testing_size * sizeof(float));
        memcpy(training_out + f * training_size, data_out + f * total_size + testing_size, training_size * sizeof(float));
    }

    std::cout << "\nSplitting data...\n"
              << " Testing set: " << testing_size << " / " << total_size << " (" << 100 * testing_size / total_size << "%)\n"
              << " Training set: " << training_size / batch_size << 'x' << batch_size << " / " << total_size << " (" << 100 * training_size / total_size << "%)\n\n\n";
}

void load_network(NN &net, int &D, int &N, std::size_t &input_size, std::size_t &output_size, float &LR, std::ifstream &load_file, const std::size_t &batch_size)
{
    std::cout << "Loading neural network from file...\n";

    std::string header;
    std::getline(load_file, header);
    std::istringstream in(header);
    in >> D;
    in >> N;
    in >> input_size;
    in >> output_size;
    in >> LR;

    if (D <= 0)
    {
        std::cerr << "Error: can't load network. D!>0\n";
        exit(-1);
    }

    CC(cudaMallocHost(&net, (D + 2) * sizeof(layer)));
    std::string *lines = new std::string[std::max((std::size_t)N, std::max(input_size, output_size))];

    for (int i = 0; i < input_size; i++)
        std::getline(load_file, lines[i]);
    new (&net[0]) layer(input_size, N, lines, batch_size); // Input layer

    for (int i = 1; i < D; i++)
    {
        for (int j = 0; j < N; j++)
            std::getline(load_file, lines[j]);
        new (&net[i]) layer(N, N, lines, batch_size); // Hidden layers
    }

    for (int i = 0; i < N; i++)
        std::getline(load_file, lines[i]);
    new (&net[D]) layer(N, output_size, lines, batch_size); // Last hidden layer

    for (int i = 0; i < output_size; i++)
        std::getline(load_file, lines[i]);
    new (&net[D + 1]) layer(output_size, 0, lines, batch_size); // Output layer

    std::cout << "Network structure: " << D << " hidden layers, " << N << " nodes per layer.  |  Input layer: " << input_size << " nodes, output layer: " << output_size << " nodes.  |  Saved LR: " << LR << "\n\n";
}

void save_network(const NN &net, const int &D, const int &N, const std::size_t &input_size, const std::size_t &output_size, const float &LR, std::ofstream &save_file)
{
    std::cout << "Saving NN to file...\n";
    save_file << D << ' ' << N << ' ' << input_size << ' ' << output_size << ' ' << LR << "\n"; // Save the network structure

    for (int l = 0; l <= D + 1; l++)
        for (int i = 0; i < net[l].size(); i++)
        {
            for (int j = 0; j < net[l].output_size; j++)
                save_file << net[l].weight[i * net[l].output_size + j] << ' ';
            save_file << net[l].bias[i] << "\n";
        }
}

int main(int argc, char **argv)
{
    auto start = std::chrono::system_clock::now();
    std::cout << "\n\n\n";

    cublasCreate(&handle);

    int D, N;
    std::size_t input_size, output_size, batch_size = (std::size_t)-1;
    float part_testing = 0.2f, LR = -0.7734f;

    int whereToSave = 0, print = 5000;
    std::ifstream data_file, load_file, inference_file;
    std::ofstream output_file, save_file;

    initialize(argc, argv, data_file, load_file, inference_file, output_file, whereToSave, batch_size, part_testing, LR, print);

    float *data_in = nullptr, *data_out = nullptr, *training_in = nullptr, *training_out = nullptr, *testing_in = nullptr, *testing_out = nullptr, *inferencing_in = nullptr;
    std::size_t data_size, training_size, testing_size, inferencing_size;

    // Get Input/output format of the data file and the number of data points in it.
    if (data_file.is_open())
    {
        analyzeDataFile(data_file, input_size, output_size, data_size);
        get_training_data(data_file, data_size, data_in, data_out, input_size, output_size);
        if (batch_size == (std::size_t)-1)
            batch_size = data_size * (1.0f - part_testing) / 4;
        if (!(batch_size > 0 && batch_size <= data_size * (1.0f - part_testing)))
            batch_size = data_size * (1.0f - part_testing);
    }

    NN net;

    // Load from file / create the neural network
    std::mt19937 RNG(std::chrono::system_clock::to_time_t(start)); // Random number generator for generating initial weight and biases and for shuffling the training data
    if (load_file.is_open())
        load_network(net, D, N, input_size, output_size, LR, load_file, batch_size);
    else if (data_file.is_open())
    {
        std::cout << "Creating neural network... Define its size <depth D> <layer width N>: ";
        std::cin >> D >> N;

        int temp = ((N + 31) / 32) * 32;
        if (temp != N)
        {
            std::string command;
            std::cout << " Note: The GPU parallelizes 32 threads at a time (warps). Round N up to " << temp << "? [Y/n]  ";
            std::getline(std::cin, command); // Eats the leftover newline from std::cin >> D >> N;
            std::getline(std::cin, command);
            if (command.empty() || command[0] == 'Y' || command[0] == 'y')
                N = temp;
            std::cout << '\n';
        }

        float bound = (N > 0) ? 1.0f / sqrt(N) : 0.0f;
        std::uniform_real_distribution<float> random_real(-1.0f * bound, +1.0f * bound);
        build_NN(net, D, N, input_size, output_size, random_real, RNG, batch_size);
    }
    else
    {
        std::cerr << "Error: no data file or load file specified. Failed making an NN.\n";
        exit(-1);
    }

    if (LR == -0.7734f) // If uninitialized
        LR = 0.1f / D;  // Both build_NN and load_network ensure that D>0

    d_NN d_net;
    NN_to_device(d_net, net, D, batch_size);

    cudaGetLastError(); // For some reason, CUDA sometimes has a harmless error on launch. Here I clear it before any of the actual CUDA code below.

    bool did_something = false;
    // Train the NN on data from data file
    if (data_file.is_open())
    {
        // std::shuffle(data.begin(), data.end(), RNG);  // Commented to keep the testing set consistent between attempts. Uncomment to enable better initial randomization.
        split_data(data_in, data_out, data_size, training_in, training_out, training_size, testing_in, testing_out, testing_size, batch_size, input_size, output_size, part_testing);

        std::cout << std::setprecision(4) << std::scientific;
        train(d_net, net, D, N, training_in, training_out, training_size, LR, RNG, batch_size, print);
        did_something = true;

        float *d_testing_in = nullptr, *d_testing_out = nullptr;
        dataset_to_device(testing_in, testing_out, d_testing_in, d_testing_out, testing_size, input_size, output_size);
        std::cout << "\n\nTraining Complete!\nLoss on testing dataset: " << get_loss(d_net, net, D, N, d_testing_in, d_testing_out, testing_size, batch_size) << "\n\n\n";
        CC(cudaFree(d_testing_in));
        CC(cudaFree(d_testing_out));

        NN_from_device(net, d_net, D);
    }

    // Inference on data from input file
    if (inference_file.is_open())
    {
        analyzeInputFile(inference_file, inferencing_size);
        get_inference_data(inference_file, inferencing_size, inferencing_in, input_size);

        if (!output_file.is_open())
            output_file.open("out.txt");

        inference(d_net, net, D, N, inferencing_in, inferencing_size, batch_size, input_size, output_file);
    }
    else
        std::cout << "No input file specified. Skipping inference.\n\n";
    std::cout << '\n';

    // Save the trained NN to a file
    if (whereToSave != 0)
        save_file.open(argv[whereToSave]);
    else if (did_something)
        save_file.open("trained_net.txt");
    if (save_file.is_open())
        save_network(net, D, N, input_size, output_size, LR, save_file);
    else if (did_something)
        std::cerr << "Warning: save file inaccessible. NN not saved.\n";

    free_d_NN(d_net, D);
    free_NN(net, D);
    cublasDestroy(handle);

    if (CUDA_errorlog.is_open())
    {
        std::cerr << "\nCUDA errors occurred; details were saved to CUDA_errorlog.txt...\n";
        CUDA_errorlog.close();
    }

    data_file.close();
    load_file.close();
    inference_file.close();
    output_file.close();
    save_file.close();

    std::cout << "\nProgram Time: " << std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now() - start).count() << "s\n\n\n";
    return 0;
}
