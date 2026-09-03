/*  Author: Nikola D. Lilov, 02 Sep 2026
 *
 *  Desc: Trains a simple NxM neural network with decreasing LR, a squared
 *  loss function and the x/1+abs(x) activation function. Instead of propagating
 *  one training example at a time, the GPU is fed `parallel_samples` at once.
 *  Requires a training dataset file with alternating input and output
 *  lines. Saves (and can load) the trained net to a file to later
 *  retrain or inference on.
 *
 *  Compilation eg.:    nvcc CUDA-trainNN.cu -o CUDA -O2
 *                      nvcc CUDA-trainNN.cu -o CUDA -O3 -arch=sm_XX
 *
 *  Common Usage:       train.exe -d dataset.txt
 *  Expanded Usage:     train.exe -h
 */

/* The data should be in the format:

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
#include <ctime>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime_api.h>
#include <algorithm>
#include <numeric>

std::ofstream CUDA_errorlog;
void report_cuda_error(const char *file, int line, cudaError_t result)
{
    std::time_t timestamp = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    char time_buffer[32];
    std::strftime(time_buffer, sizeof(time_buffer), "%Y-%m-%d %H:%M:%S", std::localtime(&timestamp));

    if (!CUDA_errorlog.is_open())
        CUDA_errorlog.open("CUDA_errorlog.txt", std::ios::out | std::ios::trunc);

    if (CUDA_errorlog.is_open())
        CUDA_errorlog << '[' << time_buffer << "] CUDA Runtime Error: " << file << ':' << line << ':' << result << " = " << cudaGetErrorString(result) << '\n';
    else
        std::fprintf(stderr, "CUDA Runtime Error: %s:%d = %s\n", file, line, cudaGetErrorString(result));
}
#define CC(expr_to_check)                                  \
    do                                                     \
    {                                                      \
        cudaError_t result = expr_to_check;                \
        if (result != cudaSuccess)                         \
            report_cuda_error(__FILE__, __LINE__, result); \
    } while (false)

void help(const std::string &exe)
{
    std::cerr << " Common Usages:\n"
              << "   " << exe << " -d dataset.txt\n"
              << "   " << exe << " -l trained_net.txt -i input.txt \n"
              << "\n"
              << " All Options: (order isn't important)\n"
              << "   -d --data <path>     Training data file containing space-separated input and output lines.\n"
              << "   -l --load <path>     Load a trained model from file to inference* with.\n"
              << "   -s --save <path>     Save the trained model to file. Defaults to trained_net.txt.\n"
              << "   -t --testing <float> Set what proportion of the dataset (-d) should be for testing. Defaults to 0.20.\n"
              << "   -b --batch_size <int>         Set how many samples should be propagated in parallel before evolving (0 <=> BGD). Defaults to 500.\n"
              << "   -lr --learning_rate <float>   Set the initial learning rate for training. Defaults to 0.5/D.\n"
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

void initialize(int &argc, char **argv, std::ifstream &data_file, std::ifstream &load_file, std::ifstream &input_file,
                std::ofstream &output_file, int &whereToSave, std::size_t &batch_size, float &part_testing, float &LR, int &print)
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
                input_file.open(argv[i]);
                if (!input_file.is_open())
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
            else if (arg == "--learning_rate" || arg == "-lr")
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
    float *value, *delta, *bias, *grad_b, *weight_f, *weight_b, *grad_w;
    // Every sample has its node.value (2D array) and its delta=dL/dv (2D array).
    // Every node has a bias (1D array).
    // Every node has a weight to every next node (2D array):
    // To coalesce for the backward pass, the weights in weight_f (weights going forwards) are to be accessed as weight[node][next_node] = weight[node * output_size + next_node].
    // To coalesce for the forward pass, the weights in weight_b (weights going backwards) are to be accessed as weight[node][prev_node] = weight[node * input_size + prev_node].

    layer() = default;

    layer(const std::size_t &in_size, const std::size_t &batch_size) // Used only as an expected output to compare against => no logic stored inside
    {
        input_size = in_size;
        output_size = 0;
        value = (float *)malloc(batch_size * input_size * sizeof(float));
    }

    layer(const std::size_t &in_size, const std::size_t &out_size, const std::string *load, const std::size_t &batch_size) // Used only for loading a ready-made network. Assumes load_size=in.
    {
        input_size = in_size;
        output_size = out_size;

        value = (float *)malloc(batch_size * input_size * sizeof(float));
        memset(value, 0.0f, batch_size * input_size * sizeof(float));
        delta = (float *)malloc(batch_size * input_size * sizeof(float));
        memset(delta, 0.0f, batch_size * input_size * sizeof(float));

        grad_w = (float *)malloc(input_size * output_size * sizeof(float));
        memset(grad_w, 0.0f, input_size * output_size * sizeof(float));
        grad_b = (float *)malloc(input_size * sizeof(float));
        memset(grad_b, 0.0f, input_size * sizeof(float));

        weight_f = (float *)malloc(input_size * output_size * sizeof(float));
        weight_b = (float *)malloc(input_size * output_size * sizeof(float));
        bias = (float *)malloc(input_size * sizeof(float));
        std::istringstream in;
        for (int j = 0; j < input_size; j++)
        {
            in.clear();
            in.str(load[j]);
            for (int k = 0; k < output_size; k++)
            {
                in >> weight_f[j * output_size + k];
                weight_b[k * input_size + j] = weight_f[j * output_size + k];
            }
            in >> bias[j];
        }
    }

    layer(const std::size_t &in_size, const std::size_t &out_size, std::uniform_real_distribution<float> &distribution, std::mt19937 &gen, const std::size_t &batch_size) // Normal Constructor
    {
        input_size = in_size;
        output_size = out_size;

        value = (float *)malloc(batch_size * input_size * sizeof(float));
        memset(value, 0.0f, batch_size * input_size * sizeof(float));
        delta = (float *)malloc(batch_size * input_size * sizeof(float));
        memset(delta, 0.0f, batch_size * input_size * sizeof(float));

        grad_w = (float *)malloc(input_size * output_size * sizeof(float));
        memset(grad_w, 0.0f, input_size * output_size * sizeof(float));
        grad_b = (float *)malloc(input_size * sizeof(float));
        memset(grad_b, 0.0f, input_size * sizeof(float));

        weight_f = (float *)malloc(input_size * output_size * sizeof(float));
        weight_b = (float *)malloc(input_size * output_size * sizeof(float));
        bias = (float *)malloc(input_size * sizeof(float));
        for (int i = 0; i < input_size; i++)
        {
            for (int j = 0; j < output_size; j++)
            {
                weight_f[i * output_size + j] = distribution(gen);
                weight_b[j * input_size + i] = weight_f[i * output_size + j];
            }
            bias[i] = distribution(gen);
        }
    }

    __host__ __device__ std::size_t size() const
    {
        return input_size;
    }

    ~layer()
    {
        if (onDevice)
        {
            cudaFreeHost(value);
            cudaFreeHost(delta);
            cudaFreeHost(bias);
            cudaFreeHost(grad_b);
            cudaFreeHost(weight_f);
            cudaFreeHost(weight_b);
            cudaFreeHost(grad_w);
        }
        else
        {
            free(value);
            free(delta);
            free(bias);
            free(grad_b);
            free(weight_f);
            free(weight_b);
            free(grad_w);
        }
    }
};
using NN = layer *;
struct d_NN
{
    NN head;
    NN shells;
};

// Using cudaMemcpy on just "net" would only make device-side copies of the pointers towards the arrays (which are host-side).
// Copying "net" to/from device is a two-step process. Same for datasets.
layer layer_to_device_shell(const layer &host_layer, std::size_t value_count)
{
    layer d;
    d.onDevice = true;
    d.input_size = host_layer.input_size;
    d.output_size = host_layer.output_size;

    CC(cudaMalloc(&d.value, value_count * sizeof(float)));
    CC(cudaMemcpy(d.value, host_layer.value, value_count * sizeof(float), cudaMemcpyHostToDevice));
    CC(cudaMalloc(&d.delta, value_count * sizeof(float)));
    CC(cudaMemcpy(d.delta, host_layer.delta, value_count * sizeof(float), cudaMemcpyHostToDevice));

    if (host_layer.bias != nullptr)
    {
        std::size_t wsize = host_layer.size() * host_layer.output_size * sizeof(float);
        std::size_t bsize = host_layer.size() * sizeof(float);

        CC(cudaMalloc(&d.bias, bsize));
        CC(cudaMemcpy(d.bias, host_layer.bias, bsize, cudaMemcpyHostToDevice));
        CC(cudaMalloc(&d.grad_b, bsize));
        CC(cudaMemcpy(d.grad_b, host_layer.grad_b, bsize, cudaMemcpyHostToDevice));
        CC(cudaMalloc(&d.weight_f, wsize));
        CC(cudaMemcpy(d.weight_f, host_layer.weight_f, wsize, cudaMemcpyHostToDevice));
        CC(cudaMalloc(&d.weight_b, wsize));
        CC(cudaMemcpy(d.weight_b, host_layer.weight_b, wsize, cudaMemcpyHostToDevice));
        CC(cudaMalloc(&d.grad_w, wsize));
        CC(cudaMemcpy(d.grad_w, host_layer.grad_w, wsize, cudaMemcpyHostToDevice));
    }
    else
        d.delta = d.bias = d.grad_b = d.weight_f = d.weight_b = d.grad_w = nullptr;

    return d;
}
d_NN NN_to_device(const NN &host_net, const int &D, const std::size_t &batch_size)
{
    d_NN d_net;
    d_net.shells = (NN)malloc((D + 2) * sizeof(layer));
    for (int i = 0; i < D + 2; i++)
    {
        std::size_t value_count = batch_size * host_net[i].size();
        d_net.shells[i] = layer_to_device_shell(host_net[i], value_count);
    }

    CC(cudaMalloc(&d_net.head, (D + 2) * sizeof(layer)));
    CC(cudaMemcpy(d_net.head, d_net.shells, (D + 2) * sizeof(layer), cudaMemcpyHostToDevice));
    return d_net;
}
void NN_from_device(NN host_net, const d_NN &d_net, const int &D)
{
    for (int i = 0; i < D + 2; i++)
    {
        std::size_t wsize = host_net[i].input_size * host_net[i].output_size * sizeof(float);
        std::size_t bsize = host_net[i].input_size * sizeof(float);
        CC(cudaMemcpy(host_net[i].weight_f, d_net.shells[i].weight_f, wsize, cudaMemcpyDeviceToHost));
        CC(cudaMemcpy(host_net[i].weight_b, d_net.shells[i].weight_b, wsize, cudaMemcpyDeviceToHost));
        CC(cudaMemcpy(host_net[i].bias, d_net.shells[i].bias, bsize, cudaMemcpyDeviceToHost));
    }
}
void free_NN(NN &host_net, const int &D)
{
    for (int i = 0; i < D + 2; ++i)
        host_net[i].~layer();
    cudaFreeHost(host_net);
}
void free_d_NN(d_NN &d_net, const int &D)
{
    for (int i = 0; i < D + 2; i++)
    {
        CC(cudaFree(d_net.shells[i].value));
        CC(cudaFree(d_net.shells[i].delta));
        CC(cudaFree(d_net.shells[i].bias));
        CC(cudaFree(d_net.shells[i].grad_b));
        CC(cudaFree(d_net.shells[i].weight_f));
        CC(cudaFree(d_net.shells[i].weight_b));
        CC(cudaFree(d_net.shells[i].grad_w));
    }
    free(d_net.shells);
    CC(cudaFree(d_net.head));
}

void dataset_to_device(const float *host_data_in, const float *host_data_out, float *&dev_data_in, float *&dev_data_out, const std::size_t n, const std::size_t input_size, const std::size_t output_size)
{
    CC(cudaMalloc(&dev_data_in, n * input_size * sizeof(float)));
    CC(cudaMemcpy(dev_data_in, host_data_in, n * input_size * sizeof(float), cudaMemcpyHostToDevice));
    CC(cudaMalloc(&dev_data_out, n * output_size * sizeof(float)));
    CC(cudaMemcpy(dev_data_out, host_data_out, n * output_size * sizeof(float), cudaMemcpyHostToDevice));
}

__device__ __forceinline__ unsigned int bitceil(unsigned int x)
{
    return 1u << (32 - __clz(x - 1));
}
__device__ float reduce(float val)
{
    extern __shared__ float new_vec[]; // [blockDim.x] (which is >= size)
    new_vec[threadIdx.x] = val;
    __syncthreads();

    for (int stride = bitceil(blockDim.x) / 2; stride > 0; stride >>= 1)
    {
        __syncthreads();
        if (threadIdx.x + stride < blockDim.x && threadIdx.x < stride)
            new_vec[threadIdx.x] += new_vec[threadIdx.x + stride];
    }

    __syncthreads();
    return new_vec[0];
}

__global__ void loss(float *ans, NN net, const int D, const float *data_out, const std::size_t data_size, const int *order)
{
    const int sample = blockIdx.x, node = threadIdx.x;

    if (order[sample] >= data_size)
        return;

    if (node < net[D + 1].size())
    {
        float diff = net[D + 1].value[sample * net[D + 1].size() + node] - data_out[order[sample] * net[D + 1].size() + node];
        atomicAdd(ans, diff * diff);
    }
}
__device__ float d_loss(const float &expected, const float &output)
{
    return 2 * (output - expected); // Derivative of the loss function with respect to the output
}
__device__ float activation(const float &x)
{
    return x / (1.0f + abs(x));
}
__device__ float d_activation(const float &y)
{
    float t = 1.0f - abs(y);
    return t * t;
}

__global__ void set_input(NN net, const float *training_in, const std::size_t training_size, const int *order)
{
    const int sample = blockIdx.x, node = threadIdx.x;

    if (order[sample] >= training_size)
        return;

    if (node < net[0].size())
        net[0].value[sample * net[0].size() + node] = training_in[order[sample] * net[0].size() + node];
}
__global__ void forward_propagate_layer(NN net, const int l, const std::size_t training_size, const int *order)
{
    const int sample = blockIdx.y, node = blockIdx.x, in = threadIdx.x;

    if (order[sample] >= training_size)
        return;

    if (node < net[l].size())
    {
        const int bid = sample * net[l].size() + node;
        net[l].value[bid] = reduce(in < net[l - 1].size() ? net[l - 1].weight_b[node * net[l - 1].size() + in] * net[l - 1].value[sample * net[l - 1].size() + in] : 0.0f);
        if (threadIdx.x == 0)
            net[l].value[bid] = activation(net[l].bias[node] + net[l].value[bid]);
    }
}

__global__ void compare_outputs(NN net, const int D, const float *training_out, const std::size_t training_size, const int *order)
{
    const int sample = blockIdx.x, node = threadIdx.x;

    if (order[sample] >= training_size)
        return;

    if (node < net[D + 1].size())
    {
        const int bid = sample * net[D + 1].size() + node;
        net[D + 1].delta[bid] = d_loss(training_out[order[sample] * net[D + 1].size() + node], net[D + 1].value[bid]) * d_activation(net[D + 1].value[bid]);
        atomicAdd(&net[D + 1].grad_b[node], net[D + 1].delta[bid]);
    }
}
__global__ void backward_propagate_layer(NN net, const int l, const std::size_t training_size, const int *order)
{
    const int sample = blockIdx.y, node = blockIdx.x, in = threadIdx.x;

    if (order[sample] >= training_size)
        return;

    if (node < net[l].size())
    {
        const int bid = sample * net[l].size() + node;
        net[l].delta[bid] = reduce(in < net[l + 1].size() ? net[l].weight_f[node * net[l + 1].size() + in] * net[l + 1].delta[sample * net[l + 1].size() + in] : 0.0f);
        if (in < net[l + 1].size())
            atomicAdd(&net[l].grad_w[node * net[l + 1].size() + in], net[l].value[bid] * net[l + 1].delta[sample * net[l + 1].size() + in]);
        if (threadIdx.x == 0)
        {
            net[l].delta[bid] *= d_activation(net[l].value[bid]);
            atomicAdd(&net[l].grad_b[node], net[l].delta[bid]);
        }
    }
}

__global__ void update(NN net, const int l, const float LR)
{
    const int node = blockIdx.x, in = threadIdx.x;
    if (node >= net[l].size())
        return;

    if (in == 0)
    {
        net[l].bias[node] -= LR * net[l].grad_b[node];
        net[l].grad_b[node] = 0.0f;
    }
    if (in < net[l + 1].size())
    {
        const int idx = node * net[l + 1].size() + in;
        net[l].weight_f[idx] -= LR * net[l].grad_w[idx];
        net[l].grad_w[idx] = 0.0f;
    }
}
__global__ void update_last_bias(NN net, const int l, const float LR)
{
    const int node = threadIdx.x;
    if (node >= net[l].size())
        return;

    net[l].bias[node] -= LR * net[l].grad_b[node];
    net[l].grad_b[node] = 0.0f;
}
#define TILE_DIM 32
__global__ void transpose_weights(NN net, const int l)
{
    __shared__ float tile[TILE_DIM][TILE_DIM + 1]; // +1 padding avoids shared-mem bank conflicts

    const std::size_t in_size = net[l].input_size;   // rows of weight_f
    const std::size_t out_size = net[l].output_size; // cols of weight_f

    int x = blockIdx.x * TILE_DIM + threadIdx.x; // column (out index) in weight_f
    int y = blockIdx.y * TILE_DIM + threadIdx.y; // row (in index) in weight_f
    if (x < out_size && y < in_size)
        tile[threadIdx.y][threadIdx.x] = net[l].weight_f[y * out_size + x];

    __syncthreads();

    int tx = blockIdx.y * TILE_DIM + threadIdx.x; // now an "in" index
    int ty = blockIdx.x * TILE_DIM + threadIdx.y; // now an "out" index
    if (tx < in_size && ty < out_size)
        net[l].weight_b[ty * in_size + tx] = tile[threadIdx.x][threadIdx.y];
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
    dim3 propGrid(N, batch_size);

    float total_loss, *d_total_loss;
    CC(cudaMalloc(&d_total_loss, sizeof(float)));
    CC(cudaMemset(d_total_loss, 0.0f, sizeof(float)));

    int b;
    for (b = 0; b < data_size; b += batch_size)
    {
        int *order = orders.second + b;

        set_input<<<batch_size, host_net[0].size()>>>(d_net.head, d_data_in, data_size, order);
        for (int i = 1; i <= D + 1; i++)
            forward_propagate_layer<<<propGrid, host_net[i - 1].size(), host_net[i - 1].size() * sizeof(float)>>>(d_net.head, i, data_size, order);
        loss<<<batch_size, host_net[D + 1].size()>>>(d_total_loss, d_net.head, D, d_data_out, data_size, order);
        CC(cudaGetLastError());

        CC(cudaDeviceSynchronize());
    }
    if (b != data_size)
    {
        batch_size = batch_size - (b - data_size);
        b = data_size - batch_size;

        propGrid = dim3(N, batch_size);
        int *order = orders.second + b;
        set_input<<<batch_size, host_net[0].size()>>>(d_net.head, d_data_in, data_size, order);
        for (int i = 1; i <= D + 1; i++)
            forward_propagate_layer<<<propGrid, host_net[i - 1].size(), host_net[i - 1].size() * sizeof(float)>>>(d_net.head, i, data_size, order);
        loss<<<batch_size, host_net[D + 1].size()>>>(d_total_loss, d_net.head, D, d_data_out, data_size, order);
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

    dim3 propGrid(N, batch_size), transBlock(TILE_DIM, TILE_DIM);

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
            if (epochs % 10 == 0)
            {
                curr_loss = get_loss(d_net, net, D, N, d_training_in, d_training_out, training_size, batch_size);
                if (curr_loss > 1.025f * last_loss)
                    LR *= 0.975f; // If the loss increased, the training is unstable. Reduce the learning rate.
                last_loss = curr_loss;
            }

            curr_time = std::chrono::high_resolution_clock::now();
            ms = std::chrono::duration_cast<std::chrono::milliseconds>(curr_time - last_time);
            if (ms.count() >= print)
            {
                const auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(curr_time - start).count();
                const auto estimated_total_ms = elapsed_ms + (ms.count() * epochs) / (epochs_at_last_print - epochs);
                std::cout << "   Elapsed: " << elapsed_ms / 1000 << "s / " << estimated_total_ms / 1000 << "s;   \tEpochs remaining: " << epochs << ";\t\tLR: " << LR << ";   \tLoss: " << last_loss << "\n";
                last_time = curr_time;
                epochs_at_last_print = epochs;
            }

            std::shuffle(orders.first, orders.first + training_size, gen); // Shuffle the order of the training data for this epoch
            CC(cudaMemcpy(orders.second, orders.first, training_size * sizeof(int), cudaMemcpyHostToDevice));

            for (int i = 0; i < training_size; i += batch_size)
            {
                int *order = orders.second + i;

                set_input<<<batch_size, net[0].size()>>>(d_net.head, d_training_in, training_size, order);
                for (int l = 1; l <= D + 1; l++)
                    forward_propagate_layer<<<propGrid, net[l - 1].size(), net[l - 1].size() * sizeof(float)>>>(d_net.head, l, training_size, order);
                CC(cudaGetLastError());

                compare_outputs<<<batch_size, net[D + 1].size()>>>(d_net.head, D, d_training_out, training_size, order);
                for (int l = D; l >= 0; l--)
                    backward_propagate_layer<<<propGrid, net[l + 1].size(), net[l + 1].size() * sizeof(float)>>>(d_net.head, l, training_size, order);
                CC(cudaGetLastError());

                for (int l = 0; l <= D; l++)
                {
                    update<<<net[l].size(), net[l + 1].size()>>>(d_net.head, l, LR / batch_size);
                    transpose_weights<<<dim3((net[l + 1].size() + TILE_DIM - 1) / TILE_DIM, (net[l].size() + TILE_DIM - 1) / TILE_DIM), transBlock>>>(d_net.head, l);
                }
                update_last_bias<<<net[D + 1].size()>>>(d_net.head, D + 1, LR / batch_size);
                CC(cudaGetLastError());
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
    std::cout << "Making predictions on data from input file... Saving to output file...\n";

    float *d_in = nullptr, *d_out = nullptr; // d_out should remain nullptr
    dataset_to_device(testing_in, nullptr, d_in, d_out, testing_size, host_net[0].size(), 0);
    std::pair<int *, int *> orders = range(testing_size);

    dim3 propGrid(N, batch_size);

    int b;
    for (b = 0; b < testing_size; b += batch_size)
    {
        int *order = orders.second + b;

        set_input<<<batch_size, host_net[0].size()>>>(d_net.head, d_in, testing_size, order);
        for (int i = 1; i <= D + 1; i++)
            forward_propagate_layer<<<propGrid, host_net[i - 1].size(), host_net[i - 1].size() * sizeof(float)>>>(d_net.head, i, testing_size, order);
        CC(cudaGetLastError());

        CC(cudaMemcpy(host_net[D + 1].value, d_net.shells[D + 1].value, batch_size * host_net[D + 1].size() * sizeof(float), cudaMemcpyDeviceToHost));

        for (int i = 0; i < batch_size && b + i < testing_size; i++)
        {
            for (int j = 0; j < host_net[D + 1].size(); j++)
                output_file << host_net[D + 1].value[i * host_net[D + 1].size() + j] << " ";
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
        std::cerr << "Error: can't make such a network. D > 0 required";
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

    std::string l1, l2;
    std::getline(data_file, l1);
    std::getline(data_file, l2);
    std::istringstream in(l1), out(l2);

    float temp;
    while (in >> temp)
        input_size++;
    while (out >> temp)
        output_size++;

    lines = 1;
    while (std::getline(data_file, l1) && std::getline(data_file, l2))
        lines++;

    data_file.clear();
    data_file.seekg(0);
}
void analyzeInputFile(std::ifstream &data_file, const std::size_t &input_size, std::size_t &lines)
{
    std::string line;

    lines = 0;
    while (std::getline(data_file, line))
        lines++;

    data_file.clear();
    data_file.seekg(0);
}

void read(const std::string &line, float *v, const std::size_t &size)
{
    std::istringstream in(line);
    float curr;
    for (int i = 0; in >> curr && i < size; i++)
        v[i] = curr;
}
void get_training_data(std::ifstream &data_file, const int &lines, float *&data_in, float *&data_out, const std::size_t &input_size, const std::size_t &output_size)
{
    std::string l1, l2;
    data_in = (float *)malloc(lines * input_size * sizeof(float));
    data_out = (float *)malloc(lines * output_size * sizeof(float));
    for (int i = 0; i < lines && std::getline(data_file, l1) && std::getline(data_file, l2); i++)
    {
        read(l1, data_in + i * input_size, input_size);
        read(l2, data_out + i * output_size, output_size);
    }
    data_file.clear();
    data_file.seekg(0);
}
void get_inference_data(std::ifstream &data_file, const int &lines, float *&data_in, const std::size_t &input_size)
{
    std::string line;
    CC(cudaMallocHost(&data_in, lines * input_size * sizeof(float)));
    for (int i = 0; i < lines && std::getline(data_file, line); i++)
        read(line, data_in + i * input_size, input_size);

    data_file.clear();
    data_file.seekg(0);
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

    memcpy(testing_in, data_in, testing_size * input_size * sizeof(float));
    memcpy(testing_out, data_out, testing_size * output_size * sizeof(float));
    memcpy(training_in, data_in + testing_size * input_size, training_size * input_size * sizeof(float));
    memcpy(training_out, data_out + testing_size * output_size, training_size * output_size * sizeof(float));

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
        std::cerr << "Error: can't load network. D!>0";
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

    std::cout << "Network structure: " << D << " hidden layers, " << N << " nodes per layer.  |  Input layer: " << input_size << " nodes, output layer: " << output_size << " nodes.  |  Saved LR: " << LR << "\n";
}

void save_network(const NN &net, const int &D, const int &N, const std::size_t &input_size, const std::size_t &output_size, const float &LR, std::ofstream &save_file)
{
    std::cout << "Saving NN to file...\n";
    save_file << D << ' ' << N << ' ' << input_size << ' ' << output_size << ' ' << LR << "\n"; // Save the network structure

    for (int l = 0; l <= D + 1; l++)
        for (int i = 0; i < net[l].size(); i++)
        {
            for (int j = 0; j < net[l].output_size; j++)
                save_file << net[l].weight_f[i * net[l].output_size + j] << ' ';
            save_file << net[l].bias[i] << "\n";
        }

    save_file.close();
}

int main(int argc, char **argv)
{
    auto start = std::chrono::system_clock::now();
    std::cout << "\n\n\n";

    int D, N;
    std::size_t input_size, output_size, batch_size = 500;
    float part_testing = 0.2f, LR = -0.7734f;

    int whereToSave = 0, print = 5000;
    std::ifstream data_file, load_file, input_file;
    std::ofstream output_file, save_file;

    initialize(argc, argv, data_file, load_file, input_file, output_file, whereToSave, batch_size, part_testing, LR, print);

    float *data_in = nullptr, *data_out = nullptr, *training_in = nullptr, *training_out = nullptr, *testing_in = nullptr, *testing_out = nullptr, *inferencing_in = nullptr;
    std::size_t data_size, training_size, testing_size, inferencing_size;
    if (data_file.is_open())
    {
        analyzeDataFile(data_file, input_size, output_size, data_size);
        get_training_data(data_file, data_size, data_in, data_out, input_size, output_size);
    }
    if (!(batch_size > 0 && batch_size <= data_size * (1.0f - part_testing)))
        batch_size = data_size * (1.0f - part_testing);

    NN net;

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
        LR = 0.5f / D;  // Both build_NN and load_network ensure that D>0

    d_NN d_net = NN_to_device(net, D, batch_size);

    cudaGetLastError(); // For some reason, CUDA sometimes has a harmless error on launch. Here I clear it before any of the actual CUDA code below.

    bool did_something = false;
    if (data_file.is_open())
    {
        // std::shuffle(data.begin(), data.end(), RNG);  // Commented to keep the testing set consistent between attempts. Uncomment to enable better initial randomization.
        split_data(data_in, data_out, data_size, training_in, training_out, training_size, testing_in, testing_out, testing_size, batch_size, input_size, output_size, part_testing);

        std::cout << std::setprecision(4) << std::scientific;
        train(d_net, net, D, N, training_in, training_out, training_size, LR, RNG, batch_size, print);
        did_something = true;

        float *d_testing_in = nullptr, *d_testing_out = nullptr;
        dataset_to_device(testing_in, testing_out, d_testing_in, d_testing_out, testing_size, input_size, output_size);
        std::cout << "\nTraining Complete!\nLoss on testing dataset: " << get_loss(d_net, net, D, N, d_testing_in, d_testing_out, testing_size, batch_size) << "\n\n";
        CC(cudaFree(d_testing_in));
        CC(cudaFree(d_testing_out));

        NN_from_device(net, d_net, D);
    }

    if (input_file.is_open())
    {
        if (!output_file.is_open())
            output_file.open("output.txt");

        analyzeInputFile(input_file, input_size, inferencing_size);
        get_inference_data(input_file, inferencing_size, inferencing_in, input_size);

        inference(d_net, net, D, N, inferencing_in, inferencing_size, batch_size, input_size, output_file);
    }
    else
        std::cout << "No input file specified. Skipping inference.\n\n";
    std::cout << '\n';

    // Close files
    data_file.close();
    load_file.close();
    input_file.close();
    output_file.close();

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

    if (CUDA_errorlog.is_open())
    {
        std::cerr << "\nCUDA errors occurred; details were saved to CUDA_errorlog.txt...\n";
        CUDA_errorlog.close();
    }

    std::cout << "\nProgram Time: " << std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now() - start).count() << "s\n\n\n";
    return 0;
}
