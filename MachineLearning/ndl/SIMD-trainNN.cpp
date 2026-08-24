/*  Author: Nikola D. Lilov, 18 Aug 2026
 *
 *  Desc: Trains a simple NxM neural network with decreasing LR, a SIMD-
 *  friendly activation function x/1+abs(x) and a squared loss function.
 *  Instead of propagating one training example at a time, the CPU is
 *  fed through `vec_width` examples at once: one example per SIMD lane.
 *  Requires a training dataset file with alternating input and output
 *  lines. Saves (and can load) the trained net to a file to later
 *  retrain or inference on.
 *
 *  Compilation eg.:    g++ SIMD-trainNN.cpp -o SIMD -O2 -march=native
 *                      clang++ SIMD-trainNN.cpp -o SIMD -O2 -march=native
 *
 *  Common Usage:       SIMD.exe -d dataset.txt
 *  Expanded Usage:     SIMD.exe -h
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
#include <iomanip>
#include <cmath>
#include <random>
#include <chrono>
#include <cstdlib>
// #include <simd>   // Uncomment in C++26 or higher.
#include <experimental/simd> // Comment in C++26 or higher.
#include <algorithm>
#include <string>
#include <vector>

// namespace simd = std::simd;  // See l36
namespace simd = std::experimental; // See l37
using simd_t = simd::native_simd<float>;
constexpr std::size_t vec_width = simd_t::size();

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
              << "   -b --batch_size <int>         Set how big the training batches** should be (0 <=> BGD). Defaults to 4.\n"
              << "   -lr --learning_rate <float>   Set the initial learning rate for training. Defaults to 0.5/N.\n"
              << "   -p --print <int>     How often to print status updates. Defaults to 250 (once every 250 epochs).\n"
              << "   -i --inputs <path>   Load inference input from file.\n"
              << "   -o --output <path>   Save inference output to file. Defaults to output.txt.\n"
              << "   -h --help            Show this usage information.\n"
              << "\n"
              << "  *Note: If both --load and --data are specified, the model will be loaded and then further trained on the data as needed.\n"
              << " **Note: This code utilizes the CPU's capability to train on multiple examples at a time (SIMD). Your inputted batch_size will then be multiplied by the number of samples your CPU can parallelize.\n"
              << "\n"
              << " Example --data file format:\n"
              << "   <input1> <input2> <input3> ... <inputX>\n   <output1> <output2> ... <outputY>\n   <input1> <input2> <input3> ... <inputX>\n   <output1> <output2> ... <outputY>\n   ..."
              << "\n\n\n";

    exit(0);
}

void initialize(int &argc, char **argv, std::ifstream &data_file, std::ifstream &load_file, std::ifstream &input_file,
                std::ofstream &output_file, int &whereToSave, int &batch_size, float &part_testing, float &LR, int &print)
{

    if (argc == 1)
        help(argv[0]);

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
                    std::cerr << "Missing value for --batches\n";
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
}

struct layer
{
    std::vector<simd_t> value, delta;

    std::vector<std::vector<simd_t>> weight, grad_w;
    std::vector<simd_t> bias, grad_b;

    layer() = default;

    layer(const int &M) // Used only as an expected output to compare against => no logic stored inside
    {
        value.resize(M);
        weight.resize(0);
        grad_w.resize(0);
        bias.resize(0);
        grad_b.resize(0);
        delta.resize(0);
    }

    layer(const int &M, const int &output_size, std::uniform_real_distribution<float> &distribution, std::mt19937 &gen)
    {
        value.resize(M);
        weight.resize(M);
        grad_w.resize(M);
        bias.resize(M);
        grad_b.resize(M);
        delta.resize(M);

        for (int i = 0; i < M; i++)
        {
            weight[i].resize(output_size);
            grad_w[i].resize(output_size);
            for (int j = 0; j < output_size; j++)
            {
                weight[i][j] = simd_t(distribution(gen));
                grad_w[i][j] = simd_t(0.0f);
            }
            bias[i] = simd_t(distribution(gen));
            grad_b[i] = simd_t(0.0f);
            // value[i]=simd_t(0.0f);
            // delta[i]=simd_t(0.0f);
        }
    }

    std::size_t size() const
    {
        return value.size();
    }
};
using NN = std::vector<layer>;

using dataset = std::vector<std::pair<std::vector<float>, std::vector<float>>>; // A dataset is a vector of pairs of training input and output layers

simd_t d_loss(const simd_t &expected, const simd_t &output)
{
    return 2 * (output - expected); // Derivative of the loss function with respect to the output
}
simd_t activation(const simd_t &x)
{
    return x / (simd_t(1.0f) + simd::abs(x)); // Sigmoid activation function (from -1 to 1)
}
simd_t d_activation(const simd_t &y)
{
    simd_t t = simd_t(1.0f) - simd::abs(y);
    return t * t;
}

void forward_propagation(NN &net, const int &N)
{
    for (int i = 1; i <= N + 1; i++)
        for (int j = 0; j < net[i].size(); j++)
        {
            net[i].value[j] = net[i].bias[j];
            for (int k = 0; k < net[i - 1].size(); k++)
                net[i].value[j] += net[i - 1].weight[k][j] * net[i - 1].value[k];
            net[i].value[j] = activation(net[i].value[j]);
        }
}

void backward_propagation(NN &net, const int &N, layer &expected_output)
{
    // Calculate the delta for the output layer
    for (int j = 0; j < net[N + 1].size(); j++)
    {
        net[N + 1].delta[j] = d_loss(expected_output.value[j], net[N + 1].value[j]) * d_activation(net[N + 1].value[j]);
        net[N + 1].grad_b[j] += net[N + 1].delta[j];
    }

    // Backward pass
    for (int i = N; i >= 0; i--)
        for (int j = 0; j < net[i].size(); j++)
        {
            net[i].delta[j] = simd_t(0.0f);
            for (int k = 0; k < net[i + 1].size(); k++)
            {
                net[i].delta[j] += net[i + 1].delta[k] * net[i].weight[j][k];
                net[i].grad_w[j][k] += net[i + 1].delta[k] * net[i].value[j];
            }
            net[i].delta[j] *= d_activation(net[i].value[j]);
            net[i].grad_b[j] += net[i].delta[j];
        }
}

void update(NN &net, const int &N, const float &LR, const int &batch_size)
{
    const float LR_ = LR / batch_size;
    for (int i = 0; i <= N + 1; i++)
        for (int j = 0; j < net[i].size(); j++)
        {
            net[i].bias[j] -= simd_t(LR_ * simd::reduce(net[i].grad_b[j]));
            net[i].grad_b[j] = simd_t(0.0f);
            for (int k = 0; k < net[i].weight[j].size(); k++)
            {
                net[i].weight[j][k] -= simd_t(LR_ * simd::reduce(net[i].grad_w[j][k]));
                net[i].grad_w[j][k] = simd_t(0.0f);
            }
        }
}

float get_loss(NN &net, const int &N, dataset &data)
{
    layer expected_output(net.back().size());
    int cnt;
    float total_loss = 0.0f;

    for (cnt = 0; cnt < data.size(); cnt += vec_width)
    {
        for (int lane = 0; lane < vec_width; lane++)
            for (int j = 0; j < std::max(data[cnt + lane].first.size(), data[cnt + lane].second.size()); j++)
            {
                if (j < data[cnt + lane].first.size())
                    net[0].value[j][lane] = data[cnt + lane].first[j];
                if (j < data[cnt + lane].second.size())
                    expected_output.value[j][lane] = data[cnt + lane].second[j];
            }
        forward_propagation(net, N);
        for (int j = 0; j < net[N + 1].size(); j++)
        {
            expected_output.value[j] -= net[N + 1].value[j];
            total_loss += simd::reduce(expected_output.value[j] * expected_output.value[j]);
        }
    }

    return (cnt == 0) ? 0 : total_loss / cnt;
}

void train(NN &net, const int &N, const int &M, const int &input_size, const int &output_size, dataset &training, float &LR, std::mt19937 &gen, const int &batch_size = 32, const int &print = 250)
{
    float last_loss = get_loss(net, N, training), curr_loss;
    auto start = std::chrono::high_resolution_clock::now(), last_time = std::chrono::high_resolution_clock::now(), curr_time = std::chrono::high_resolution_clock::now();
    std::cout << "Training neural network on " << training.size() << " data points. Initial loss: " << last_loss << "\n\n";
    layer expected_output(net.back().size());
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

        while (epochs--)
        {
            if (epochs != 0)
            {
                if (epochs % 10 == 0)
                {
                    curr_loss = get_loss(net, N, training);
                    if (curr_loss > 1.025f * last_loss)
                        LR *= 0.975f; // If the loss increased, the training is unstable. Reduce the learning rate.
                    last_loss = curr_loss;
                }
                if (epochs % print == 0)
                {
                    curr_time = std::chrono::high_resolution_clock::now();
                    ms = std::chrono::duration_cast<std::chrono::milliseconds>(curr_time - last_time);
                    std::cout << "   Time Left: " << epochs / print * ms.count() / 1000 << "s;   \tEpochs: " << epochs << ";\t\tLR: " << LR << ";   \tLoss: " << last_loss << "\n";
                    last_time = curr_time;
                }
            }

            std::shuffle(training.begin(), training.end(), gen); // Shuffle the order of the training data for this epoch
            for (int i = 0; i < training.size(); i += vec_width)
            {
                for (int lane = 0; lane < vec_width; lane++)
                    for (int j = 0; j < std::max(training[i + lane].first.size(), training[i + lane].second.size()); j++)
                    {
                        if (j < training[i + lane].first.size())
                            net[0].value[j][lane] = training[i + lane].first[j];
                        if (j < training[i + lane].second.size())
                            expected_output.value[j][lane] = training[i + lane].second[j];
                    }
                forward_propagation(net, N);
                backward_propagation(net, N, expected_output);
                if ((i + vec_width) % batch_size == 0) // Update the weight and biases after each batch
                    update(net, N, LR, batch_size);
            }
        }

        last_loss = get_loss(net, N, training);
        ms = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::high_resolution_clock::now() - start);
        std::cout << "Training complete. Elapsed time: " << ms.count() / 1000 << "s; Loss: " << last_loss << "\n\n";
    }
}

void build_NN(NN &net, const int &N, const int &M, const int &input_size, const int &output_size, std::uniform_real_distribution<float> &distribution, std::mt19937 &gen)
{
    if (N <= 0)
    {
        std::cerr << "Error: can't load network. N!>0";
        exit(-1);
    }

    net.reserve(N + 2);

    net.emplace_back(input_size, M, distribution, gen); // NN[0]  -  input layer
    for (int i = 1; i < N; i++)
        net.emplace_back(M, M, distribution, gen);       // NN[1 ... N-1]  -  hidden layers
    net.emplace_back(M, output_size, distribution, gen); // NN[N]  -  last hidden layer
    net.emplace_back(output_size, 0, distribution, gen); // NN[N+1]  -  output layer
}

void getIOsize(std::ifstream &data_file, int &input_size, int &output_size)
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

    data_file.clear();
    data_file.seekg(0);
}

void read(const std::string &line, std::vector<float> &v)
{
    std::istringstream in(line);
    float curr;
    for (int i = 0; in >> curr && i < v.size(); i++)
        v[i] = curr;
}
void read(const std::string (&line)[vec_width], layer &l)
{
    float curr;
    std::istringstream in;
    for (int lane = 0; lane < vec_width; lane++)
    {
        in.clear();
        in.str(line[lane]);
        for (int i = 0; in >> curr && i < l.size(); i++)
            l.value[i][lane] = curr;
    }
}

void get_data(std::ifstream &data_file, dataset &data, const int &input_size, const int &output_size)
{
    std::vector<float> in(input_size), out(output_size);
    std::string l1, l2;
    while (std::getline(data_file, l1) && std::getline(data_file, l2))
    {
        read(l1, in);
        read(l2, out);
        data.push_back({in, out});
    }
    data_file.clear();
    data_file.seekg(0);
}

void split_data(const dataset &data, dataset &training, dataset &testing, const int &batch_size, const float &part_testing = 0.2f)
{
    int testing_size = data.size() * part_testing;
    while ((data.size() - testing_size) % batch_size != 0)
        testing_size++; // Make sure the training data can be split into batches evenly
    for (int i = 0; i < testing_size; i++)
        testing.push_back(data[i]);
    while (testing.size() % vec_width != 0)
        testing.push_back(testing[0]);
    for (int i = testing_size; i < data.size(); i++)
        training.push_back(data[i]);
    std::cout << "\nSplitting data...\n"
              << " Testing set: " << testing_size << " / " << data.size() << " (" << 100 * testing.size() / data.size() << "%)\n"
              << " Training set: " << training.size() / batch_size << 'x' << batch_size << " / " << data.size() << " (" << 100 * training.size() / data.size() << "%)\n\n\n";
}

void inference(NN &net, const int &N, std::ifstream &input_file, std::ofstream &output_file)
{
    // Now we can use the trained NN to make predictions on new data
    std::cout << "Making predictions on data from input file... Saving to output file...\n";
    std::string line[vec_width];
    int laneI, laneO;
    while (true)
    {
        for (laneI = 0; laneI < vec_width; laneI++)
            if (!std::getline(input_file, line[laneI]))
                break;
        if (laneI == 0)
            break; // Nothing to do.

        read(line, net[0]);
        forward_propagation(net, N);

        for (laneO = 0; laneO < laneI; laneO++)
        {
            for (auto v : net[N + 1].value)
                output_file << v[laneO] << " ";
            output_file << "\n";
        }

        if (laneI != vec_width)
            break;
    }
}

void load_network(NN &net, int &N, int &M, int &input_size, int &output_size, float &LR, std::ifstream &load_file)
{
    std::cout << "Loading neural network from file...\n";
    std::string line;

    std::getline(load_file, line);
    std::istringstream in(line);
    in >> N;
    in >> M;
    in >> input_size;
    in >> output_size;
    in >> LR;
    net.resize(N + 2);

    if (N > 0)
    {
        net[0].value.resize(input_size);
        net[0].delta.resize(input_size);
        net[0].bias.resize(input_size);
        net[0].grad_b.resize(input_size);
        net[0].weight.resize(input_size);
        net[0].grad_w.resize(input_size);
        for (int j = 0; j < input_size; j++)
        {
            net[0].weight[j].resize(M);
            net[0].grad_w[j].resize(M);
        }
        for (int i = 1; i < N; i++)
        {
            net[i].value.resize(M);
            net[i].delta.resize(M);
            net[i].bias.resize(M);
            net[i].grad_b.resize(M);
            net[i].weight.resize(M);
            net[i].grad_w.resize(M);
            for (int j = 0; j < M; j++)
            {
                net[i].weight[j].resize(M);
                net[i].grad_w[j].resize(M);
            }
        }
        net[N].value.resize(M);
        net[N].delta.resize(M);
        net[N].bias.resize(M);
        net[N].grad_b.resize(M);
        net[N].weight.resize(M);
        net[N].grad_w.resize(M);
        for (int j = 0; j < M; j++)
        {
            net[N].weight[j].resize(output_size);
            net[N].grad_w[j].resize(output_size);
        }
        net[N + 1].value.resize(output_size);
        net[N + 1].delta.resize(output_size);
        net[N + 1].bias.resize(output_size);
        net[N + 1].grad_b.resize(output_size);
        net[N + 1].weight.resize(output_size);
        net[N + 1].grad_w.resize(output_size);
        for (int j = 0; j < output_size; j++)
        {
            net[N + 1].weight[j].resize(0);
            net[N + 1].grad_w[j].resize(0);
        }
    }
    else
    {
        std::cerr << "Error: can't load network. N!>0";
        exit(-1);
    }

    std::cout << "Network structure: " << N << " hidden layers, " << M << " nodes per layer.  |  Input layer: " << input_size << " nodes, output layer: " << output_size << " nodes.\n";

    float temp;
    // layer = 0
    for (int j = 0; j < input_size; j++)
    {
        std::getline(load_file, line);
        in.clear();
        in.str(line);
        for (int k = 0; k < M; k++)
        {
            in >> temp;
            net[0].weight[j][k] = simd_t(temp);
        }
        in >> temp;
        net[0].bias[j] = simd_t(temp);
    }

    // layer = 1..N-1
    for (int i = 1; i < N; i++)
        for (int j = 0; j < M; j++)
        {
            std::getline(load_file, line);
            in.clear();
            in.str(line);
            for (int k = 0; k < M; k++)
            {
                in >> temp;
                net[i].weight[j][k] = simd_t(temp);
            }
            in >> temp;
            net[i].bias[j] = simd_t(temp);
        }

    // layer = N
    for (int j = 0; j < M; j++)
    {
        std::getline(load_file, line);
        in.clear();
        in.str(line);
        for (int k = 0; k < output_size; k++)
        {
            in >> temp;
            net[N].weight[j][k] = simd_t(temp);
        }
        in >> temp;
        net[N].bias[j] = simd_t(temp);
    }

    // layer = N+1
    for (int j = 0; j < output_size; j++)
    {
        std::getline(load_file, line);
        in.clear();
        in.str(line);
        in >> temp;
        net[N + 1].bias[j] = simd_t(temp);
    }
}

void save_network(const NN &net, const int &N, const int &M, const int &input_size, const int &output_size, const float &LR, std::ofstream &save_file)
{
    std::cout << "Saving NN to file...\n";
    save_file << N << ' ' << M << ' ' << input_size << ' ' << output_size << ' ' << LR << "\n"; // Save the network structure
    save_file.precision(6);

    for (const layer &l : net)
        for (int i = 0; i < l.weight.size(); i++)
        {
            for (auto &w : l.weight[i])
                save_file << w[0] << ' ';
            save_file << l.bias[i][0] << "\n";
        }

    save_file.close();
}

int main(int argc, char **argv)
{
    auto start = std::chrono::high_resolution_clock::now();

    std::cout << "\n\n\n";

    int N, M, input_size, output_size;
    float part_testing = 0.2f, LR = -1234;

    int whereToSave = 0, batch_size = 4, print = 250;
    std::ifstream data_file, load_file, input_file;
    std::ofstream output_file, save_file;

    initialize(argc, argv, data_file, load_file, input_file, output_file, whereToSave, batch_size, part_testing, LR, print);

    dataset data, training, testing;
    if (data_file.is_open())
    {
        getIOsize(data_file, input_size, output_size);
        get_data(data_file, data, input_size, output_size);
    }

    NN net;

    std::mt19937 RNG(std::chrono::system_clock::to_time_t(start)); // Random number generator for generating initial weight and biases and for shuffling the training data
    float bound;

    if (load_file.is_open())
        load_network(net, N, M, input_size, output_size, LR, load_file);
    else if (data_file.is_open())
    {
        std::cout << "Creating neural network... Define its size <depth N> <layer width M>: ";
        std::cin >> N >> M;
        bound = (M > 0) ? 1.0f / sqrt(M) : 0.0f;
        std::uniform_real_distribution<float> random_real(-1.0f * bound, +1.0f * bound); // Uniform distribution for shuffling the training data
        build_NN(net, N, M, input_size, output_size, random_real, RNG);
    }
    else
    {
        std::cerr << "Error: no data file or load file specified. Failed making an NN.\n";
        exit(-1);
    }
    if (LR == -1234)
        LR = 0.5 / N; // Both build_NN and load_network ensure that N>0

    bool did_something = false;
    if (data_file.is_open())
    {
        // std::shuffle(data.begin(), data.end(), RNG);  // Commented to keep the testing set consistent between attempts. Uncomment to enable better initial randomization.
        batch_size = (batch_size > 0 && batch_size * vec_width < data.size() * (1.0f - part_testing)) ? batch_size * vec_width : data.size() * (1.0f - part_testing);
        split_data(data, training, testing, batch_size, part_testing);

        std::cout << std::setprecision(4) << std::scientific;
        train(net, N, M, input_size, output_size, training, LR, RNG, batch_size, print);
        did_something = true;
        std::cout << "\nTraining Complete!\nLoss on testing dataset: " << get_loss(net, N, testing) << "\n\n";
    }

    if (input_file.is_open())
    {
        if (!output_file.is_open())
            output_file.open("output.txt");
        inference(net, N, input_file, output_file);
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
        save_network(net, N, M, input_size, output_size, LR, save_file);
    else if (did_something)
        std::cerr << "Warning: save file inaccessible. NN not saved.\n";

    std::cout << "\nProgram Time: " << std::chrono::duration_cast<std::chrono::seconds>(std::chrono::high_resolution_clock::now() - start) << "\n\n\n";
    return 0;
}