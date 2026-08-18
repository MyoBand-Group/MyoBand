/*   Author: Nikola D. Lilov, 18 Aug 2026
 *
 *  Desc: ...
 *
 *  Common Usage: makeNN.exe -d dataset.txt
 *  For expanded usage: makeNN.exe -h
 */
/* The data should be in the format:
    feature1 feature2 feature3 ... featureX
    output1 output2 output3 ... outputY
    feature1 feature2 feature3 ... featureX
    output1 output2 output3 ... outputY
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
// #include <simd>
#include <experimental/simd>
#include <numeric>
#include <algorithm>
#include <string>
#include <vector>

// namespace simd = std::simd;
namespace simd = std::experimental;
using simd_t = simd::native_simd<float>;
constexpr std::size_t vec_width = simd_t::size();

void help(const std::string &exe)
{
    std::cerr << "Common Usages:\n"
              << "  " << exe << " -d dataset.txt\n"
              << "  " << exe << " -l trained_net.txt -i inputs.txt \n"
              << "\n"
              << "All Options: (order isn't important)\n"
              << "  -d --data <path>     Training data file containing space-separated input and output lines.\n"
              << "  -l --load <path>     Load a trained model from file to inference* with.\n"
              << "  -s --save <path>     Save the trained model to file. Defaults to trained_net.txt.\n"
              << "  -t --testing <float> Set what proportion of the dataset (-d) should be for testing. Defaults to 0.20.\n"
              << "  -b --batch_size <int>         Set how big the training batches** should be (0 <=> BGD). Defaults to 1.\n"
              << "  -lr --learning_rate <float>   Set the initial learning rate for training. Defaults to 0.5/N.\n"
              << "  -i --inputs <path>   Load inference input from file.\n"
              << "  -o --output <path>   Save inference output to file. Defaults to output.txt.\n"
              << "  -h --help            Show this usage information.\n"
              << "\n"
              << " *Note: If both --load and --data are specified, the model will be loaded and then further trained on the data as needed.\n"
              << "**Note: This code utilizes the CPU's capability to train on multiple examples at a time. Your inputted batch_size will then be multiplied by the number of samples your CPU can parallelize.\n"
              << "\n"
              << "Example --data file format:\n"
              << "  <input1> <input2> ... <inputX>\n  <output1> <output2> ... <outputY>\n  <input1> <input2> ... <inputX>\n  <output1> <output2> ... <outputY>\n  ...\n";
    exit(0);
}

void initialize(int &argc, char **argv, std::ifstream &data_file, std::ifstream &load_file, std::ifstream &input_file,
                std::ofstream &output_file, int &whereToSave, int &batch_size, float &part_testing, float &LR)
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
                batch_size = std::stod(argv[i]);
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

    std::vector<std::vector<float>> weight, grad_w;
    std::vector<float> bias, grad_b;

    layer() = default;

    layer(const int &M)
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
                weight[i][j] = distribution(gen);
                grad_w[i][j] = 0.0f;
            }
            bias[i] = distribution(gen);
            grad_b[i] = 0.0f;
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

void buildNN(NN &net, const int &N, const int &M, const int &input_size, const int &output_size, std::uniform_real_distribution<float> &distribution, std::mt19937 &gen)
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
void read(const std::string &line, layer &l)
{
    std::istringstream in(line);
    float curr;
    for (int i = 0; in >> curr && i < l.size(); i++)
        l.value[i] = simd_t(curr);
}

using dataset = std::vector<std::pair<std::vector<float>, std::vector<float>>>; // A dataset is a vector of pairs of training input and output layers
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
    for (int i = testing_size; i < data.size(); i++)
        training.push_back(data[i]);
    std::cout << "\nSplitting data...\n"
              << " Testing set: " << testing_size << " / " << data.size() << " (" << 100.0f * testing.size() / data.size() << "%)\n"
              << " Training set: " << training.size() / batch_size << 'x' << batch_size << " / " << data.size() << " (" << 100.0f * training.size() / data.size() << "%)\n\n\n";
}

simd_t d_loss(const simd_t &expected, const simd_t &output)
{
    return 2 * (output - expected); // Derivative of the loss function with respect to the output
}
simd_t activation(const simd_t &x)
{
    return x / (simd_t(1.0f) + simd::abs(x)); // Sigmoid activation function (from -1 to 1)
}
simd_t d_activation(const simd_t &x)
{
    simd_t t = simd_t(1.0f) - simd::abs(x);
    return t * t;
}

void forward_propagation(NN &net, const int &N)
{
    for (int i = 1; i <= N + 1; i++)
        for (int j = 0; j < net[i].size(); j++)
        {
            net[i].value[j] = simd_t(net[i].bias[j]);
            for (int k = 0; k < net[i - 1].size(); k++)
                net[i].value[j] += simd_t(net[i - 1].weight[k][j]) * net[i - 1].value[k];
            net[i].value[j] = activation(net[i].value[j]);
        }
}

void backward_propagation(NN &net, const int &N, layer &expected_output)
{
    // Calculate the delta for the output layer
    for (int j = 0; j < net[N + 1].size(); j++)
    {
        net[N + 1].delta[j] = d_loss(expected_output.value[j], net[N + 1].value[j]) * d_activation(net[N + 1].value[j]);
        net[N + 1].grad_b[j] += simd::reduce(net[N + 1].delta[j]);
    }

    // Backward pass
    simd_t d_act;
    for (int i = N; i >= 0; i--)
        for (int j = 0; j < net[i].size(); j++)
        {
            net[i].delta[j] = simd_t(0.0f);
            d_act = d_activation(net[i].value[j]);
            for (int k = 0; k < net[i + 1].size(); k++)
            {
                net[i].delta[j] += net[i + 1].delta[k] * d_act * simd_t(net[i].weight[j][k]);
                net[i].grad_w[j][k] += simd::reduce(net[i + 1].delta[k] * net[i].value[j]);
            }
            net[i].grad_b[j] += simd::reduce(net[i].delta[j]);
        }
}

void update(NN &net, const int &N, const float &LR, const int &batch_size)
{
    const float LR_ = LR / batch_size;
    for (int i = 0; i <= N + 1; i++)
        for (int j = 0; j < net[i].size(); j++)
        {
            net[i].bias[j] -= LR_ * net[i].grad_b[j];
            net[i].grad_b[j] = 0.0f;
            for (int k = 0; k < net[i].weight[j].size(); k++)
            {
                net[i].weight[j][k] -= LR_ * net[i].grad_w[j][k];
                net[i].grad_w[j][k] = 0.0f;
            }
        }
}

float get_loss(NN &net, const int &N, dataset &data)
{
    layer expected_output(net.back().size());
    int cnt = 0;
    float total_loss = 0.0f, diff;

    for (const auto &data_point : data)
    {
        cnt++;
        for (int j = 0; j < data_point.first.size(); j++)
            net[0].value[j] = simd_t(data_point.first[j]);
        for (int j = 0; j < data_point.second.size(); j++)
            expected_output.value[j] = simd_t(data_point.second[j]);
        forward_propagation(net, N);
        for (int j = 0; j < net[N + 1].size(); j++)
        {
            diff = (net[N + 1].value[j][0] - expected_output.value[j][0]);
            total_loss += diff * diff;
        }
    }

    return (cnt == 0) ? 0 : total_loss / cnt;
}

void train(NN &net, const int &N, const int &M, const int &input_size, const int &output_size, dataset &training, float &LR, std::mt19937 &gen, const int &batch_size = 8)
{
    float last_loss = get_loss(net, N, training), curr_loss;
    auto last_time = std::chrono::high_resolution_clock::now(), curr_time = std::chrono::high_resolution_clock::now();
    std::cout << "Training neural network on " << training.size() << " data points. Initial loss: " << last_loss << "\n\n";
    layer expected_output(net.back().size());
    int epochs;

    while (true)
    {
        std::cout << "How many epochs to train the NN for: ";
        std::cin >> epochs;
        if (epochs <= 0)
            break;

        last_time = std::chrono::high_resolution_clock::now();

        while (epochs--)
        {
            if (epochs != 0)
            {
                if (epochs % 5 == 0)
                {
                    curr_loss = get_loss(net, N, training);
                    if (curr_loss > 1.025 * last_loss)
                        LR *= 0.975; // If the loss increased, the training is unstable. Reduce the learning rate.
                    last_loss = curr_loss;
                }
                if (epochs % 100 == 0)
                {
                    curr_time = std::chrono::high_resolution_clock::now();
                    std::cout << "   Time Left: " << epochs * std::chrono::duration_cast<std::chrono::seconds>(curr_time - last_time) / 100 << ";   \tEpochs: " << epochs << ";\t\tLR: " << LR << ";   \tLoss: " << last_loss << "\n";
                    last_time = curr_time;
                }
            }

            std::shuffle(training.begin(), training.end(), gen); // Shuffle the order of the training data for this epoch
            for (int i = 0; i < training.size(); i += vec_width)
            {
                for (int lane = 0; lane < vec_width; lane++)
                {
                    for (int j = 0; j < training[i].first.size(); j++)
                        net[0].value[j][lane] = training[i + lane].first[j];
                    for (int j = 0; j < training[i].second.size(); j++)
                        expected_output.value[j][lane] = training[i + lane].second[j];
                }
                forward_propagation(net, N);
                backward_propagation(net, N, expected_output);
                if ((i + vec_width) % batch_size == 0) // Update the weight and biases after each batch
                    update(net, N, LR, batch_size);
            }
        }

        last_loss = get_loss(net, N, training);
        std::cout << "Training complete. Loss: " << last_loss << "\n\n";
    }
}

void inference(NN &net, const int &N, std::ifstream &input_file, std::ofstream &output_file)
{
    // Now we can use the trained NN to make predictions on new data
    std::cout << "Making predictions on data from input file... Saving to output file...\n";
    std::string line;
    while (std::getline(input_file, line))
    {
        read(line, net[0]);
        forward_propagation(net, N);

        for (auto v : net[N + 1].value)
            output_file << v[0] << " ";
        output_file << "\n";
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

    // layer = 0
    for (int j = 0; j < input_size; j++)
    {
        std::getline(load_file, line);
        in.clear();
        in.str(line);
        for (int k = 0; k < M; k++)
            in >> net[0].weight[j][k];
        in >> net[0].bias[j];
    }

    // layer = 1..N-1
    for (int i = 1; i < N; i++)
        for (int j = 0; j < M; j++)
        {
            std::getline(load_file, line);
            in.clear();
            in.str(line);
            for (int k = 0; k < M; k++)
                in >> net[i].weight[j][k];
            in >> net[i].bias[j];
        }

    // layer = N
    for (int j = 0; j < M; j++)
    {
        std::getline(load_file, line);
        in.clear();
        in.str(line);
        for (int k = 0; k < output_size; k++)
            in >> net[N].weight[j][k];
        in >> net[N].bias[j];
    }

    // layer = N+1
    for (int j = 0; j < output_size; j++)
    {
        std::getline(load_file, line);
        in.clear();
        in.str(line);
        in >> net[N + 1].bias[j];
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
                save_file << w << ' ';
            save_file << l.bias[i] << "\n";
        }

    save_file.close();
}

int main(int argc, char **argv)
{

    std::cout << "\n\n\n";

    int N, M, input_size, output_size;
    float part_testing = 0.2f, LR = -1234;

    int whereToSave = 0, batch_size = 1;
    std::ifstream data_file, load_file, input_file;
    std::ofstream output_file, save_file;

    initialize(argc, argv, data_file, load_file, input_file, output_file, whereToSave, batch_size, part_testing, LR);

    dataset data, training, testing;
    if (data_file.is_open())
    {
        getIOsize(data_file, input_size, output_size);
        get_data(data_file, data, input_size, output_size);
    }

    NN net;

    std::mt19937 RNG(time(0)); // Random number generator for generating initial weight and biases and for shuffling the training data
    float bound;

    if (load_file.is_open())
        load_network(net, N, M, input_size, output_size, LR, load_file);
    else if (data_file.is_open())
    {
        std::cout << "Creating neural network... Define its size <depth N> <layer width M>: ";
        std::cin >> N >> M;
        bound = (M > 0) ? 1.0f / sqrt(M) : 0.0f;
        std::uniform_real_distribution<float> random_real(-1.0f * bound, +1.0f * bound); // Uniform distribution for shuffling the training data
        buildNN(net, N, M, input_size, output_size, random_real, RNG);
    }
    else
    {
        std::cerr << "Error: no data file or load file specified. Failed making an NN.\n";
        exit(-1);
    }
    if (LR == -1234)
        LR = 0.5 / N;

    bool did_something = false;
    if (data_file.is_open())
    {
        // std::shuffle(data.begin(), data.end(), RNG);
        batch_size = (batch_size > 0 && batch_size * vec_width < data.size() * (1.0f - part_testing)) ? batch_size * vec_width : data.size() * (1.0f - part_testing);
        split_data(data, training, testing, batch_size, part_testing);

        std::cout << std::setprecision(4) << std::scientific;
        train(net, N, M, input_size, output_size, training, LR, RNG, batch_size);
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

    std::cout << "\n\n";
    return 0;
}