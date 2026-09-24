// circle_nn.cpp
// -----------------------------------------------------------------------------
// A small fully-connected feed-forward neural network, trained from scratch
// (no libraries) to classify a 2D point as being inside circle 1, circle 2,
// or neither, using the training data produced by generate_training_data.py.
//
// NETWORK SHAPE
//   Layer 0            : input layer      -> 2 nodes (x, y)
//   Layer 1..m         : hidden layers     -> n nodes each, sigmoid activation
//   Layer m+1 (logits)  : pre-softmax layer -> 3 nodes, linear activation
//   Layer m+2 (softmax) : output layer      -> 3 nodes, softmax probabilities
//
//   m and n are chosen by the user at training time (console prompt).
//   Every node in layer i holds one weight per node in layer i+1 (fully
//   connected). The softmax layer has no outgoing weights.
//
// TRAINING
//   Loss      : cross-entropy against the one-hot label from the data file
//               (label 0 -> [1,0,0], label 1 -> [0,1,0], label 2 -> [0,0,1])
//   Gradients : combined softmax+cross-entropy simplifies the logits-layer
//               delta to (predicted - target). Everything before that is
//               plain backprop / chain rule through the sigmoid layers.
//   Update    : online gradient descent, one training example at a time.
//
// See the long comment above main() / near backpropagate() for the exact
// math and for a note on one deliberate deviation from the prompt's wording
// of the weight-update rule.
// -----------------------------------------------------------------------------

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

// -----------------------------------------------------------------------------
// Core data structures
// -----------------------------------------------------------------------------

struct Node {
    std::vector<double> weights; // one weight per node in the NEXT layer (empty for the last layer)
    double value = 0.0;          // set during forward propagation
    double bias = 0.0;           // learnable bias
    double delta = 0.0;          // dLoss/d(pre-activation), set during backpropagation
};

using Layer = std::vector<Node>;
using NeuralNetwork = std::vector<Layer>;

struct Sample {
    double x, y;
    int label; // 0, 1, or 2
};

// -----------------------------------------------------------------------------
// Random number generation (single shared engine)
// -----------------------------------------------------------------------------

static std::mt19937 rng(std::random_device{}());

// -----------------------------------------------------------------------------
// Activation functions
// -----------------------------------------------------------------------------

inline double sigmoid(double x) {
    return 1.0 / (1.0 + std::exp(-x));
}

// Derivative of sigmoid expressed in terms of its OWN output (y = sigmoid(x)),
// which is what we have on hand after forward propagation: sigmoid'(x) = y*(1-y)
inline double sigmoid_derivative_from_output(double y) {
    return y * (1.0 - y);
}

// -----------------------------------------------------------------------------
// Network construction
// -----------------------------------------------------------------------------

// Builds a network with the given per-layer sizes and randomly initializes
// weights and biases.
//
// Weight init: Xavier/Glorot-style uniform init, range +/- 1/sqrt(fan_in),
// where fan_in is the size of the layer the weight originates FROM. This is
// a standard, reasonable choice for sigmoid-activated networks: it keeps
// the initial pre-activation values in a range where sigmoid isn't already
// saturated, which is important for gradients to actually flow early in
// training.
// Bias init: small uniform range +/- 0.1, a common default that avoids
// biasing any node strongly one way before training starts.
NeuralNetwork build_network(const std::vector<int>& layer_sizes) {
    NeuralNetwork net(layer_sizes.size());

    for (size_t i = 0; i < layer_sizes.size(); i++) {
        net[i].resize(layer_sizes[i]);
    }

    for (size_t i = 0; i + 1 < layer_sizes.size(); i++) {
        double bound = 1.0 / std::sqrt(static_cast<double>(layer_sizes[i]));
        std::uniform_real_distribution<double> weight_dist(-bound, bound);
        std::uniform_real_distribution<double> bias_dist(-0.1, 0.1);

        for (auto& node : net[i]) {
            node.weights.resize(layer_sizes[i + 1]);
            for (auto& w : node.weights) w = weight_dist(rng);
        }
        // Biases live on the RECEIVING layer (i+1), so seed them there.
        for (auto& node : net[i + 1]) {
            node.bias = bias_dist(rng);
        }
    }
    // Input layer (layer 0) biases are unused (no activation is applied to
    // input nodes) but we still leave them at 0 for cleanliness.
    for (auto& node : net[0]) node.bias = 0.0;

    return net;
}

// -----------------------------------------------------------------------------
// Forward propagation
// -----------------------------------------------------------------------------
//
// Layers 0            : input      -> values set directly from the sample
// Layers 1..(L-3)      : hidden     -> sigmoid(weighted sum + bias), per the
//                         triple for-loop pattern given in the prompt
// Layer  (L-2)         : logits     -> LINEAR (weighted sum + bias), no
//                         squashing function, since softmax needs the raw
//                         logits together
// Layer  (L-1)         : softmax    -> computed from the logits layer as a
//                         whole (can't be done node-by-node like the others)
//
// L = net.size().

void forward_propagation(NeuralNetwork& net, double x, double y) {
    // Layer 0: input
    net[0][0].value = x;
    net[0][1].value = y;

    int L = static_cast<int>(net.size());
    int logits_layer = L - 2;

    // Layers 1 .. logits_layer (inclusive): the standard triple for-loop,
    // exactly as in the prompt, with the activation swapped depending on
    // whether we're in a hidden layer (sigmoid) or the logits layer (linear).
    for (int i = 1; i <= logits_layer; i++) {
        bool is_logits_layer = (i == logits_layer);
        for (int j = 0; j < (int)net[i].size(); j++) {
            net[i][j].value = net[i][j].bias;
            for (int k = 0; k < (int)net[i - 1].size(); k++) {
                net[i][j].value += net[i - 1][k].weights[j] * net[i - 1][k].value;
            }
            if (!is_logits_layer) {
                net[i][j].value = sigmoid(net[i][j].value);
            }
            // else: linear activation, leave as-is
        }
    }

    // Final layer: softmax over the logits layer's values.
    // Subtracting the max logit before exponentiating is a standard
    // numerical-stability trick; it doesn't change the resulting
    // probabilities, only avoids overflow in std::exp.
    double max_logit = net[logits_layer][0].value;
    for (auto& node : net[logits_layer]) max_logit = std::max(max_logit, node.value);

    double sum_exp = 0.0;
    std::vector<double> exp_vals(net[logits_layer].size());
    for (size_t j = 0; j < net[logits_layer].size(); j++) {
        exp_vals[j] = std::exp(net[logits_layer][j].value - max_logit);
        sum_exp += exp_vals[j];
    }
    for (size_t j = 0; j < net.back().size(); j++) {
        net.back()[j].value = exp_vals[j] / sum_exp;
    }
}

// -----------------------------------------------------------------------------
// Loss
// -----------------------------------------------------------------------------

// Cross-entropy loss for one sample, given the network's current softmax
// output and the true one-hot label.
double cross_entropy_loss(const NeuralNetwork& net, int label) {
    const Layer& out = net.back();
    // Only the true class contributes to cross-entropy since the other
    // target probabilities are 0. Clamp to avoid log(0).
    double p = std::max(out[label].value, 1e-12);
    return -std::log(p);
}

// -----------------------------------------------------------------------------
// Backpropagation + gradient descent
// -----------------------------------------------------------------------------
//
// Delta convention used throughout: net[i][j].delta = dLoss / d(pre-activation
// of node j in layer i). This is the standard convention that makes both the
// weight-update rule and the next layer's delta formula simple.
//
// 1) Logits layer delta (this is the layer the prompt calls "the output
//    layer" for backprop purposes -- the softmax layer itself has no
//    weights, so there's nothing to compute a delta FOR there):
//        delta_j = predicted_j - target_j
//    This is the classic simplification that falls out of combining
//    cross-entropy loss with a softmax activation.
//
// 2) Hidden layer delta (chain rule, "sum over the next layer" -- this is
//    the one place a sum over the next layer's deltas genuinely belongs):
//        delta_k = sigmoid'(value_k) * sum_j( weight_{k->j} * delta_j )
//    where the sum is over every node j in the next layer.
//
// 3) Weight update -- NOTE: this is one spot where I implemented something
//    slightly different from the literal wording in the prompt. The prompt
//    describes updating "each weight" using "the sum of all deltas of the
//    next layer". Taken completely literally that would mean every outgoing
//    weight of a node gets nudged by the SAME combined number, which isn't
//    correct: weight_{k->j} only influences the loss through node j
//    specifically, not through every node in the next layer. So each weight
//    is updated using only the delta of the ONE next-layer node it feeds
//    into (this is standard gradient descent for a fully-connected layer):
//        weight_{k->j} -= learning_rate * delta_j * value_k
//    The "sum over the next layer" idea IS used, just one step earlier, in
//    the hidden-layer delta formula above (step 2), which is where a sum
//    over the next layer is mathematically required.
//
// 4) Bias update, exactly as specified:
//        bias_j -= learning_rate * delta_j

void backpropagate(NeuralNetwork& net, int label, double learning_rate) {
    int L = static_cast<int>(net.size());
    int logits_layer = L - 2;

    // Step 1: delta at the logits layer.
    for (size_t j = 0; j < net[logits_layer].size(); j++) {
        double target = (static_cast<int>(j) == label) ? 1.0 : 0.0;
        double predicted = net.back()[j].value; // softmax output
        net[logits_layer][j].delta = predicted - target;
    }

    // Step 2: delta for every hidden layer, walking backwards.
    for (int i = logits_layer - 1; i >= 1; i--) {
        for (int k = 0; k < (int)net[i].size(); k++) {
            double sum = 0.0;
            for (int j = 0; j < (int)net[i + 1].size(); j++) {
                sum += net[i][k].weights[j] * net[i + 1][j].delta;
            }
            net[i][k].delta = sigmoid_derivative_from_output(net[i][k].value) * sum;
        }
    }

    // Step 3 + 4: gradient descent on weights and biases for every
    // connection from layer i to layer i+1, for i = 0 .. logits_layer - 1.
    for (int i = 0; i <= logits_layer - 1; i++) {
        for (int k = 0; k < (int)net[i].size(); k++) {
            for (int j = 0; j < (int)net[i + 1].size(); j++) {
                net[i][k].weights[j] -= learning_rate * net[i + 1][j].delta * net[i][k].value;
            }
        }
        for (int j = 0; j < (int)net[i + 1].size(); j++) {
            net[i + 1][j].bias -= learning_rate * net[i + 1][j].delta;
        }
    }
}

// -----------------------------------------------------------------------------
// Data loading
// -----------------------------------------------------------------------------

// Training data format (matches generate_training_data.py's output): pairs
// of lines, "x y" followed by a label (0, 1, or 2), repeated for every point.
std::vector<Sample> load_training_data(const std::string& path) {
    std::ifstream in(path);
    if (!in) {
        std::cerr << "Error: could not open training data file '" << path << "'\n";
        std::exit(1);
    }

    std::vector<Sample> data;
    double x, y;
    int label;
    while (in >> x >> y >> label) {
        if (label < 0 || label > 2) {
            std::cerr << "Warning: skipping sample with out-of-range label " << label << "\n";
            continue;
        }
        data.push_back({x, y, label});
    }

    if (data.empty()) {
        std::cerr << "Error: no valid samples found in '" << path << "'\n";
        std::exit(1);
    }
    return data;
}

// Inference input format: one point per line, "x y".
std::vector<std::pair<double, double>> load_inference_inputs(const std::string& path) {
    std::ifstream in(path);
    if (!in) {
        std::cerr << "Error: could not open inference input file '" << path << "'\n";
        std::exit(1);
    }
    std::vector<std::pair<double, double>> points;
    double x, y;
    while (in >> x >> y) points.emplace_back(x, y);

    if (points.empty()) {
        std::cerr << "Error: no valid points found in '" << path << "'\n";
        std::exit(1);
    }
    return points;
}

// -----------------------------------------------------------------------------
// Save / load a trained model
// -----------------------------------------------------------------------------
//
// Simple, human-readable text format:
//   NEURALNET_V1
//   <number of layers>
//   <layer 0 size> <layer 1 size> ... <layer L-1 size>
//   then, for every layer i and every node j in that layer, in order:
//     <bias>
//     <number of outgoing weights> <w0> <w1> ...
// Node "value" and "delta" are transient (recomputed every forward/back
// pass) so they are not saved.

void save_network(const NeuralNetwork& net, const std::string& path) {
    std::ofstream out(path);
    if (!out) {
        std::cerr << "Error: could not open '" << path << "' for writing.\n";
        std::exit(1);
    }
    out << "NEURALNET_V1\n";
    out << net.size() << "\n";
    for (const auto& layer : net) out << layer.size() << " ";
    out << "\n";

    out << std::setprecision(17);
    for (const auto& layer : net) {
        for (const auto& node : layer) {
            out << node.bias << "\n";
            out << node.weights.size();
            for (double w : node.weights) out << " " << w;
            out << "\n";
        }
    }
    std::cout << "Model saved to '" << path << "'\n";
}

NeuralNetwork load_network(const std::string& path) {
    std::ifstream in(path);
    if (!in) {
        std::cerr << "Error: could not open model file '" << path << "'\n";
        std::exit(1);
    }
    std::string header;
    in >> header;
    if (header != "NEURALNET_V1") {
        std::cerr << "Error: '" << path << "' is not a recognized model file.\n";
        std::exit(1);
    }
    size_t num_layers;
    in >> num_layers;
    std::vector<int> sizes(num_layers);
    for (size_t i = 0; i < num_layers; i++) in >> sizes[i];

    NeuralNetwork net(num_layers);
    for (size_t i = 0; i < num_layers; i++) {
        net[i].resize(sizes[i]);
        for (auto& node : net[i]) {
            in >> node.bias;
            size_t num_weights;
            in >> num_weights;
            node.weights.resize(num_weights);
            for (auto& w : node.weights) in >> w;
        }
    }
    std::cout << "Model loaded from '" << path << "' (" << num_layers << " layers)\n";
    return net;
}

// -----------------------------------------------------------------------------
// Training driver
// -----------------------------------------------------------------------------

void train(NeuralNetwork& net, const std::vector<Sample>& data, int iterations, double learning_rate) {
    double last_epoch_total_loss = 0.0;

    for (int epoch = 1; epoch <= iterations; epoch++) {
        double total_loss = 0.0;
        for (const auto& sample : data) {
            forward_propagation(net, sample.x, sample.y);
            total_loss += cross_entropy_loss(net, sample.label);
            backpropagate(net, sample.label, learning_rate);
        }
        std::cout << "Epoch " << epoch << "/" << iterations
                   << " - total loss: " << total_loss << "\n";
        last_epoch_total_loss = total_loss;
    }

    std::cout << "\nTraining complete. Total loss on the final training pass: "
               << last_epoch_total_loss << "\n";
}

// -----------------------------------------------------------------------------
// Prediction / reporting
// -----------------------------------------------------------------------------

// Runs forward propagation and returns (predicted_label, probabilities[3]).
std::pair<int, std::vector<double>> predict(NeuralNetwork& net, double x, double y) {
    forward_propagation(net, x, y);
    std::vector<double> probs(3);
    for (int c = 0; c < 3; c++) probs[c] = net.back()[c].value;
    int predicted = static_cast<int>(std::max_element(probs.begin(), probs.end()) - probs.begin());
    return {predicted, probs};
}

std::string format_prediction(int predicted, const std::vector<double>& probs) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(3);
    for (int c = 0; c < 3; c++) {
        oss << c << " (" << (probs[c] * 100.0) << "%)";
        if (c != 2) oss << "  ";
    }
    oss << "  ->  Predicted: " << predicted;
    return oss.str();
}

// -----------------------------------------------------------------------------
// Command-line argument parsing
// -----------------------------------------------------------------------------

struct Args {
    std::string data_path;
    std::string load_path;
    std::string save_path = "trained_net.txt";
    std::string inputs_path;
    std::string output_path;
    double learning_rate = 0.25;
    bool has_data = false;
    bool has_load = false;
    bool has_inputs = false;
    bool has_output = false;
    bool help = false;
};

void print_usage(const char* prog_name) {
    std::cout
        << "Usage: " << prog_name << " [options]\n\n"
        << "All Options: (order isn't important)\n"
        << "  -d --data <path>     Training data file containing space-separated input and output lines.\n"
        << "  -l --load <path>     Load a trained model from file to inference with.\n"
        << "  -s --save <path>     Save the trained model to file. Defaults to trained_net.txt.\n"
        << "  --lr <value>         Set the initial learning rate for training. Default is 0.25.\n"
        << "  -i --inputs <path>   Load inference input from file.\n"
        << "  -o --output <path>   Save inference output to file.\n"
        << "  -h --help            Show this usage information.\n"
        << "Note: If both --load and --data are specified, the model will be loaded and then further trained on the data as needed.\n"
        << "\n"
        << "If -l/--load is given without -d/--data and without -i/--inputs, the\n"
        << "program drops into an interactive prediction loop where you can type\n"
        << "\"x y\" pairs at the console and get an immediate prediction.\n";
}

Args parse_args(int argc, char** argv) {
    Args args;
    auto need_value = [&](int& i, const std::string& flag) -> std::string {
        if (i + 1 >= argc) {
            std::cerr << "Error: " << flag << " requires a value.\n";
            std::exit(1);
        }
        return argv[++i];
    };

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "-d" || a == "--data") {
            args.data_path = need_value(i, a);
            args.has_data = true;
        } else if (a == "-l" || a == "--load") {
            args.load_path = need_value(i, a);
            args.has_load = true;
        } else if (a == "-s" || a == "--save") {
            args.save_path = need_value(i, a);
        } else if (a == "--lr") {
            args.learning_rate = std::stod(need_value(i, a));
        } else if (a == "-i" || a == "--inputs") {
            args.inputs_path = need_value(i, a);
            args.has_inputs = true;
        } else if (a == "-o" || a == "--output") {
            args.output_path = need_value(i, a);
            args.has_output = true;
        } else if (a == "-h" || a == "--help") {
            args.help = true;
        } else {
            std::cerr << "Error: unrecognized option '" << a << "'\n";
            print_usage(argv[0]);
            std::exit(1);
        }
    }
    return args;
}

// -----------------------------------------------------------------------------
// main
// -----------------------------------------------------------------------------

int main(int argc, char** argv) {
    Args args = parse_args(argc, argv);

    if (args.help) {
        print_usage(argv[0]);
        return 0;
    }

    if (!args.has_load && !args.has_data) {
        std::cerr << "Error: you must specify at least -l/--load or -d/--data.\n\n";
        print_usage(argv[0]);
        return 1;
    }

    NeuralNetwork net;

    if (args.has_load) {
        net = load_network(args.load_path);
    }

    if (args.has_data) {
        std::vector<Sample> data = load_training_data(args.data_path);
        std::cout << "Loaded " << data.size() << " training samples from '" << args.data_path << "'\n";

        int iterations;

        if (!args.has_load) {
            // Brand new network: ask for its shape.
            int m, n;
            std::cout << "Number of hidden layers (m): ";
            std::cin >> m;
            std::cout << "Number of nodes per hidden layer (n): ";
            std::cin >> n;

            if (m < 0 || n < 1) {
                std::cerr << "Error: m must be >= 0 and n must be >= 1.\n";
                return 1;
            }

            std::vector<int> layer_sizes;
            layer_sizes.push_back(2); // input layer: x, y
            for (int i = 0; i < m; i++) layer_sizes.push_back(n);
            layer_sizes.push_back(3); // logits layer
            layer_sizes.push_back(3); // softmax layer

            net = build_network(layer_sizes);
            std::cout << "Built a new network with " << net.size() << " layers "
                       << "(2 -> " << std::string(m, 'n') << (m > 0 ? " (n each) -> " : "")
                       << "3 -> 3).\n";
        }

        std::cout << "Number of training iterations: ";
        std::cin >> iterations;
        if (iterations < 1) {
            std::cerr << "Error: iterations must be >= 1.\n";
            return 1;
        }

        std::cout << "Training with learning rate " << args.learning_rate << "...\n";
        train(net, data, iterations, args.learning_rate);

        save_network(net, args.save_path);
    }

    if (args.has_inputs) {
        auto points = load_inference_inputs(args.inputs_path);

        std::ostream* out = &std::cout;
        std::ofstream file_out;
        if (args.has_output) {
            file_out.open(args.output_path);
            if (!file_out) {
                std::cerr << "Error: could not open '" << args.output_path << "' for writing.\n";
                return 1;
            }
            out = &file_out;
        }

        for (auto& p : points) {
            auto [predicted, probs] = predict(net, p.first, p.second);
            (*out) << std::fixed << std::setprecision(6) << p.first << " " << p.second << "\n";
            (*out) << format_prediction(predicted, probs) << "\n";
        }

        if (args.has_output) {
            std::cout << "Inference output saved to '" << args.output_path << "'\n";
        }
    } else if (args.has_load && !args.has_data) {
        // Nothing else to do: drop into an interactive prediction loop.
        std::cout << "\nInteractive prediction mode. Enter \"x y\", or \"q\" to quit.\n";
        std::string line;
        while (true) {
            std::cout << "> ";
            if (!std::getline(std::cin, line)) break;
            if (line == "q" || line == "quit") break;
            std::istringstream iss(line);
            double x, y;
            if (!(iss >> x >> y)) {
                std::cout << "Could not parse \"" << line << "\" as \"x y\". Try again.\n";
                continue;
            }
            auto [predicted, probs] = predict(net, x, y);
            std::cout << format_prediction(predicted, probs) << "\n";
        }
    }

    return 0;
}