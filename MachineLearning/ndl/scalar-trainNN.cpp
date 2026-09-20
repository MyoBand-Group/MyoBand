/*   Author: Nikola D. Lilov, 13 Aug 2026
  *
  *  Desc: Trains a simple NxM neural network with a decreasing LR, a
  *  sigmoid activation function and squared loss function. Requires a
  *  data file with alternating input and expected output lines. Saves
  *  (and can load) the trained net to a file to later retrain or
  *  inference on.
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

#include<fstream>
#include<iostream>
#include<sstream>
#include<iomanip>
#include<cmath>
#include<random>
#include<ctime>
#include<chrono>
#include<cstdlib>
#include<numeric>
#include<algorithm>
#include<string>
#include<vector>

int N,M; //N = number of hidden layers, M = number of nodes per hidden layer
double LR=-1,bound=0.25; //Learning rate and weight initialization bound
std::mt19937 RNG(time(0)); //Random number generator for generating initial weights and biases and for shuffling the training data

/*std::string lower(std::string str){
	for(char& c:str)
		if(c>='A' && c<='Z')
			c-=('A'-'a');
	return str;
}*/

void initialize(int& argc, char** argv, std::ifstream& data_file, std::ifstream& load_file, std::ifstream& input_file,
				std::ofstream& output_file, int& whereToSave, int& batches, float& part_testing, int& print){

	bool show_help = false;
	if(argc==1) show_help = true;
    else for(int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--data" || arg == "-d") {
            if (++i >= argc) {
                std::cerr << "Missing value for --data\n";
                exit(-1);
            }
            data_file.open(argv[i]);
            if (!data_file.is_open()) {
                std::cerr << "Error: could not open data file: " << argv[i] << "\n";
                exit(-1);
            }
        } else if (arg == "--inputs" || arg == "-i") {
            if (++i >= argc) {
                std::cerr << "Missing value for --inputs\n";
                exit(-1);
            }
            input_file.open(argv[i]);
            if (!input_file.is_open()) {
                std::cerr << "Error: could not open input file: " << argv[i] << "\n";
                exit(-1);
            }
        } else if (arg == "--output" || arg == "-o") {
            if (++i >= argc) {
                std::cerr << "Missing value for --output\n";
                exit(-1);
            }
            output_file.open(argv[i]);
            if (!output_file.is_open()) {
                std::cerr << "Error: could not open output file: " << argv[i] << "\n";
                exit(-1);
            }
        } else if (arg == "--load" || arg == "-l") {
            if (++i >= argc) {
                std::cerr << "Missing value for --load\n";
                exit(-1);
            }
            load_file.open(argv[i]);
            if (!load_file.is_open()) {
                std::cerr << "Error: could not open load file: " << argv[i] << "\n";
                exit(-1);
            }

        } else if (arg == "--print" || arg == "-p"){
            if (++i >= argc){
                std::cerr << "Missing value for --print\n";
                exit(-1);
            }
            print = std::stoi(argv[i]);
            if (print <= 0){
                std::cerr << "Error: --print must be greater than zero seconds\n";
                exit(-1);
            }
        } else if (arg == "--save" || arg == "-s") {
        if (++i >= argc) {
            std::cerr << "Missing value for --save\n";
            exit(-1);
        }
        whereToSave = i;
        } else if (arg == "--learning_rate" || arg == "-lr") {
            if (++i >= argc) {
                std::cerr << "Missing value for --lr\n";
                exit(-1);
            }
            LR = std::stod(argv[i]);
        } else if (arg == "--batches" || arg == "-b") {
            if (++i >= argc) {
                std::cerr << "Missing value for --batches\n";
                exit(-1);
            }
            batches = std::stod(argv[i]);
        } else if (arg == "--testing" || arg == "-t") {
            if (++i >= argc) {
                std::cerr << "Missing value for --testing\n";
                exit(-1);
            }
            part_testing = std::stof(argv[i]);
        /*} else if (arg == "--activation" || arg == "-af" || arg == "-a") {
            if (++i >= argc) {
                std::cerr << "Missing value for --activation (activation_function)\n";
                exit(-1);
            }
            activation = lower(argv[i]);
        } else if (arg == "--loss" || arg == "-lf" || arg == "-L") {
            if (++i >= argc) {
                std::cerr << "Missing value for --loss (loss_function)\n";
                exit(-1);
            }
            loss = lower(argv[i]);*/
        } else if (arg == "--help" || arg == "-h") {
            show_help = true;
        } else {
            std::cerr << "Unknown argument: " << arg << "\n";
            show_help = true;
        }
    }

    if (show_help) {
        std::cerr << "Common Usages:\n"
              << "  " << argv[0] << " -d dataset.txt\n"
              << "  " << argv[0] << " -l trained_net.txt -i inputs.txt \n"
              << "\n"
              << "All Options: (order isn't important)\n"
              << "  -d --data <path>     Training data file containing space-separated input and output lines.\n"
              << "  -l --load <path>     Load a trained model from file to inference with.\n"
              << "  -s --save <path>     Save the trained model to file. Defaults to trained_net.txt.\n"
              << "  -p --print <int>     How often to print status updates, in milliseconds. Defaults to 5000.\n"
			  << "  -t --testing <float> Set what proportion of the dataset (-d) should be for testing. Defaults to 0.20.\n"
			  << "  -b --batches <int>   Set how many batches to divide the training set into (0 <=> SGD). Defaults to 25.\n"
			  << "  -lr --learning_rate <float>   Set the initial learning rate for training. Defaults is 0.5/N.\n"
			  //<< "  -a -af --activation <string>  Set the activation function to be used by the program. (linear, sigmoid, tanh, relu, leaky_relu, softplus, selu)\n"
			  //<< "  -L -lf --loss <string>        Set the loss function to be used by the program. ()\n"
              << "  -i --inputs <path>   Load inference input from file.\n"
              << "  -o --output <path>   Save inference output to file. Defaults to output.txt.\n"
              << "  -h --help            Show this usage information.\n"
			  << "Note: If both --load and --data are specified, the model will be loaded and then further trained on the data as needed.\n"
			  << "\n"
			  << "Example --data file format:\n"
			  << "  <input1> <input2> ... <inputX>\n  <output1> <output2> ... <outputY>\n  <input1> <input2> ... <inputX>\n  <output1> <output2> ... <outputY>\n  ...\n";
        exit(0);
    }
}

std::uniform_real_distribution<double> random_real(-1.0, 1.0); //Uniform distribution for shuffling the training data
struct node{
	std::vector<double> weights,grad_w;
	double bias,grad_b;
	double value;
	double delta;

	node(){
		weights.resize(M); grad_w.resize(M);
		//bound = 1.0/sqrt(M); //Heuristic for weight initialization
		for(int i=0;i<M;i++){
			weights[i]=random_real(RNG)*bound; //Random weights between [-1.0; 1.0] * bound
			grad_w[i]=0.;
		}
		bias=random_real(RNG)*bound;
		grad_b=0.;
		value=0.;
		delta=0.;
	}
};
using layer=std::vector<node>;
using NN=std::vector<layer>;

void read(std::string line,std::vector<double>& v){
	std::istringstream in(line);
	double curr;
	for(int i=0; in>>curr && i<v.size(); i++)
		v[i]=curr;
}
void readl(std::string line,layer& l){
	std::istringstream in(line);
	double curr;
	for(int i=0; in>>curr && i<l.size(); i++)
		l[i].value=curr;
}

void fixLayer(layer& l, std::ifstream& data_file){
	std::string line;
	if(!std::getline(data_file,line)){
		std::cerr<<"Error: could not read from data file\n";
		exit(-1);
	}
	std::istringstream in(line);
	double curr;
	while(in>>curr)
		l.push_back(node());
}

using dataset=std::vector<std::pair<std::vector<double>,std::vector<double>>>; //A dataset is a vector of pairs of training input and output layers
void get_data(std::ifstream& data_file, dataset& data, const int& input_size, const int& output_size){
	std::vector<double> input(input_size),output(output_size);
	std::string l1,l2;
	while(std::getline(data_file,l1) && std::getline(data_file,l2)){
		read(l1,input); read(l2,output);
		data.push_back({input,output});
	}
}

void split_data(const dataset& data, dataset& training, dataset& testing, const int& batches=10, const float& part_testing=0.2){
	int testing_size=data.size()*part_testing;
	while((data.size()-testing_size)%batches!=0) testing_size++; //Make sure the training data can be split into batches evenly
	int batch_size=(data.size()-testing_size)/batches;
	for(int i=0;i<testing_size;i++)
		testing.push_back(data[i]);
	for(int i=testing_size;i<data.size();i++)
		training.push_back(data[i]);
	std::cout<<"\nSplitting data...\n"<<
	" Testing set: "<<testing_size<<" / "<<data.size()<<" ("<<100.0f*testing.size()/data.size()<<"%)\n"<<
	" Training set: "<<batches<<'x'<<batch_size<<" / "<<data.size()<<" ("<<100.0f*training.size()/data.size()<<"%)\n\n\n";
}

double d_loss(const node& expected, const node& output){
	return 2*(output.value-expected.value); //Derivative of the loss function with respect to the output
}
double activation(const double& x){
	return 2./(1.+exp(-x))-1.; //Sigmoid activation function (from -1 to 1)
}
double d_activation(const double& x){
	return 0.5*(1.+x)*(1.-x); //Derivative of the (modified) sigmoid activation function
}

void forward_propagation(NN& net){
	for(int i=1;i<=N+1;i++)
		for(int j=0;j<net[i].size(); j++){
			net[i][j].value=net[i][j].bias;
			for(node& left:net[i-1])
				net[i][j].value+=left.weights[j]*left.value;
			net[i][j].value=activation(net[i][j].value);
		}
}

void backward_propagation(NN& net, layer& expected_output){
	//Calculate the delta for the output layer
	for(int j=0; j<net[N+1].size(); j++){
		net[N+1][j].delta=d_loss(expected_output[j],net[N+1][j])*d_activation(net[N+1][j].value);
		net[N+1][j].grad_b+=net[N+1][j].delta;
	}

	//Backward pass
	double d_act;
	for(int i=N; i>=0; i--)
		for(node& n:net[i]){
			n.delta=0;
			d_act=d_activation(n.value);
			for(int k=0;k<net[i+1].size(); k++){
				n.delta+=net[i+1][k].delta*d_act*n.weights[k];
				n.grad_w[k]+=net[i+1][k].delta*n.value;
			}
			n.grad_b+=n.delta;
		}
}

void update(NN& net,const int& batch_size){
	const double LR_=LR/batch_size;
	for(int i=0; i<=N+1; i++)
		for(node& n:net[i]){
			n.bias-=LR_*n.grad_b;
			n.grad_b=0.0;
			for(int k=0; k<n.weights.size(); k++){
				n.weights[k]-=LR_*n.grad_w[k];
				n.grad_w[k]=0.0;
			}
		}
}

void inference(NN& net, std::ifstream& input_file, std::ofstream& output_file){
	//Now we can use the trained NN to make predictions on new data
	std::cout<<"Making predictions on data from input file... Saving to output file...\n";
	std::string line;
	while(std::getline(input_file,line)){
		readl(line,net[0]);
		forward_propagation(net);

		for(node& n:net[N+1])
			output_file<<n.value<<" ";
		output_file<<"\n";
	}
}

double get_loss(NN& net, dataset& data){
	layer expected_output(net.back().size());
	int cnt=0; double total_loss=0;
	for(const auto& data_point:data){
		cnt++;
		for(int j=0; j<data_point.first.size(); j++)
			net[0][j].value=data_point.first[j];
		for(int j=0; j<data_point.second.size(); j++)
			expected_output[j].value=data_point.second[j];
		forward_propagation(net);
		for(int j=0;j<net[N+1].size();j++)
			total_loss+=(net[N+1][j].value-expected_output[j].value)*(net[N+1][j].value-expected_output[j].value);
	}
	return (cnt==0) ? 0 : total_loss/cnt;
}

void train(NN& net, dataset& training, const int& batches=10, const int& print=5000){
	double last_loss=get_loss(net,training),curr_loss;
	std::cout<<"Training neural network on "<<training.size()<<" data points. Initial loss: "<<last_loss<<"\n";
	layer expected_output(net.back().size());
	int epochs,batch_size=training.size()/batches;

    auto start = std::chrono::high_resolution_clock::now(), last_time = std::chrono::high_resolution_clock::now(), curr_time = std::chrono::high_resolution_clock::now();

	while(true){
		std::cout<<"How many epochs to train the NN for: "; std::cin>>epochs;
		if(epochs<=0)break;

        start = std::chrono::high_resolution_clock::now();
        last_time = std::chrono::high_resolution_clock::now();
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(curr_time - last_time);
        int epochs_at_last_print = epochs;

		while(epochs--){

			if(epochs%5==0){
                curr_loss=get_loss(net,training);
                if(curr_loss>1.025*last_loss)
                    LR*=0.975; //If the loss increased, the training is unstable. Reduce the learning rate.
                last_loss=curr_loss;
            }

            if (print != 0) // The program should print interim status reports every print milliseconds{
            {
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
            }

			std::shuffle(training.begin(), training.end(), RNG); //Shuffle the order of the training data for this epoch
			for(int i=0; i<training.size(); i++){
				for(int j=0; j<training[i].first.size(); j++)
					net[0][j].value=training[i].first[j];
				for(int j=0; j<training[i].second.size(); j++)
					expected_output[j].value=training[i].second[j];
				forward_propagation(net);
				backward_propagation(net,expected_output);
				if(i%batch_size==batch_size-1 || i==training.size()-1) //Update the weights and biases after each batch
					update(net,batch_size);
			}
		}

		last_loss=get_loss(net,training);
		std::cout<<"Training complete. Loss: "<<last_loss<<"\n\n";
	}
}

void create_nn(NN& net,std::ifstream& data_file){
	std::cout<<"Creating neural network... Define its size <depth N> <layer width M>: "; std::cin>>N>>M;
	bound = 1.0/sqrt(M); //Heuristic for weight initialization
	net.resize(N+2);
	fixLayer(net[0], data_file); fixLayer(net[N+1], data_file); //Fix the input and output layers to have nodes corresponding to data dimensions
	data_file.clear(); data_file.seekg(0);
	for(int i=1;i<=N;i++) net[i].resize(M);
	for(node& n:net[N]){
		n.weights.resize(net[N+1].size());
		n.grad_w.resize(net[N+1].size());
	}
	for(node& n:net[N+1]){
		n.weights.resize(0);
		n.grad_w.resize(0);
	}
}

void load_network(NN& net, std::ifstream& load_file){
	std::cout<<"Loading neural network from file...\n";
	std::string line;

	std::getline(load_file,line);
	std::istringstream in(line);
	in>>N; in>>M;
	net.resize(N+2);
	int temp=0; in>>temp;
	net[0].resize(temp);
	temp=0; in>>temp;
	net[N+1].resize(temp);
	std::cout<<"Network structure: "<<N<<" hidden layers, "<<M<<" nodes per layer.  |  Input layer: "<<net[0].size()<<" nodes, output layer: "<<net[N+1].size()<<" nodes.\n";

	//layer = 0
	for(node& n:net[0]){
		std::getline(load_file,line);
		in.clear(); in.str(line);
		for(int j=0;j<M;j++)
			in>>n.weights[j];
		in>>n.bias;
	}

	//layer = 1..N-1
	for(int i=1; i<N; i++){
		net[i].resize(M);
		for(node& n:net[i]){
			std::getline(load_file,line);
			in.clear(); in.str(line);
			for(int k=0;k<M;k++)
				in>>n.weights[k];
			in>>n.bias;
		}
	}

	//layer = N
	net[N].resize(M);
	for(node& n:net[N]){
		n.weights.resize(net[N+1].size());
		n.grad_w.resize(net[N+1].size());
		std::getline(load_file,line);
		in.clear(); in.str(line);
		for(int j=0;j<n.weights.size();j++)
			in>>n.weights[j];
		in>>n.bias;
	}

	//layer = N+1
	for(node& n:net[N+1]){
		std::getline(load_file,line);
		in.clear(); in.str(line);
		n.weights.resize(0);
		n.grad_w.resize(0);
		in>>n.bias;
	}
}

void save_network(const NN& net, std::ofstream& save_file){
	std::cout<<"Saving NN to file...\n";
	save_file<<N<<' '<<M<<' '<<net[0].size()<<' '<<net[N+1].size()<<"\n"; //Save the network structure
	save_file.precision(10);
	for(const layer& l:net)
		for(const node& n:l){
			for(double w:n.weights)
				save_file<<w<<' ';
			save_file<<n.bias<<"\n";
		}
}


int main(int argc, char** argv){
    auto start = std::chrono::system_clock::now();
    std::cout << "\n\n\n";

	int whereToSave=0,batches=25, print=5000; float part_testing=0.2;
	//std::string activation,loss;
	std::ifstream data_file,load_file,input_file;
	std::ofstream output_file,save_file;
	initialize(argc,argv,data_file,load_file,input_file,output_file,whereToSave,batches,part_testing,print);
	dataset data,training,testing; NN net;

	if(load_file.is_open())
		load_network(net,load_file);
	else if(data_file.is_open())
		create_nn(net,data_file);
	else{
		std::cerr<<"Error: no data file or load file specified. Failed making an NN.\n";
		exit(-1);
	}
	if(LR==-1) LR= (N>1) ? 0.5/N : 0.5;
	std::cout<<'\n';

	bool did_something=false;
	if(data_file.is_open()){
		get_data(data_file,data,net[0].size(),net[N+1].size());
		//std::shuffle(data.begin(), data.end(), RNG);
		if(batches==0 || batches>data.size()*(1.0f-part_testing)) batches=data.size()*(1.0f-part_testing); //batches=0 <=> user wants SGD
		split_data(data,training,testing,batches,part_testing);
		std::cout<<std::setprecision(5)<<std::scientific;
		train(net,training,batches,print);
		did_something=true;
		std::cout<<"\nTraining Complete!\nLoss on testing dataset: "<<get_loss(net,testing)<<"\n\n";
	}
	std::cout<<'\n';


	if(input_file.is_open()){
		if(!output_file.is_open()) output_file.open("output.txt");
		inference(net,input_file,output_file);
	}else
		std::cout<<"No input file specified. Skipping inference.\n\n";
	std::cout<<'\n';

	//Close files
	data_file.close();
	load_file.close();
	input_file.close();
	output_file.close();

	//Save the trained NN to a file
	if(whereToSave!=0)save_file.open(argv[whereToSave]);
	else if(did_something) save_file.open("trained_net.txt");
	if(save_file.is_open()) save_network(net, save_file);
	else if(did_something) std::cerr<<"Warning: save file inaccessible. NN not saved.\n";
	save_file.close();

    std::cout << "\nProgram Time: " << std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now() - start).count() << "s\n\n\n";
    return 0;
}
