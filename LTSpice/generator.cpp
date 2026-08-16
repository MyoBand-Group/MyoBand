/* Author: Nikola D. Lilov, 23.VII.2026
 *
 * Desc: Automatic iterator for LTspice simulations. It takes a base simulation file and 
 * iterates over a chosen component's value from a starting value to a target value in a 
 * specified number of steps. It generates .asc simulation files for each step and runs 
 * them in LTspice. Thus there should be many .asc, .log, .raw, and .net files generated
 * in the folder. The .raw files can be interpreted in LTspice or using interpreter.py.
 *
 * Usage: generator.exe <template_path.asc> [write_path] [LTspice_path.exe] */

/* TODO:
 * Add support for different iteration types (linear, logarithmic, etc).
 * Add support for iteration over component lists. */

#include<iostream>
#include<fstream>
#include<sstream>
#include<string>
#include<vector>
#include<algorithm>
#include<cstdlib>

/* Find where LTspice.exe is installed. It is usually in one of the following locations:
 * 1. C:\Users\<username>\AppData\Local\Programs\ADI\LTspice\LTspice.exe
 * 2. C:\Program Files\ADI\LTspice\LTspice.exe
 * 3. C:\Program Files (x86)\ADI\LTspice\LTspice.exe */
#include <windows.h>
#include <filesystem>
#include <vector>
namespace fs = std::filesystem;
std::string findLTspice(){
	std::vector<fs::path> candidates;

	std::string user = getenv("USERPROFILE");
	std::string programFiles = getenv("ProgramFiles");
	std::string programFiles86 = getenv("ProgramFiles(x86)");

	if(!user.empty())
		candidates.push_back(fs::path(user)/"AppData\\Local\\Programs\\ADI\\LTspice\\LTspice.exe");
	if(!programFiles.empty())
		candidates.push_back(fs::path(programFiles)/"ADI\\LTspice\\LTspice.exe");
	if(!programFiles86.empty())
		candidates.push_back(fs::path(programFiles86)/"ADI\\LTspice\\LTspice.exe");

	for(const auto& path : candidates)
		if(fs::exists(path))
			return path.string();

	return "";
}

struct component{
	std::string name;
	std::string value;
	int where; // line in the file where the value to be edited is located
};

struct iterationSpec{
	size_t componentIndex;
	std::string prefix;
	std::string suffix;
	std::string leading;
	std::string trailing;
	std::string targetLeading;
	std::string targetTrailing;
	int steps;
	double currentValue;
	double lastValue;
	double delta;
};

void writeSimulationFile(const std::string& templatePath,
	const std::vector<component>& components,
	const std::vector<iterationSpec>& specs,
	const std::vector<double>& currentValues,
	const fs::path& outputPath){
	std::ifstream templateStream(templatePath);
	std::ofstream outputStream(outputPath);

	if(!templateStream.is_open()){
		std::cerr << "Error: Could not open template ("<<templatePath<<").\n";
		return;
	}
	if(!outputStream.is_open()){
		std::cerr << "Error: Could not create output file ("<<outputPath.string()<<").\n";
		return;
	}

	std::string line;
	for(int lineIndex = 0; std::getline(templateStream, line); ++lineIndex){
		bool replaced = false;
		for(size_t i = 0; i < specs.size(); ++i){
			if(lineIndex == components[specs[i].componentIndex].where){
				const iterationSpec& spec = specs[i];
				std::string newValue = spec.prefix + spec.leading + std::to_string(currentValues[i]) + spec.trailing + spec.suffix;
				outputStream << "SYMATTR Value " << newValue << "\n";
				replaced = true;
				break;
			}
		}
		if(!replaced)
			outputStream << line << "\n";
	}
	outputStream.close();
}

void recurseIterations(const std::string& templatePath,
	const std::vector<component>& components,
	const std::vector<iterationSpec>& specs,
	const fs::path& outputDir,
	const std::string& baseName,
	const std::string& ltspicePath,
	size_t depth,
	std::vector<double>& currentValues,
	std::vector<int>& indexValues){
	if(depth == specs.size()){
		std::string outputName = baseName;
		for(const auto& index : indexValues)
			outputName += "_" + std::to_string(index);
		outputName += ".asc";

		fs::path outputPath = outputDir / outputName;
		writeSimulationFile(templatePath, components, specs, currentValues, outputPath);

		std::string command = "\"\"" + ltspicePath + "\" -Run -b \"" + outputPath.string() + "\"\"";
		std::system(command.c_str());
		std::cout << "Running command: \"LTspice.exe -Run -b " << outputName << "\"\n";
		return;
	}

	const iterationSpec& spec = specs[depth];
	for(int step = 0; step <= spec.steps; ++step){
		currentValues[depth] = spec.currentValue + step * spec.delta;
		indexValues[depth] = step;
		recurseIterations(templatePath, components, specs, outputDir, baseName, ltspicePath, depth + 1, currentValues, indexValues);
	}
}

bool askYesNo(const std::string& prompt){
	std::cout << prompt;
	std::string answer;
	while(true){
		std::getline(std::cin, answer);
		if(answer == "y" || answer == "Y" || answer == "yes" || answer == "Yes")
			return true;
		if(answer == "n" || answer == "N" || answer == "no" || answer == "No")
			return false;
	}
}

int main(int argc, char* argv[]){

	std::cout<<'\n';

	if(argc < 2 || argc > 4){
		std::cerr << "Usage:   "<<argv[0]<<" <template_path.asc> [write_path] [LTspice_path.exe] \n";
		return 1;
	}
	std::string LTSPICE;
	if(argc == 4)
		LTSPICE = argv[3];
	else{
		LTSPICE = findLTspice();
		if(LTSPICE.empty()){
			std::cerr << "Error: Could not find LTspice.exe.\nPlease provide its path as a third argument:\nUsage:   "<<argv[0]<<" <template_path.asc> [write_path] [LTspice_path.exe]\n";
			while(LTSPICE.empty()) std::getline(std::cin, LTSPICE);
			return 1;
		}
	}

	fs::path absoluteTemplatePath = fs::absolute(argv[1]);
	fs::path executablePath = fs::absolute(argv[0]);
	fs::path interpreterPath = executablePath.parent_path() / "interpreter.py";
	std::string baseSimFile = absoluteTemplatePath.string();
	std::ifstream baseSim(baseSimFile);

	if(!baseSim.is_open()){
		std::cerr << "Error: Could not open template ("<<baseSimFile<<").\n";
		return 1;
	}

	fs::path absoluteLtspicePath;
	if(argc == 4){
		absoluteLtspicePath = fs::absolute(argv[3]);
		LTSPICE = absoluteLtspicePath.string();
	}

	// Make a directory, so that the generated files don't clutter the current folder.
	std::string generatedDirName = absoluteTemplatePath.string();
	if(generatedDirName.size() >= 4 && generatedDirName.substr(generatedDirName.size() - 4) == ".asc")
		generatedDirName = generatedDirName.substr(0, generatedDirName.size() - 4);
	if(argc >= 3) generatedDirName = argv[2];
	else generatedDirName += "_generated";
	if(std::filesystem::exists(generatedDirName)){
		if(askYesNo("Warning: The directory " + generatedDirName + " already exists. Overwrite it? (y/n): "))
			std::filesystem::remove_all(generatedDirName);
		else{
			std::cerr << "Aborting.\n";
			return 1;
		}
		std::cout<<'\n';
	}std::filesystem::create_directory(generatedDirName);

	// Read line by line and find the components to iterate over: "SYMATTR InstName [name]", followed by "SYMATTR Value [value]".
	std::string line;
	std::vector<component> comp;
	for(size_t i=0; std::getline(baseSim, line); i++)
		if(line.find("SYMATTR InstName") != std::string::npos){
			comp.push_back(component());

			//get the name - following "SYMATTR InstName ", aka. the substring starting at line[17].
			comp.back().name = line.substr(17);
			
			//the vallue always follows and it's at line[14] (after "SYMATTR Value ").
			std::getline(baseSim, line); i++;
			comp.back().value = line.substr(14);

			comp.back().where = i;
		}

	if(comp.empty()){
		std::cerr << "Error: No components found in the template file ("<<argv[1]<<").\n";
		return 1;
	}
	line.clear();

	//sort alphabetically by name, so that the user can easily find the component to iterate over.
	//std::sort(comp.begin(), comp.end(), [](const component& a, const component& b){ return a.name < b.name; });

	std::cout<<"Choose which component(s) to iterate over:\n";
	for(size_t i=0; i<comp.size(); i++)
		std::cout<<i<<": "<<comp[i].name<<" = "<<comp[i].value<<"\n";
	std::cout<<"Enter the number(s) of the component(s) to iterate over (separate with spaces) (0 - "<<comp.size()-1<<"): ";
	std::vector<int> choices; while(line.empty()) std::getline(std::cin,line);
	std::stringstream ss(line);
	int choice;
	while(ss >> choice)
		choices.push_back(choice);
	// Ask for the starting and target values for the chosen component. There should be only one differing value between the two.
	std::vector<std::string> start(choices.size()), target(choices.size());
	std::cout<<std::endl<<"The starting and the target values for each iterator should be in the same units:\n";
	for(size_t i=0; i<choices.size(); i++){
		while(choices[i] < 0 || choices[i] >= comp.size()){
			std::cerr << "Error: Invalid choice ("<<choices[i]<<"). Must choose from [0 - "<<comp.size()-1<<"].\n";
			return 1;
		}
		std::cout<<"\nFor component "<<comp[choices[i]].name<<" = "<<comp[choices[i]].value<<":\n";
		std::cout<<"Starting value:\t"; while(start[i].empty())std::getline(std::cin, start[i]);
		std::cout<<"Target value:\t"; while(target[i].empty())std::getline(std::cin, target[i]);
	}


	//parse the start and target values to find the differing character and its position. start by parsing all the "words".
	std::vector<std::string> startWords,targetWords;
	std::vector<iterationSpec> specs;
	specs.reserve(choices.size());
	size_t prev=0;
	std::vector<std::string> prefix(choices.size()), suffix(choices.size()),
	leading(choices.size()),targetLeading(choices.size()),trailing(choices.size()),targetTrailing(choices.size());
	std::vector<int> steps(choices.size());
	std::vector<size_t> trailPos(choices.size()),targetTrailPos(choices.size());
	std::vector<double> curr(choices.size()),last(choices.size()),delta(choices.size());

	for(size_t j=0; j<choices.size(); j++){
		startWords.clear(); targetWords.clear(); prev = 0;
		for(size_t i=0; i<start[j].size(); i++)
			if(start[j][i] == ' '){
				startWords.push_back(start[j].substr(prev,i-prev));
				prev = i+1;
			}
		startWords.push_back(start[j].substr(prev,start[j].size())); //add the last word (not followed by a space).
		prev = 0;
		for(size_t i=0; i<target[j].size(); i++)
			if(target[j][i] == ' '){
				targetWords.push_back(target[j].substr(prev,i-prev));
				prev = i+1;
			}
		targetWords.push_back(target[j].substr(prev,target[j].size())); //add the last word (not followed by a space).
		
		//check if the difference is valid - there should be only one differing word between the two, and they should have the same units. If the difference is valid, find the differing word and its position.
		if(startWords.size() != targetWords.size()){
			std::cerr << "Error: The starting and target values have different formats.\n";
			return 1;
		}
		//prefix[j],suffix[j] - The wanted change is one differing value and therefore everything before and after it should be the same.
		int diffPos = -1;
		for(size_t i=0; i<startWords.size() && i<targetWords.size(); i++)
			if(startWords[i] != targetWords[i]){
				if(diffPos != -1){
					std::cerr << "Error: More than one differing word between the starting and target values.\n";
					return 1;
				}
				diffPos = i;
			}else
				if(diffPos == -1)
					prefix[j] += startWords[i] + " ";
				else
					suffix[j] += " " + startWords[i];

		if(diffPos == -1){
			std::cerr << "Error: The starting and target values are the same.\n";
			return 1;
		}

		//Get any leading characters from the differing word. It is the non-number prefix of both start and target.
		leading[j] = startWords[diffPos].substr(0,startWords[diffPos].find_first_of("0123456789.e"));
		targetLeading[j] = targetWords[diffPos].substr(0,targetWords[diffPos].find_first_of("0123456789.e"));
		if(leading[j] != targetLeading[j]){
			std::cerr << "Error: The starting and target values have different leading characters.\n";
			return 1;
		}

		//Get the units (K, m, Meg, etc) from the end. It is the non-number suffix of both start and target.
		trailPos[j] = startWords[diffPos].find_first_not_of("0123456789.e", leading[j].size());
		trailing[j] = (trailPos[j] == std::string::npos) ? "" : startWords[diffPos].substr(trailPos[j]);
		targetTrailPos[j] = targetWords[diffPos].find_first_not_of("0123456789.e", targetLeading[j].size());
		targetTrailing[j] = (targetTrailPos[j] == std::string::npos) ? "" : targetWords[diffPos].substr(targetTrailPos[j]);
		if(trailing[j] != targetTrailing[j]){
			std::cerr << "Error: The starting and target values have different units.\n"<<trailing[j]<<" vs "<<targetTrailing[j]<<"\n";
			return 1;
		}	
		
		std::cout<<'\n'<<comp[choices[j]].name<<": How many steps to iterate from "<<start[j]<<" to "<<target[j]<<"? "; std::cin>>steps[j]; std::cout<<"\n";
		if(steps[j] <= 0){
			std::cerr << "Error: The number of steps must be greater than zero.\n";
			return 1;
		}

		// Generate the new simulations by replacing the chosen component's value with the iterated values.
		curr[j] = std::stod(startWords[diffPos].substr(leading[j].size(),startWords[diffPos].size()-leading[j].size()-trailing[j].size()));
		last[j] = std::stod(targetWords[diffPos].substr(targetLeading[j].size(),targetWords[diffPos].size()-targetLeading[j].size()-targetTrailing[j].size()));
		delta[j]=(last[j]-curr[j]) / steps[j];

		iterationSpec spec;
		spec.componentIndex = choices[j];
		spec.prefix = prefix[j];
		spec.suffix = suffix[j];
		spec.leading = leading[j];
		spec.trailing = trailing[j];
		spec.targetLeading = targetLeading[j];
		spec.targetTrailing = targetTrailing[j];
		spec.steps = steps[j];
		spec.currentValue = curr[j];
		spec.lastValue = last[j];
		spec.delta = delta[j];
		specs.push_back(spec);
	}

	std::vector<double> currentValues(specs.size(), 0.0);
	std::vector<int> indexValues(specs.size(), 0);
	std::string baseName = fs::path(baseSimFile).stem().string();

	recurseIterations(baseSimFile, comp, specs, fs::path(generatedDirName), baseName, LTSPICE, 0, currentValues, indexValues);

	std::cout << "\nGeneration complete.\n";
	if(askYesNo("Run the interpreter on the generated .raw files? (y/n): ")){
		std::string interpreterCommand = "python \"" + interpreterPath.string() + "\" \"" + generatedDirName + "\" \"" + baseName + "\"";
		std::cout << "Running command: " << interpreterCommand << "\n\n\n";
		std::system(interpreterCommand.c_str());
	}

	return 0;
}
