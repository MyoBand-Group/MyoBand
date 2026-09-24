#include<fstream>
#include<iostream>
#include<sstream>
#include<cmath>
#include<ctime>
#include<cstdlib>
#include<string>
#include<vector>

using namespace std;

#include "json.hpp"
using json = nlohmann::json;

float Lr=0.25;
 int types=4;
json readJsonFile(const std::string& filepath) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + filepath);
    }

    json data;
    file >> data;
    return data;
}
float dLoss(float expect,float actual,int types){
float d=2*(actual-expect);
if(actual<0||actual>types){d+=actual*2-types;}
return d;
}
float Loss(float expect,float actual,int types){
float x=(actual-expect)*(actual-expect);
if(actual<0||actual>types){x+=actual*actual-types*actual;}
return x;
}
double activation(const double& x){
	// Clamp x to prevent overflow
	double clamped = (x > 100) ? 100 : (x < -100) ? -100 : x;
	return 4./(1.+exp(-clamped)); //Sigmoid activation function (from 0 to 4)
}
double d_activation(const double& activated_output){
	// Derivative: dy/dx = y(4-y)/4 where y is the activated output
	return activated_output * (4. - activated_output) / 4.;
}

void forwardProp(vector<vector<vector<float>>>& w, vector<vector<float>>& node, vector<vector<float>>& bias, vector<float>& wform){
    
for(int j=0;j<(int)node[0].size();j++){
	node[0][j]=wform[j];
}

for(int i=0; i<(int)node.size()-1;i++){
    for(int k=0;k<(int)node[i+1].size();k++){
        double sum = bias[i+1][k];  // Use double for accumulation
        for(int j=0;j<(int)node[i].size();j++){
            sum += node[i][j]*w[i][j][k];
        }
        node[i+1][k] = (float)activation(sum);  // Apply activation with double precision
    }
}

}

void backwardProp(vector<vector<vector<float>>>& w, vector<vector<float>>& node, vector<vector<float>>& bias,vector<vector<float>>&d,vector<float> e){
int n=node.size()-1; 
    for(int j=0;j<(int)node[n].size();j++){
d[n][j]=dLoss(e[j],node[n][j],types)*d_activation(node[n][j]);
bias[n][j]-=Lr*2*(node[n][j]-e[j]);
}
for(int i=n; i>0;i--){
    for(int k=0;k<(int)node[i-1].size();k++){
        d[i-1][k]=0;
        for(int j=0;j<(int)node[i].size();j++){
        d[i-1][k]+=d[i][j]*w[i-1][k][j]*d_activation(node[i-1][j]);
        }}}
for(int i=0; i<(int)node.size()-1;i++){
    for(int j=0;j<(int)node[i].size();j++){
        for(int k=0;k<(int)node[i+1].size();k++){
        w[i][j][k]-=Lr*node[i][j]*d[i+1][k];
        bias[i][j]-=Lr*d[i][j];    
        }}}
}

float totalLoss(int currdata, vector<vector<float>>& node,vector<float>& amp, vector<float>& wave){
    float x=0;
    x+=Loss(amp[currdata],node[node.size()-1][0],types);    
    x+=Loss(wave[currdata],node[node.size()-1][1],types);
    return x; 
}

void Train (vector<vector<vector<float>>>& w, vector<vector<float>>& node, vector<vector<float>>& bias,vector<vector<float>>&d,vector<float>& wave,vector<vector<float>>& wform,vector<float>& amp,int epochs=10){
   float currLoss=0;
   float lastLoss=0;
   int q=0;
    forwardProp(w,node,bias,wform[0]);
    lastLoss=totalLoss(0,node,amp,wave);
    for(int i=0;i<epochs;i++){
        if(i%5==0){
            currLoss=totalLoss(q,node,amp,wave);
            if(currLoss>1.05*lastLoss){
                Lr*=0.9;
            } 
            lastLoss=currLoss;
        }
        for(q=0; q< amp.size();q++){
        forwardProp(w,node,bias,wform[q]);
        backwardProp(w,node,bias,d,{amp[q],wave[q]});
        
    }
   }
   lastLoss=totalLoss(0,node,amp,wave);
   cout<<"  loss : "<<totalLoss(q,node,amp,wave);
}

int main(){
	int N;
	int M;
	cin >> N >> M;
	vector<vector<vector<float>>> weights(N+2, vector<vector<float>>(M, vector<float>(M,1)));
    vector<vector<float>> nodes(N+2, vector<float>(M, 1));
    vector<vector<float>> biases(N+2, vector<float>(M, 1));
    vector<vector<float>> deltas(N+2, vector<float>(M, 1));
    nodes[N+1].resize(2);
    biases[N+1].resize(2);
    vector<float>waves;
    vector<vector<float>>wform;
    vector<float>amp;

	cout << weights[0][0][0];
    json data;
     try {
        data = readJsonFile("data.json");
        nodes[0].resize((int)data[0]["wform"].size());
        biases[0].resize((int)data[0]["wform"].size());
        weights[0].resize((int)data[0]["wform"].size());
        for(int j=0;j<nodes[0].size();j++){
        weights[0][j]={};
        for(int k=0;k<(int)nodes[1].size();k++){
        weights[0][j].push_back(1);
        } 
}
for(int i=0;i<data.size();i++){
            waves.push_back(data[i]["wave"]);
            amp.push_back(data[i]["amp"]);
            wform.push_back({});
         for(int j=0;j<data[i]["wform"].size();j++){
            wform[i].push_back(data[i]["wform"][j]);
         }   
         
        }
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }
    Train(weights,nodes,biases,deltas,waves,wform,amp,10000);
	return 0;
}
