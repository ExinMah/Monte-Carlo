#include <iostream>
#include <random>
#include <vector>
#include <cmath>
#include <chrono>

// CUDA includes
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand.h>
#include <curand_kernel.h>

using namespace std;
using namespace chrono;

// Function to integrate: f(x) = (1 + r)^(-t)
__device__ double functionToIntegrate(double x) {
    double r = 0.08; // Annual discount rate 
    double t = x; // Time in years 

    return pow(1.0 + r, -t);
}

__global__ void monteCarloIntegrationKernel(double* results, double lowerBound, double upperBound, int numSamples, unsigned int seed, double* totalSum) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Initialize random number generator
    curandState_t state;
    curand_init(seed, idx, 0, &state);

    double partialSum = 0.0;
    // Perform Monte Carlo integration for the assigned samples
    for (int i = idx; i < numSamples; i += blockDim.x * gridDim.x) {
        double x = lowerBound + (upperBound - lowerBound) * curand_uniform(&state);
        partialSum += functionToIntegrate(x);
    }

    // Store the partial sum in the results array
    results[idx] = partialSum;

    // Synchronize all threads to ensure all partial sums are stored
    __syncthreads();

    // Perform reduction to calculate the total sum
    if (idx == 0) {
        double total = 0.0;
        for (int i = 0; i < blockDim.x * gridDim.x; ++i) {
            total += results[i];
        }
        *totalSum = total;
    }
}

__global__ void performSimulationsKernel(double* returns, double initialInvestment, double expectedReturn, double volatility,
                                         int numSimulations, int investmentPeriod, unsigned int seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Initialize random number generator
    curandState_t state;
    curand_init(seed, idx, 0, &state);

    // Perform simulations for the assigned index
    for (int i = idx; i < numSimulations; i += blockDim.x * gridDim.x) {
        double cumulativeReturn = 1.0;

        for (int j = 0; j < investmentPeriod; ++j) {
            double randomValue = curand_normal(&state);
            double investmentReturn = expectedReturn + volatility * randomValue;
            cumulativeReturn *= (1.0 + investmentReturn);
        }

        returns[i] = cumulativeReturn - 1.0;
    }
}

void calculateRiskMetrics(const vector<double>& finalReturns, double& averageReturn, double& standardDeviation,
                          double& minReturn, double& maxReturn) {
    double totalReturns = 0.0;
    minReturn = finalReturns[0];
    maxReturn = finalReturns[0];

    for (const auto& returnVal : finalReturns) {
        totalReturns += returnVal;
        if (returnVal < minReturn) {
            minReturn = returnVal;
        }
        if (returnVal > maxReturn) {
            maxReturn = returnVal;
        }
    }

    averageReturn = totalReturns / finalReturns.size();

    double variance = 0.0;
    for (const auto& returnVal : finalReturns) {
        variance += (returnVal - averageReturn) * (returnVal - averageReturn);
    }

    variance /= (finalReturns.size() - 1);
    standardDeviation = sqrt(variance);
}

double calcAverage(double* array, int iterations)
{
    double sum = 0.0;
    for (int i = 0; i < iterations; i++) {
        sum += array[iterations];
    }
    
    return sum;
}

void displayResults(double initialInvestment, double expectedReturn, double volatility, int investmentPeriod,
                    int numSimulations, double averageReturn, double standardDeviation, double minReturn,
                    double maxReturn, double integrationResult, double elapsedTime) {
    cout << "Risk Assessment Results:" << endl;
    cout << "--------------------------------------" << endl;
    cout << "Initial Investment: $" << initialInvestment << endl;
    cout << "Expected Annual Return: " << (expectedReturn * 100.0) << "%" << endl;
    cout << "Volatility (Annual Standard Deviation): " << (volatility * 100.0) << "%" << endl;
    cout << "Investment Period: " << investmentPeriod << " years" << endl;
    cout << "Number of Simulations: " << numSimulations << endl;
    cout << "--------------------------------------" << endl;
    cout << "Average Return: " << (averageReturn * 100.0) << "%" << endl;
    cout << "Standard Deviation of Returns: " << (standardDeviation * 100.0) << "%" << endl;
    cout << "Minimum Return: " << (minReturn * 100.0) << "%" << endl;
    cout << "Maximum Return: " << (maxReturn * 100.0) << "%" << endl;
    cout << "Monte Carlo Integration Result: " << integrationResult << endl;
    cout << "Elapsed Time: " << elapsedTime << " seconds" << endl;
}

int main() {
    // Define your input parameters
    double initialInvestment = 10000.0;
    double expectedReturn = 0.08;
    double volatility = 0.15;
    int investmentPeriod = 5;
    int numSimulations = 1000000;
    double lowerBound = 0.0;
    double upperBound = 1;
    int numThreads = 8;

    // Allocate memory for results
    double* returns = new double[numSimulations];
    double* integral = new double[numSimulations];

    // Set up CUDA memory
    double* d_returns;
    double* d_integral;
    cudaMalloc((void**)&d_returns, numSimulations * sizeof(double));
    cudaMalloc((void**)&d_integral, numSimulations * sizeof(double));

    double* integralResult;
    cudaMalloc((void**)&integralResult, sizeof(double));

    // Set up CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Launch CUDA kernels in parallel
    cudaEventRecord(start);

    // Create an array of CUDA streams
    cudaStream_t* streams = new cudaStream_t[numThreads];

    unsigned int seed = static_cast<unsigned int>(time(nullptr));
    // Launch parallel simulations and integration kernels using separate streams
    for (int i = 0; i < numThreads; ++i) {
        // Create a new CUDA stream for the current thread
        cudaStreamCreate(&streams[i]);

        // Perform simulations on the device (GPU) using the current stream
        performSimulationsKernel << <1, 256, 0, streams[i] >> > (d_returns + i * numSimulations / numThreads,
            initialInvestment, expectedReturn, volatility,
            numSimulations / numThreads, investmentPeriod, seed + i);

        // Launch Monte Carlo integration kernel on the device (GPU) using the current stream
        monteCarloIntegrationKernel << < 1, 256, 0, streams[i] >> > (d_integral + i * numSimulations / numThreads,
            lowerBound, upperBound, numSimulations / numThreads, seed + i, integralResult);
    }

    // Synchronize all CUDA streams before copying results back to the host
    for (int i = 0; i < numThreads; ++i) {
        cudaStreamSynchronize(streams[i]);
    }
    
    // Copy results from device to host
    cudaMemcpy(returns, d_returns, numSimulations * sizeof(double), cudaMemcpyDeviceToHost);
    //cudaMemcpy(integral, d_integral, numSimulations * sizeof(double), cudaMemcpyDeviceToHost);
    double totalSum;
    cudaMemcpy(&totalSum, integralResult, sizeof(double), cudaMemcpyDeviceToHost);
    double avgIntegral = (totalSum / numSimulations) * (upperBound - lowerBound);

    // Calculate risk metrics on the host (CPU)
    double averageReturn, standardDeviation, minReturn, maxReturn;
    calculateRiskMetrics(vector<double>(returns, returns + numSimulations), averageReturn, standardDeviation,
        minReturn, maxReturn);

    // Record stop time and calculate elapsed time
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    // CUDA Calculate Time in ms, therefore divide by 1000
    elapsedTime /= 1000;

    // Display results
    displayResults(initialInvestment, expectedReturn, volatility, investmentPeriod, numSimulations,
        averageReturn, standardDeviation, minReturn, maxReturn, avgIntegral,
        elapsedTime);

    // Clean up CUDA resources
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_returns);
    delete[] returns;
    delete[] streams;

    system("pause");

    return 0;
}