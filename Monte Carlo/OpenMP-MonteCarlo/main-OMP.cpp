#include <iostream>
#include <random>
#include <vector>
#include <cmath>
#include <chrono>
#include <omp.h>

#define NUM_THREADS 8

using namespace std;
using namespace chrono;

// Function to integrate: f(x) = (1 + r)^(-t)
double functionToIntegrate(double x) {
    double r = 0.08; // Annual discount rate 
    double t = x; // Time in years 

    return pow(1.0 + r, -t);
}

double monteCarloIntegration(double (*function)(double), double lowerBound, double upperBound, int numSamples) {
    random_device rd;
    mt19937 generator(rd());
    uniform_real_distribution<double> distribution(lowerBound, upperBound);

    double sum = 0.0;

#pragma omp parallel num_threads(NUM_THREADS) reduction(+:sum)
    {
        int threadId = omp_get_thread_num();
        mt19937 generator_local(rd());  // Create a local RNG for each thread
        double threadStartTime = omp_get_wtime(); // Measure thread start time

#pragma omp for
        for (int i = 0; i < numSamples; ++i) {
            double x = distribution(generator_local);
            double result = function(x);
            sum += result;
        }

        double threadEndTime = omp_get_wtime(); // Measure thread end time
        cout << "Thread " << threadId << " took " << (threadEndTime - threadStartTime) << " seconds." << endl;
    }

    double intervalLength = upperBound - lowerBound;
    double average = sum / numSamples;
    return average * intervalLength;
}

// Perform Monte Carlo Simulations
vector<double> performSimulations(double initialInvestment, double expectedReturn, double volatility, int numSimulations, int investmentPeriod) {
    random_device rd;
    mt19937 gen(rd());
    normal_distribution<double> dis(0.0, 1.0);

    vector<double> finalReturns(numSimulations);

#pragma omp parallel num_threads(NUM_THREADS)
    {
        int threadId = omp_get_thread_num();
        int numSimulationsPerThread = numSimulations / NUM_THREADS;
        int startSimulation = threadId * numSimulationsPerThread;
        int endSimulation = (threadId == 3) ? numSimulations : startSimulation + numSimulationsPerThread;

        mt19937 gen_local(rd());  // Create a local RNG for each thread

        for (int i = startSimulation; i < endSimulation; ++i) {
            double cumulativeReturn = 1.0;

            for (int j = 0; j < investmentPeriod; ++j) {
                double randomValue = dis(gen_local);
                double investmentReturn = expectedReturn + volatility * randomValue;
                cumulativeReturn *= (1.0 + investmentReturn);
            }

            finalReturns[i] = cumulativeReturn - 1.0;
        }
    }

    return finalReturns;
}

// Function to calculate risk metrics
void calculateRiskMetrics(const vector<double>& finalReturns, double& averageReturn, double& standardDeviation, double& minReturn, double& maxReturn) {
    double totalReturns = 0.0;
    minReturn = finalReturns[0];
    maxReturn = finalReturns[0];

#pragma omp parallel num_threads(NUM_THREADS)
    {
        int threadId = omp_get_thread_num();
        double privateTotalReturns = 0.0;
        double privateMinReturn = minReturn;
        double privateMaxReturn = maxReturn;

#pragma omp for
        for (int i = 0; i < finalReturns.size(); ++i) {
            double returnVal = finalReturns[i];
            privateTotalReturns += returnVal;
            privateMinReturn = min(privateMinReturn, returnVal);
            privateMaxReturn = max(privateMaxReturn, returnVal);
        }

#pragma omp critical
        {
            totalReturns += privateTotalReturns;
            minReturn = min(minReturn, privateMinReturn);
            maxReturn = max(maxReturn, privateMaxReturn);
        }
    }

    averageReturn = totalReturns / finalReturns.size();

    double variance = 0.0;

#pragma omp parallel num_threads(NUM_THREADS) reduction(+:variance)
    {
        int threadId = omp_get_thread_num();

#pragma omp for
        for (int i = 0; i < finalReturns.size(); ++i) {
            double returnVal = finalReturns[i];
            double diff = returnVal - averageReturn;
            variance += diff * diff;
        }
    }

    variance /= (finalReturns.size() - 1);
    standardDeviation = sqrt(variance);
}


// Function to calculate risk metrics for a subset of simulations
void calculateRiskMetricsPartial(const vector<double>& finalReturns, int start, int end, double& averageReturn, double& standardDeviation, double& minReturn, double& maxReturn) {
    double totalReturns = 0.0;
    minReturn = finalReturns[start];
    maxReturn = finalReturns[start];

    for (int i = start; i < end; ++i) {
        double returnVal = finalReturns[i];
        totalReturns += returnVal;
        minReturn = min(minReturn, returnVal);
        maxReturn = max(maxReturn, returnVal);
    }

    averageReturn = totalReturns / (end - start);

    double variance = 0.0;

    for (int i = start; i < end; ++i) {
        double returnVal = finalReturns[i];
        double diff = returnVal - averageReturn;
        variance += diff * diff;
    }

    variance /= (end - start - 1);
    standardDeviation = sqrt(variance);
}

void performRiskAssessment(const vector<double>& finalReturns, int numThreads, double initialInvestment, double expectedReturn, double volatility, int investmentPeriod, int numSimulations, double integrationResult, double elapsedTime) {
#pragma omp parallel num_threads(numThreads)
    {
        int threadId = omp_get_thread_num();
        int numSimulationsPerThread = finalReturns.size() / numThreads;
        int startSimulation = threadId * numSimulationsPerThread;
        int endSimulation = (threadId == numThreads - 1) ? finalReturns.size() : startSimulation + numSimulationsPerThread;

        double averageReturn, standardDeviation, minReturn, maxReturn;
        double threadStartTime = omp_get_wtime();

        calculateRiskMetricsPartial(finalReturns, startSimulation, endSimulation, averageReturn, standardDeviation, minReturn, maxReturn);

        double threadEndTime = omp_get_wtime();

        // Use a barrier to ensure all threads finish their assessments before proceeding
#pragma omp barrier

// Each thread displays its own results
#pragma omp critical
        {
            // Display Risk Assessment Results for each thread
            cout << endl << "Thread " << threadId << " Risk Assessment Results:" << endl;
            cout << "------------------------" << endl;
            cout << "Initial Investment: $" << initialInvestment << endl;
            cout << "Expected Annual Return: " << (expectedReturn * 100.0) << "%" << endl;
            cout << "Volatility (Annual Standard Deviation): " << (volatility * 100.0) << "%" << endl;
            cout << "Investment Period: " << investmentPeriod << " years" << endl;
            cout << "Number of Simulations: " << numSimulations << endl;
            cout << "------------------------" << endl;
            cout << "Average Return: " << (averageReturn * 100.0) << "%" << endl;
            cout << "Standard Deviation of Returns: " << (standardDeviation * 100.0) << "%" << endl;
            cout << "Minimum Return: " << (minReturn * 100.0) << "%" << endl;
            cout << "Maximum Return: " << (maxReturn * 100.0) << "%" << endl;
            cout << "Monte Carlo Integration Result: " << integrationResult << endl;
            cout << "Elapsed Time for Thread " << threadId << ": " << (threadEndTime - threadStartTime) << " seconds" << endl;
            cout << "------------------------" << endl;
        }
    }
}

int main() {
    double initialInvestment = 10000.0;
    double expectedReturn = 0.08;
    double volatility = 0.15;
    int numSimulations = 1000000;
    int investmentPeriod = 5;
    double lowerBound = 0.0;
    double upperBound = 1.0;
    int numSamples = 1000000;

    auto startTime = omp_get_wtime(); // Start overall time measurement

    vector<double> finalReturns = performSimulations(initialInvestment, expectedReturn, volatility, numSimulations, investmentPeriod);

    double integrationResult = monteCarloIntegration(functionToIntegrate, lowerBound, upperBound, numSamples);

    auto endTime = omp_get_wtime(); // End overall time measurement
    double elapsedTime = endTime - startTime; // Calculate elapsed time

    // Display Risk Assessment Results for each thread and overall results
    performRiskAssessment(finalReturns, NUM_THREADS, initialInvestment, expectedReturn, volatility, investmentPeriod, numSimulations, integrationResult, elapsedTime);

    system("pause");

    return 0;
}

