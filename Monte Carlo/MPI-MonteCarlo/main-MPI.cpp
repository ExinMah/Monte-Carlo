#include <iostream>
#include <random>
#include <vector>
#include <cmath>
#include <chrono>
#include <mpi.h>
#include <assert.h>

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
    for (int i = 0; i < numSamples; ++i) {
        double x = distribution(generator);
        sum += function(x);
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

    vector<double> finalReturns;

    for (int i = 0; i < numSimulations; ++i) {
        double cumulativeReturn = 1.0;

        for (int j = 0; j < investmentPeriod; ++j) {
            double randomValue = dis(gen);
            double investmentReturn = expectedReturn + volatility * randomValue;
            cumulativeReturn *= (1.0 + investmentReturn);
        }

        double finalReturn = cumulativeReturn - 1.0;
        finalReturns.push_back(finalReturn);
    }

    return finalReturns;
}

// Function to calculate risk metrics
void calculateRiskMetrics(const vector<double>& finalReturns, double& averageReturn, double& standardDeviation, double& minReturn, double& maxReturn) {
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

double findSmallest(const double* array, int numProcesses) {
    double smallest = array[0];  // Assume the first element is the smallest.

    for (int i = 1; i < numProcesses; ++i) {
        if (array[i] < smallest) {
            smallest = array[i];
        }
    }

    return smallest;
}

double findLargest(const double* array, int numProcesses) {
    double largest = array[0];  // Assume the first element is the largest.

    for (int i = 1; i < numProcesses; ++i) {
        if (array[i] > largest) {  // Check for a larger element.
            largest = array[i];
        }
    }

    return largest;
}

double computeSumData(const double* array, int numProcesses) {
    double sum = 0.f;
    int i;
    for (i = 0; i < numProcesses; i++) {
        sum += array[i];
    }

    return sum / numProcesses;
}

void displayResults(double initialInvestment, double expectedReturn, double volatility, int investmentPeriod, int numSimulations, double averageReturn, double standardDeviation, double minReturn, double maxReturn, double integrationResult, double elapsedTime) {
    cout << "Risk Assessment Results:" << endl;
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
    cout << "Elapsed Time: " << elapsedTime << " seconds" << endl;
}

int main(int argc, char** argv) {

    int numProcesses = 0, worldRank = 0;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcesses);

    double initialInvestment = 10000.0;
    double expectedReturn = 0.08;
    double volatility = 0.15;
    int numSimulations = 1000000;
    int investmentPeriod = 5;
    double lowerBound = 0.0;
    double upperBound = 1.0;
    int numSamples = 1000000;

    double startTime = MPI_Wtime();

    int numSimulationsPerProcess = numSimulations / numProcesses;
    vector<double> finalReturns(numSimulationsPerProcess);
    vector<double> subFinalReturns(numSimulationsPerProcess);

    MPI_Scatter(finalReturns.data(), numSimulationsPerProcess, MPI_FLOAT,
        subFinalReturns.data(), numSimulationsPerProcess, MPI_FLOAT, 0,
        MPI_COMM_WORLD);

    vector<double> computeSimulation = performSimulations(initialInvestment, expectedReturn, volatility, numSimulationsPerProcess, investmentPeriod);
    
    double averageReturn, standardDeviation, minReturn, maxReturn;
    calculateRiskMetrics(computeSimulation, averageReturn, standardDeviation, minReturn, maxReturn);

    double integrationResult = monteCarloIntegration(functionToIntegrate, lowerBound, upperBound, numSimulationsPerProcess);

    double* averageReturns = NULL, *standardDeviations = NULL, *minReturns = NULL, *maxReturns = NULL, *integrationResults = NULL;
    if (worldRank == 0) {
        averageReturns = (double*)malloc(sizeof(double) * numProcesses);
        standardDeviations = (double*)malloc(sizeof(double) * numProcesses);
        minReturns = (double*)malloc(sizeof(double) * numProcesses);
        maxReturns = (double*)malloc(sizeof(double) * numProcesses);
        integrationResults = (double*)malloc(sizeof(double) * numProcesses);

        assert(averageReturns != NULL);
        assert(standardDeviations != NULL);
        assert(minReturns != NULL);
        assert(maxReturns != NULL);
        assert(integrationResults != NULL);
    }

    MPI_Gather(&averageReturn, 1, MPI_DOUBLE, averageReturns, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gather(&standardDeviation, 1, MPI_DOUBLE, standardDeviations, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gather(&minReturn, 1, MPI_DOUBLE, minReturns, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gather(&maxReturn, 1, MPI_DOUBLE, maxReturns, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gather(&integrationResult, 1, MPI_DOUBLE, integrationResults, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    /*vector<double> computeSimulations;
    if (worldRank == 0) {
        computeSimulations.resize(numSimulations);
        assert(!computeSimulations.empty());
    }

    MPI_Gather(computeSimulation.data(), numSimulationsPerProcess, MPI_FLOAT, computeSimulations.data(), numSimulationsPerProcess, MPI_FLOAT, 0, MPI_COMM_WORLD);*/


    // START = BACKUP - IGNORE HERE //
    // auto startTime = steady_clock::now();

    // finalReturns = performSimulations(initialInvestment, expectedReturn, volatility, numSimulations, investmentPeriod);
    

    // auto endTime = steady_clock::now();
    // double elapsedTime = duration<double>(endTime - startTime).count();
    // END = BACKUP - IGNORE HERE //

    if (worldRank == 0)
    {
        double finalAverageReturn = computeSumData(averageReturns, numProcesses);
        double finalStandardDeviation = computeSumData(standardDeviations, numProcesses);
        double finalMinReturn = findSmallest(minReturns, numProcesses);
        double finalMaxReturn = findLargest(maxReturns, numProcesses);
        double finalIntegrationResult = computeSumData(integrationResults, numProcesses);
        
        double endTime = MPI_Wtime();
        double elapsedTime = endTime - startTime;

        displayResults(initialInvestment, expectedReturn, volatility, investmentPeriod, numSimulations, finalAverageReturn, finalStandardDeviation, finalMinReturn, finalMaxReturn, finalIntegrationResult, elapsedTime);
    
        free(averageReturns);
        free(standardDeviations);
        free(minReturns);
        free(maxReturns);
        free(integrationResults);
    }

    MPI_Finalize();

    system("pause");

    return 0;
}