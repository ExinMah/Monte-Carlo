#include <iostream>
#include <random>
#include <vector>
#include <chrono>
using namespace std;

// Function to perform Monte Carlo simulations
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

// Function to display the results
void displayResults(double initialInvestment, double expectedReturn, double volatility, int investmentPeriod, int numSimulations, double averageReturn, double standardDeviation, double minReturn, double maxReturn, double elapsedTime) {
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
    cout << "Elapsed Time: " << elapsedTime << " seconds" << endl;
}

int main() {
    double initialInvestment = 10000.0;
    double expectedReturn = 0.08;
    double volatility = 0.15;
    int numSimulations = 1000000;
    int investmentPeriod = 5;

    // Start the timer
    auto startTime = chrono::steady_clock::now();

    vector<double> finalReturns = performSimulations(initialInvestment, expectedReturn, volatility, numSimulations, investmentPeriod);

    double averageReturn, standardDeviation, minReturn, maxReturn;
    calculateRiskMetrics(finalReturns, averageReturn, standardDeviation, minReturn, maxReturn);

    // Stop the timer
    auto endTime = chrono::steady_clock::now();
    double elapsedTime = chrono::duration<double>(endTime - startTime).count();

    displayResults(initialInvestment, expectedReturn, volatility, investmentPeriod, numSimulations, averageReturn, standardDeviation, minReturn, maxReturn, elapsedTime);

    return 0;
}