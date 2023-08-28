#include <iostream>
#include <random>
#include <vector>
using namespace std;

int main() {
    double initialInvestment = 10000.0; // Initial investment amount
    double expectedReturn = 0.08; // Expected annual return
    double volatility = 0.15; // Annual volatility (standard deviation)
    int numSimulations = 10000; // Number of simulations
    int investmentPeriod = 5; // Investment period in years

    // Create a random number generator
    random_device rd;
    mt19937 gen(rd());
    normal_distribution<double> dis(0.0, 1.0);

    vector<double> finalReturns;

    // Perform the Monte Carlo simulations
    for (int i = 0; i < numSimulations; ++i) {
        double cumulativeReturn = 1.0;

        // Simulate investment returns for each year
        for (int j = 0; j < investmentPeriod; ++j) {
            double randomValue = dis(gen);
            double investmentReturn = expectedReturn + volatility * randomValue;
            cumulativeReturn *= (1.0 + investmentReturn);
        }

        double finalReturn = cumulativeReturn - 1.0;
        finalReturns.push_back(finalReturn);
    }

    // Calculate various risk metrics
    double averageReturn = 0.0;
    double totalReturns = 0.0;
    double minReturn = finalReturns[0];
    double maxReturn = finalReturns[0];

    for (const auto& returnVal : finalReturns) {
        totalReturns += returnVal;
        if (returnVal < minReturn) {
            minReturn = returnVal;
        }
        if (returnVal > maxReturn) {
            maxReturn = returnVal;
        }
    }

    averageReturn = totalReturns / numSimulations;
    double variance = 0.0;
    double standardDeviation = 0.0;

    for (const auto& returnVal : finalReturns) {
        variance += (returnVal - averageReturn) * (returnVal - averageReturn);
    }

    variance /= (numSimulations - 1);
    standardDeviation = sqrt(variance);

    // Display the results
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

    return 0;
}